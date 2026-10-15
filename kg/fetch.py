"""Tam metin ve bölümleme.

Sıra: Europe PMC açık erişim JATS XML → yerel/indirilebilir PDF (pymupdf) → yalnızca özet
(fulltext=false). Şekil açıklamaları (figure legends) ayrı bölümler olarak tutulur, çünkü n ve p
değerleri büyük ölçüde oradadır.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Literal, Optional

from lxml import etree
from pydantic import BaseModel

from .config import DATA_DIR
from .http import Http

log = logging.getLogger(__name__)
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"

SectionKind = Literal["abstract", "body", "methods", "legend"]


class Section(BaseModel):
    name: str
    kind: SectionKind
    text: str
    figure: Optional[str] = None  # legend bölümleri için "Fig. 3", "Extended Data Fig. 2"


class Document(BaseModel):
    doi: str
    fulltext: bool
    source: Literal["jats", "pdf", "abstract", "text"]
    sections: list[Section]

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)

    def legends(self) -> list[Section]:
        return [s for s in self.sections if s.kind == "legend"]

    def legend_for(self, figure: str | None) -> Section | None:
        key = figure_key(figure)
        if not key:
            return None
        return next((s for s in self.legends() if figure_key(s.figure) == key), None)


FIG_RE = re.compile(r"(extended\s+data\s+|supplementary\s+)?fig(?:ure)?\.?\s*(s?\d+)", re.I)


def figure_key(figure: str | None) -> str | None:
    """'Figure 3', 'Fig. 3d' → 'fig3'; 'Extended Data Fig. 2' → 'edfig2'."""
    if not figure:
        return None
    m = FIG_RE.search(figure)
    if not m:
        return None
    prefix = "ed" if m.group(1) and m.group(1).lower().startswith("extended") else \
        ("sup" if m.group(1) else "")
    return f"{prefix}fig{m.group(2).lower()}"


def _clean(text: str) -> str:
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


METHOD_TITLES = re.compile(r"method|materials|experimental procedures|statistic", re.I)

# ---------------------------------------------------------------- JATS


def parse_jats(xml: str | bytes, doi: str) -> Document:
    root = etree.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml,
                            parser=etree.XMLParser(recover=True, huge_tree=True))
    sections: list[Section] = []

    def text_of(el) -> str:
        # Kaynakça işaretlerini ve tablo içeriğini çıkar, kalan metni birleştir
        el = etree.fromstring(etree.tostring(el))
        for bad in el.xpath(".//table-wrap|.//fig|.//disp-formula"):
            bad.getparent().remove(bad)
        return _clean(" ".join(el.itertext()))

    for ab in root.xpath("//front//abstract"):
        t = text_of(ab)
        if t:
            sections.append(Section(name="Abstract", kind="abstract", text=t))

    body = root.find(".//body")
    if body is not None:
        for sec in body.xpath("./sec"):
            title_el = sec.find("title")
            title = _clean("".join(title_el.itertext())) if title_el is not None else "Section"
            kind: SectionKind = "methods" if METHOD_TITLES.search(title) else "body"
            subsecs = sec.xpath("./sec")
            if subsecs:
                own = [p for p in sec.xpath("./p")]
                if own:
                    sections.append(Section(name=title, kind=kind, text=_clean("\n".join(text_of(p) for p in own))))
                for sub in subsecs:
                    st = sub.find("title")
                    sub_title = _clean("".join(st.itertext())) if st is not None else ""
                    sections.append(Section(name=f"{title} / {sub_title}".strip(" /"), kind=kind, text=text_of(sub)))
            else:
                sections.append(Section(name=title, kind=kind, text=text_of(sec)))
        loose = body.xpath("./p")
        if loose:
            sections.append(Section(name="Body", kind="body", text=_clean("\n".join(text_of(p) for p in loose))))

    for back_sec in root.xpath("//back/sec"):
        title_el = back_sec.find("title")
        title = _clean("".join(title_el.itertext())) if title_el is not None else "Back matter"
        if METHOD_TITLES.search(title):
            sections.append(Section(name=title, kind="methods", text=text_of(back_sec)))

    for fig in root.xpath("//fig"):
        label_el = fig.find("label")
        label = _clean("".join(label_el.itertext())) if label_el is not None else None
        cap = fig.find("caption")
        if cap is None:
            continue
        cap_text = _clean(" ".join(cap.itertext()))
        if cap_text:
            sections.append(Section(name=f"Legend {label or ''}".strip(), kind="legend",
                                    text=f"{label + ' | ' if label else ''}{cap_text}", figure=label))

    sections = [s for s in sections if s.text]
    return Document(doi=doi, fulltext=True, source="jats", sections=sections)


# ---------------------------------------------------------------- PDF / düz metin

HEADING_RE = re.compile(
    r"^(abstract|introduction|results|discussion|conclusions?|methods|online methods|"
    r"materials and methods|statistics(?: and reproducibility)?|references|acknowledg(?:e)?ments|"
    r"data availability|code availability|reporting summary|author contributions|competing interests)\s*$",
    re.I,
)
LEGEND_START_RE = re.compile(
    r"^((?:extended\s+data\s+|supplementary\s+)?fig(?:ure)?\.?\s*s?\d+)\s*[|.:]\s*", re.I)
STOP_SECTIONS = {"references", "acknowledgements", "acknowledgments", "author contributions",
                 "competing interests", "reporting summary", "data availability", "code availability"}


def segment_plain_text(text: str, doi: str, source: Literal["pdf", "text"] = "text") -> Document:
    """Düz metni başlık ve şekil açıklamalarına göre bölümler."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 1:  # PDF çıktısında boş satır yoksa satır bazlı böl
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    sections: list[Section] = []
    current_name, current_kind, buf = "Front", "body", []
    legend: Section | None = None
    skipping = False

    def flush() -> None:
        nonlocal buf
        if buf and not skipping:
            sections.append(Section(name=current_name, kind=current_kind, text=_clean("\n".join(buf))))
        buf = []

    for para in paragraphs:
        first_line = para.split("\n", 1)[0].strip()
        h = HEADING_RE.match(first_line)
        if h:
            if legend:
                sections.append(legend)
                legend = None
            flush()
            current_name = h.group(1).strip().title()
            low = current_name.lower()
            skipping = low in STOP_SECTIONS
            current_kind = "abstract" if low == "abstract" else ("methods" if METHOD_TITLES.search(low) else "body")
            rest = para[len(first_line):].strip()
            if rest:
                buf.append(rest)
            continue
        m = LEGEND_START_RE.match(para)
        if m:
            if legend:
                sections.append(legend)
            label = _clean(m.group(1))
            legend = Section(name=f"Legend {label}", kind="legend", text=_clean(para), figure=label)
            continue
        if legend and len(para) > 0 and not skipping and _looks_like_legend_continuation(para):
            legend.text += "\n" + _clean(para)
            continue
        if legend:
            sections.append(legend)
            legend = None
        buf.append(para)
    if legend:
        sections.append(legend)
    flush()
    return Document(doi=doi, fulltext=True, source=source, sections=[s for s in sections if s.text])


def _looks_like_legend_continuation(para: str) -> bool:
    # Panel açıklamaları ("d, ...", "(e) ...") veya istatistik satırları legend'in devamıdır
    return bool(re.match(r"^(\(?[a-z]\)?[,.)]\s|[a-z]–[a-z][,.]\s|n\s*=|data are|statistical|p\s*[<=])", para, re.I))


def parse_pdf(path: Path, doi: str) -> Document:
    import pymupdf

    with pymupdf.open(path) as pdf:
        pages = [page.get_text("text") for page in pdf]
    text = "\n\n".join(pages)
    # satır sonu tirelemesini düzelt
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return segment_plain_text(text, doi, source="pdf")


# ---------------------------------------------------------------- üst düzey


class Fetcher:
    def __init__(self, http: Http | None = None, raw_dir: Path | None = None):
        self.http = http or Http()
        self.raw_dir = raw_dir or DATA_DIR / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _slug(self, doi: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", doi.lower()).strip("_")

    def fetch(self, doi: str, pmcid: str | None = None, pdf_path: str | Path | None = None,
              abstract: str | None = None) -> Document:
        cache = self.raw_dir / f"{self._slug(doi)}.json"
        if cache.exists():
            return Document.model_validate_json(cache.read_text(encoding="utf-8"))

        doc: Document | None = None
        if pmcid:
            try:
                xml = self.http.get(f"{EPMC}/{pmcid}/fullTextXML", as_text=True)
                if xml:
                    doc = parse_jats(xml, doi)
                    if not any(s.kind in ("body", "legend") for s in doc.sections):
                        doc = None
            except ConnectionError as exc:
                log.warning("JATS alınamadı (%s): %s", doi, exc)
        if doc is None and pdf_path and Path(pdf_path).exists():
            doc = parse_pdf(Path(pdf_path), doi)
        if doc is None:
            if abstract is None:
                abstract = self._abstract(doi)
            doc = Document(doi=doi, fulltext=False, source="abstract",
                           sections=[Section(name="Abstract", kind="abstract", text=abstract)] if abstract else [])
        cache.write_text(doc.model_dump_json(indent=1), encoding="utf-8")
        return doc

    def _abstract(self, doi: str) -> str | None:
        try:
            data = self.http.get(f"{EPMC}/search", params={"query": f'DOI:"{doi}"', "format": "json",
                                                           "resultType": "core", "pageSize": 1})
            results = ((data or {}).get("resultList") or {}).get("result") or []
            if results:
                return re.sub(r"<[^>]+>", "", results[0].get("abstractText") or "") or None
        except ConnectionError:
            pass
        return None


def save_document(doc: Document, path: Path) -> None:
    path.write_text(json.dumps(doc.model_dump(), ensure_ascii=False, indent=1), encoding="utf-8")
