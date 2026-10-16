"""Belgelerden düz metin çıkarma ve parçalama: PDF, DOCX, TXT, MD, HTML."""
from __future__ import annotations

import io
import re
import zipfile

from lxml import etree, html as lxml_html

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


class UnsupportedFile(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext == ".pdf":
        text = _pdf(data)
    elif ext == ".docx":
        text = _docx(data)
    elif ext in (".html", ".htm"):
        text = lxml_html.fromstring(data).text_content()
    elif ext in (".txt", ".md"):
        text = _decode(data)
    else:
        raise UnsupportedFile(f"desteklenmeyen dosya türü: {ext or filename} (desteklenen: {', '.join(sorted(SUPPORTED))})")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1254", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "ignore")


def _pdf(data: bytes) -> str:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        pages = [p.get_text("text") for p in doc]
    text = "\n\n".join(pages)
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # satır sonu tirelemesi


def _docx(data: bytes) -> str:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = etree.fromstring(z.read("word/document.xml"))
    paras = ["".join(p.itertext()) for p in root.iterfind(".//w:p", ns)]
    return "\n".join(p for p in paras if p.strip())


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Paragraf sınırında parçalar; tek paragraf çok uzunsa cümle sınırında böler."""
    if len(text) <= max_chars:
        return [text] if text else []
    parts, buf = [], ""
    for para in text.split("\n"):
        while len(para) > max_chars:
            cut = para.rfind(". ", 0, max_chars) + 1 or max_chars
            if buf:
                parts.append(buf)
                buf = ""
            parts.append(para[:cut])
            para = para[cut:].lstrip()
        if len(buf) + len(para) + 1 > max_chars and buf:
            parts.append(buf)
            buf = ""
        buf = f"{buf}\n{para}" if buf else para
    if buf:
        parts.append(buf)
    return parts
