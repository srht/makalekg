"""Proje deposu: her proje data/projects/<slug>/ altında.

  project.json      ad, açıklama
  ontology.json     ontoloji
  documents.json    belge listesi ve durumları
  files/<id>.<ext>  yüklenen özgün dosya
  texts/<id>.txt    çıkarılan düz metin
  graph.json        varlıklar, ilişkiler, reddedilen çıkarımlar

Graf, belge belge kurulur: bir belge yeniden işlendiğinde önce o belgenin katkıları silinir, sonra
yenileri eklenir (idempotent). Neo4j ayarlıysa her değişiklikten sonra proje Neo4j'e aynalanır.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import DATA_DIR, env
from .ontology import Ontology, ascii_ident

log = logging.getLogger(__name__)
_locks: dict[str, threading.RLock] = {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(name: str) -> str:
    s = ascii_ident(name).lower().strip("_")
    return re.sub(r"_+", "-", s)[:40] or "proje"


def norm_name(name: str) -> str:
    t = unicodedata.normalize("NFKC", name).casefold().strip()
    return re.sub(r"\s+", " ", t).strip(" .,;:\"'()[]")


def projects_root() -> Path:
    return DATA_DIR / "projects"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def list_projects() -> list[dict[str, Any]]:
    root = projects_root()
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if (d / "project.json").exists():
            p = Project(d.name)
            meta = p.meta()
            g = p.graph()
            out.append({**meta, "slug": d.name, "n_docs": len(p.documents()),
                        "n_entities": len(g["entities"]), "n_relations": len(g["relations"])})
    return sorted(out, key=lambda m: m.get("created_at", ""), reverse=True)


def create_project(name: str, description: str = "") -> "Project":
    base = slugify(name)
    slug, i = base, 2
    while (projects_root() / slug).exists():
        slug, i = f"{base}-{i}", i + 1
    p = Project(slug)
    _write_json(p.dir / "project.json", {"name": name.strip(), "description": description.strip(), "created_at": now()})
    p.save_ontology(Ontology())
    return p


class Project:
    def __init__(self, slug: str):
        if not re.fullmatch(r"[a-z0-9][a-z0-9\-]*", slug):
            raise ValueError("geçersiz proje adı")
        self.slug = slug
        self.dir = projects_root() / slug
        self.lock = _locks.setdefault(slug, threading.RLock())

    def exists(self) -> bool:
        return (self.dir / "project.json").exists()

    # ------------------------------------------------------------ meta / ontoloji

    def meta(self) -> dict[str, Any]:
        return _read_json(self.dir / "project.json", {})

    def ontology(self) -> Ontology:
        return Ontology.model_validate(_read_json(self.dir / "ontology.json", {}))

    def save_ontology(self, onto: Ontology) -> None:
        _write_json(self.dir / "ontology.json", onto.cleaned().model_dump())

    def suggestion(self) -> Ontology | None:
        data = _read_json(self.dir / "ontology_suggestion.json", None)
        return Ontology.model_validate(data) if data else None

    def save_suggestion(self, onto: Ontology | None) -> None:
        path = self.dir / "ontology_suggestion.json"
        if onto is None:
            path.unlink(missing_ok=True)
        else:
            _write_json(path, onto.cleaned().model_dump())

    # ------------------------------------------------------------ belgeler

    def documents(self) -> list[dict[str, Any]]:
        return _read_json(self.dir / "documents.json", [])

    def document(self, doc_id: str) -> dict[str, Any] | None:
        return next((d for d in self.documents() if d["id"] == doc_id), None)

    def _save_documents(self, docs: list[dict[str, Any]]) -> None:
        _write_json(self.dir / "documents.json", docs)

    def add_document(self, filename: str, data: bytes, text: str) -> dict[str, Any]:
        doc_id = uuid.uuid4().hex[:10]
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
        (self.dir / "files").mkdir(parents=True, exist_ok=True)
        (self.dir / "texts").mkdir(parents=True, exist_ok=True)
        (self.dir / "files" / f"{doc_id}.{ext}").write_bytes(data)
        (self.dir / "texts" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        doc = {"id": doc_id, "filename": filename, "chars": len(text), "status": "new", "uploaded_at": now(),
               "n_entities": 0, "n_relations": 0, "n_rejected": 0, "error": None, "processed_at": None}
        with self.lock:
            docs = self.documents()
            docs.append(doc)
            self._save_documents(docs)
        return doc

    def update_document(self, doc_id: str, **fields: Any) -> None:
        with self.lock:
            docs = self.documents()
            for d in docs:
                if d["id"] == doc_id:
                    d.update(fields)
            self._save_documents(docs)

    def text(self, doc_id: str) -> str:
        return (self.dir / "texts" / f"{doc_id}.txt").read_text(encoding="utf-8")

    def delete_document(self, doc_id: str) -> None:
        with self.lock:
            self.remove_contributions(doc_id)
            self._save_documents([d for d in self.documents() if d["id"] != doc_id])
            for f in list((self.dir / "files").glob(f"{doc_id}.*")) + [self.dir / "texts" / f"{doc_id}.txt"]:
                f.unlink(missing_ok=True)
        self.sync_neo4j()

    # ------------------------------------------------------------ graf

    def graph(self) -> dict[str, Any]:
        return _read_json(self.dir / "graph.json", {"entities": {}, "relations": {}, "rejected": []})

    def _save_graph(self, g: dict[str, Any]) -> None:
        _write_json(self.dir / "graph.json", g)

    @staticmethod
    def entity_key(etype: str, name: str) -> str:
        return f"{etype}::{norm_name(name)}"

    def remove_contributions(self, doc_id: str) -> None:
        with self.lock:
            g = self.graph()
            for key in list(g["entities"]):
                e = g["entities"][key]
                e["mentions"] = [m for m in e["mentions"] if m["doc_id"] != doc_id]
            for key in list(g["relations"]):
                r = g["relations"][key]
                r["sources"] = [s for s in r["sources"] if s["doc_id"] != doc_id]
                if not r["sources"]:
                    del g["relations"][key]
            used = {r["subject"] for r in g["relations"].values()} | {r["object"] for r in g["relations"].values()}
            for key in list(g["entities"]):
                if not g["entities"][key]["mentions"] and key not in used:
                    del g["entities"][key]
            g["rejected"] = [x for x in g["rejected"] if x.get("doc_id") != doc_id]
            self._save_graph(g)

    def add_extraction(self, doc_id: str, entities: list[dict[str, Any]], relations: list[dict[str, Any]],
                       rejected: list[dict[str, Any]]) -> None:
        """entities: {type, name, mention, description}; relations: {subject_key, predicate, object_key, sentence, confidence}"""
        with self.lock:
            g = self.graph()
            for e in entities:
                key = self.entity_key(e["type"], e["name"])
                node = g["entities"].setdefault(key, {"id": uuid.uuid4().hex[:10], "name": e["name"], "type": e["type"],
                                                      "description": None, "mentions": []})
                if not node.get("description") and e.get("description"):
                    node["description"] = e["description"]
                if not any(m["doc_id"] == doc_id and m["text"] == e["mention"] for m in node["mentions"]):
                    node["mentions"].append({"doc_id": doc_id, "text": e["mention"]})
            for r in relations:
                key = f"{r['subject_key']}|{r['predicate']}|{r['object_key']}"
                rel = g["relations"].setdefault(key, {"id": uuid.uuid4().hex[:10], "subject": r["subject_key"],
                                                      "predicate": r["predicate"], "object": r["object_key"], "sources": []})
                if not any(s["doc_id"] == doc_id and s["sentence"] == r["sentence"] for s in rel["sources"]):
                    rel["sources"].append({"doc_id": doc_id, "sentence": r["sentence"], "confidence": r["confidence"]})
            g["rejected"] += [{**x, "doc_id": doc_id} for x in rejected]
            self._save_graph(g)

    def stats(self) -> dict[str, Any]:
        g = self.graph()
        docs = self.documents()
        by_type: dict[str, int] = {}
        for e in g["entities"].values():
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        return {"n_docs": len(docs), "n_done": sum(1 for d in docs if d["status"] == "done"),
                "n_entities": len(g["entities"]), "n_relations": len(g["relations"]),
                "n_rejected": len(g["rejected"]), "by_type": by_type}

    # ------------------------------------------------------------ Neo4j aynası

    def sync_neo4j(self) -> str | None:
        """Projeyi Neo4j'e yazar (proje düğümlerini silip yeniden oluşturur). Neo4j yoksa sessizce geçer."""
        if (env("GRAPH_BACKEND", "neo4j") or "").lower() != "neo4j" or not env("NEO4J_URI"):
            return None
        try:
            from neo4j import GraphDatabase
        except ImportError:
            return None
        g = self.graph()
        docs = {d["id"]: d for d in self.documents()}
        try:
            driver = GraphDatabase.driver(env("NEO4J_URI"), auth=(env("NEO4J_USER", "neo4j"), env("NEO4J_PASSWORD", "")))
            with driver, driver.session(database=env("NEO4J_DATABASE")) as s:
                s.run("MATCH (n {project: $p}) DETACH DELETE n", p=self.slug)
                s.run("UNWIND $docs AS d MERGE (n:Document {project: $p, uid: d.id}) SET n.filename = d.filename",
                      p=self.slug, docs=[{"id": k, "filename": v["filename"]} for k, v in docs.items()])
                by_label: dict[str, list[dict[str, Any]]] = {}
                for key, e in g["entities"].items():
                    by_label.setdefault(ascii_ident(e["type"]), []).append(
                        {"uid": key, "name": e["name"], "type": e["type"], "description": e.get("description"),
                         "docs": sorted({m["doc_id"] for m in e["mentions"]})})
                for label, rows in by_label.items():
                    s.run(f"UNWIND $rows AS r CREATE (n:Entity:`{label}` {{project: $p, uid: r.uid}}) "
                          "SET n.name = r.name, n.type = r.type, n.description = r.description "
                          "WITH n, r UNWIND r.docs AS d MATCH (doc:Document {project: $p, uid: d}) "
                          "MERGE (n)-[:MENTIONED_IN]->(doc)", p=self.slug, rows=rows)
                by_type: dict[str, list[dict[str, Any]]] = {}
                for r in g["relations"].values():
                    by_type.setdefault(ascii_ident(r["predicate"]).upper(), []).append(
                        {"s": r["subject"], "o": r["object"], "name": r["predicate"],
                         "sentences": [x["sentence"] for x in r["sources"]],
                         "docs": [x["doc_id"] for x in r["sources"]]})
                for rtype, rows in by_type.items():
                    s.run(f"UNWIND $rows AS r MATCH (a:Entity {{project: $p, uid: r.s}}), (b:Entity {{project: $p, uid: r.o}}) "
                          f"CREATE (a)-[x:`{rtype}`]->(b) SET x.name = r.name, x.sentences = r.sentences, x.docs = r.docs",
                          p=self.slug, rows=rows)
            return None
        except Exception as exc:  # Neo4j erişilemese de yerel graf geçerlidir
            log.warning("Neo4j'e yazılamadı: %s", exc)
            return f"{type(exc).__name__}: {exc}"
