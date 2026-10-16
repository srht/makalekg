"""Ontoloji: projede hangi varlık türlerinin ve hangi ilişki türlerinin çıkarılacağı.

Kullanıcı tanımlar (veya şablondan / belgeden önerilen ontolojiyle başlar). LLM'den istenen JSON
şeması bu ontolojiden üretilir: varlık türleri ve ilişki adları birer seçenek listesi (enum) olur,
böylece model ontoloji dışında tür uyduramaz.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Literal, Optional

from pydantic import BaseModel, Field, create_model, field_validator

_TR = str.maketrans("çğıöşüÇĞİÖŞÜâîûÂÎÛ", "cgiosuCGIOSUaiuAIU")


def ascii_ident(text: str) -> str:
    """Neo4j etiketi/ilişki tipi için güvenli ASCII tanımlayıcı."""
    t = unicodedata.normalize("NFKD", text.translate(_TR)).encode("ascii", "ignore").decode()
    t = re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")
    if not t or t[0].isdigit():
        t = "T_" + t
    return t


def tr_upper(text: str) -> str:
    """Büyük harfe çevirir; Türkçe karakter içeren adlarda Türkçe kuralı uygular (i → İ, ı → I).
    İngilizce standart ontoloji adları (ör. isPartOf) olağan biçimde büyütülür."""
    if any(c in text for c in "çğıöşüÇĞİÖŞÜ"):
        return text.replace("i", "İ").replace("ı", "I").upper()
    return text.upper()


class EntityType(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    description: str = ""
    uri: Optional[str] = None  # standart ontolojideki karşılığı (ör. https://schema.org/Person)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class RelationType(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    description: str = ""
    source: list[str] = Field(default_factory=list)  # boş = her tür
    target: list[str] = Field(default_factory=list)
    uri: Optional[str] = None

    @field_validator("name")
    @classmethod
    def _norm(cls, v: str) -> str:
        return tr_upper(re.sub(r"\s+", "_", v.strip()))


class Ontology(BaseModel):
    entity_types: list[EntityType] = Field(default_factory=list)
    relation_types: list[RelationType] = Field(default_factory=list)
    domain: str = ""  # alanın kısa tanımı; istemlere bağlam olarak girer

    def type_names(self) -> list[str]:
        return [t.name for t in self.entity_types]

    def relation(self, name: str) -> RelationType | None:
        return next((r for r in self.relation_types if r.name == name), None)

    def is_ready(self) -> bool:
        return bool(self.entity_types)

    def cleaned(self) -> "Ontology":
        """Boş/yinelenen satırları atar, ilişkilerdeki bilinmeyen tür adlarını temizler."""
        seen, ents = set(), []
        for t in self.entity_types:
            if t.name and t.name.lower() not in seen:
                seen.add(t.name.lower())
                ents.append(t)
        names = {t.name for t in ents}
        rels, rseen = [], set()
        for r in self.relation_types:
            if not r.name or r.name in rseen:
                continue
            rseen.add(r.name)
            rels.append(r.model_copy(update={"source": [s for s in r.source if s in names],
                                             "target": [s for s in r.target if s in names]}))
        return Ontology(entity_types=ents, relation_types=rels, domain=self.domain.strip())

    def prompt_text(self) -> str:
        lines = ["ENTITY TYPES:"]
        for t in self.entity_types:
            lines.append(f"- {t.name}: {t.description or '(no description)'}")
        lines.append("\nRELATION TYPES (subject → object):")
        if not self.relation_types:
            lines.append("- (none defined; return an empty relations list)")
        for r in self.relation_types:
            src = " | ".join(r.source) or "any type"
            tgt = " | ".join(r.target) or "any type"
            lines.append(f"- {r.name}: [{src}] → [{tgt}]. {r.description}")
        return "\n".join(lines)

    def extraction_schema(self) -> type[BaseModel]:
        """Bu ontolojiye özel çıkarım şeması (Pydantic)."""
        types = tuple(self.type_names()) or ("Thing",)
        TypeLit = Literal[types]  # type: ignore[valid-type]
        rel_names = tuple(r.name for r in self.relation_types) or ("RELATED_TO",)
        RelLit = Literal[rel_names]  # type: ignore[valid-type]
        Ent = create_model(
            "ExtractedEntity",
            name=(str, Field(description="Canonical name of the entity")),
            type=(TypeLit, Field(description="One of the ontology entity types")),
            mention=(str, Field(description="The entity exactly as written in the text (verbatim span)")),
            description=(Optional[str], Field(None, description="One short sentence, in Turkish")),
        )
        Rel = create_model(
            "ExtractedRelation",
            subject=(str, Field(description="Name of the subject entity (must be in entities)")),
            predicate=(RelLit, Field(description="One of the ontology relation types")),
            object=(str, Field(description="Name of the object entity (must be in entities)")),
            source_sentence=(str, Field(description="The sentence stating the relation, copied VERBATIM, max 400 chars")),
            confidence=(float, Field(description="0-1")),
        )
        return create_model("ExtractionResult", entities=(list[Ent], ...), relations=(list[Rel], ...))


class SuggestedOntology(BaseModel):
    """Belgeden ontoloji önerisi için LLM çıktısı."""
    domain: str
    entity_types: list[EntityType]
    relation_types: list[RelationType]


# ---------------------------------------------------------------- hazır şablonlar

def _o(domain: str, ents: list[tuple[str, str]], rels: list[tuple[str, list[str], list[str], str]]) -> Ontology:
    return Ontology(domain=domain,
                    entity_types=[EntityType(name=n, description=d) for n, d in ents],
                    relation_types=[RelationType(name=n, source=s, target=t, description=d) for n, s, t, d in rels])


def _schema_org() -> Ontology:
    S = "https://schema.org/"
    ents = [("Kişi", "Person", "Gerçek bir kişi"), ("Kurum", "Organization", "Şirket, okul, devlet kurumu, örgüt"),
            ("Yer", "Place", "Ülke, şehir, bina, bölge"), ("Olay", "Event", "Belirli zamanda gerçekleşen olay"),
            ("Eser", "CreativeWork", "Kitap, makale, rapor, yazılım vb."), ("Ürün", "Product", "Ürün veya hizmet")]
    rels = [("ÇALIŞIR", "worksFor", ["Kişi"], ["Kurum"], "kişi kurumda çalışır"),
            ("ÜYESİDİR", "memberOf", ["Kişi", "Kurum"], ["Kurum"], "üyedir"),
            ("BULUNUR", "location", ["Kurum", "Olay"], ["Yer"], "konumu"),
            ("YAZAR", "author", ["Eser"], ["Kişi", "Kurum"], "eserin yazarı (eser → yazar)"),
            ("KURUCUSU", "founder", ["Kurum"], ["Kişi", "Kurum"], "kurumun kurucusu"),
            ("ÜRETİCİSİ", "manufacturer", ["Ürün"], ["Kurum"], "ürünün üreticisi"),
            ("KATILIMCI", "attendee", ["Olay"], ["Kişi", "Kurum"], "olayın katılımcısı"),
            ("HAKKINDADIR", "about", ["Eser"], [], "eserin konusu")]
    return Ontology(domain="Genel (schema.org eşlemeli)",
                    entity_types=[EntityType(name=n, description=d, uri=S + u) for n, u, d in ents],
                    relation_types=[RelationType(name=n, description=d, source=a, target=b, uri=S + u)
                                    for n, u, a, b, d in rels])


TEMPLATES: dict[str, tuple[str, Ontology]] = {
    "schema_org": ("schema.org (standart, genel)", _schema_org()),
    "genel": ("Genel (her tür belge)", _o(
        "Genel belgeler",
        [("Kişi", "Gerçek bir kişi"), ("Kurum", "Şirket, üniversite, devlet kurumu, örgüt"),
         ("Yer", "Ülke, şehir, bölge, mekân"), ("Olay", "Belirli bir zamanda olan olay"),
         ("Kavram", "Soyut fikir, terim, konu"), ("Eser", "Kitap, makale, yasa, rapor gibi belge veya ürün")],
        [("ÇALIŞIR", ["Kişi"], ["Kurum"], "kişi kurumda çalışır/üyedir"),
         ("BULUNUR", ["Kurum", "Olay"], ["Yer"], "bir yerde bulunur/gerçekleşir"),
         ("KATILIR", ["Kişi", "Kurum"], ["Olay"], "olaya katılır"),
         ("YAZAR", ["Kişi", "Kurum"], ["Eser"], "eseri yazar/üretir"),
         ("ETKİLER", [], [], "biri diğerini etkiler"),
         ("İLGİLİDİR", [], [], "aralarında açık bir ilişki vardır")])),
    "akademik": ("Akademik makale / tez", _o(
        "Akademik araştırma",
        [("Kavram", "Araştırılan kavram, değişken veya olgu"), ("Teori", "Kuram, model, yaklaşım"),
         ("Yöntem", "Araştırma yöntemi veya tekniği"), ("Veri seti", "Kullanılan veri kaynağı veya örneklem"),
         ("Bulgu", "Çalışmanın vardığı belirli sonuç"), ("Araştırmacı", "Yazar veya atıf yapılan kişi"),
         ("Kurum", "Üniversite, enstitü, fon kuruluşu")],
        [("KULLANIR", ["Bulgu", "Teori"], ["Yöntem", "Veri seti"], "bulgu/çalışma bu yöntemi veya veriyi kullanır"),
         ("DESTEKLER", ["Bulgu"], ["Teori", "Kavram"], "bulgu bir kuramı veya ilişkiyi destekler"),
         ("ÇÜRÜTÜR", ["Bulgu"], ["Teori", "Kavram"], "bulgu bir kuramla çelişir"),
         ("ETKİLER", ["Kavram"], ["Kavram"], "bir değişken diğerini etkiler"),
         ("ÖNERİR", ["Araştırmacı"], ["Teori", "Yöntem"], "kişi kuramı/yöntemi önerir"),
         ("BAĞLIDIR", ["Araştırmacı"], ["Kurum"], "araştırmacı kuruma bağlıdır")])),
    "biyomedikal": ("Biyomedikal", _o(
        "Biyomedikal araştırma",
        [("Gen/Protein", "Gen veya protein"), ("Hastalık", "Hastalık veya bozukluk"),
         ("Müdahale", "İlaç, tedavi, deneysel müdahale"), ("Hücre tipi", "Hücre veya doku tipi"),
         ("Organizma", "İnsan, fare vb."), ("Sonuç ölçümü", "Ölçülen sonuç (test, biyobelirteç)")],
        [("ARTIRIR", [], [], "artırır / aktive eder"), ("AZALTIR", [], [], "azaltır / baskılar"),
         ("TEDAVİ_EDER", ["Müdahale"], ["Hastalık"], "tedavi eder / iyileştirir"),
         ("NEDEN_OLUR", [], ["Hastalık", "Sonuç ölçümü"], "neden olur"),
         ("GEREKTİRİR", [], ["Gen/Protein"], "etki bu gene/proteine bağlıdır"),
         ("İFADE_EDİLİR", ["Gen/Protein"], ["Hücre tipi"], "bu hücrede ifade edilir")])),
    "uluslararasi": ("Uluslararası ilişkiler", _o(
        "Uluslararası ilişkiler",
        [("Devlet", "Ülke / devlet"), ("Uluslararası örgüt", "BM, NATO, AB vb."), ("Lider", "Devlet veya örgüt lideri"),
         ("Antlaşma", "Antlaşma, anlaşma, sözleşme"), ("Olay", "Çatışma, kriz, zirve, seçim"),
         ("Politika", "Politika, yaptırım, doktrin"), ("Bölge", "Coğrafi bölge")],
        [("MÜTTEFİKTİR", ["Devlet"], ["Devlet"], "ittifak / işbirliği"),
         ("KARŞITTIR", ["Devlet", "Uluslararası örgüt"], ["Devlet", "Uluslararası örgüt"], "çatışma / rekabet"),
         ("YAPTIRIM_UYGULAR", ["Devlet", "Uluslararası örgüt"], ["Devlet"], "yaptırım uygular"),
         ("İMZALAR", ["Devlet"], ["Antlaşma"], "antlaşmayı imzalar"),
         ("ÜYESİDİR", ["Devlet"], ["Uluslararası örgüt"], "örgüte üyedir"),
         ("YÖNETİR", ["Lider"], ["Devlet", "Uluslararası örgüt"], "yönetir"),
         ("TARAFTIR", ["Devlet", "Uluslararası örgüt"], ["Olay"], "olayın tarafıdır"),
         ("BULUNUR", ["Olay"], ["Bölge", "Devlet"], "olay burada gerçekleşir")])),
}
