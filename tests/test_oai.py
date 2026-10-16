"""OAI-PMH (DSpace) çekme: sahte arşiv sunucusuyla."""
import httpx
import pytest

from kg.http import Http
from kg.okg import oai
from kg.okg.store import create_project

HEAD = '<?xml version="1.0" encoding="UTF-8"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
DC = ('<metadata><oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" '
      'xmlns:dc="http://purl.org/dc/elements/1.1/">{}</oai_dc:dc></metadata>')


def rec(ident, stamp, fields, deleted=False):
    status = ' status="deleted"' if deleted else ""
    body = "" if deleted else DC.format("".join(f"<dc:{k}>{v}</dc:{k}>" for k, v in fields))
    return f'<record><header{status}><identifier>{ident}</identifier><datestamp>{stamp}</datestamp>' \
           f'<setSpec>col_1</setSpec></header>{body}</record>'


R1 = rec("oai:arsiv:1", "2024-01-01", [("title", "Türk dış politikasında NATO"), ("creator", "Yılmaz, Ayşe"),
                                        ("contributor", "Demir, Mehmet (Danışman)"), ("subject", "NATO; Dış politika"),
                                        ("subject", "Güvenlik"), ("publisher", "Ornek Üniversitesi"),
                                        ("description", "Bu tez Türkiye ile NATO ilişkisini inceler."),
                                        ("identifier", "https://arsiv.ornek.edu.tr/handle/1/1"), ("date", "2023")])
R2 = rec("oai:arsiv:2", "2024-02-01", [("title", "Enerji güvenliği"), ("creator", "Kaya, Ali"), ("subject", "Güvenlik")])
R3 = rec("oai:arsiv:3", "2024-03-01", [], deleted=True)


def make_server(state):
    def handler(request):
        q = dict(request.url.params)
        state.setdefault("calls", []).append(q)
        verb = q.get("verb")
        if verb == "Identify":
            body = ("<Identify><repositoryName>Örnek Açık Arşiv</repositoryName><protocolVersion>2.0</protocolVersion>"
                    "<earliestDatestamp>2010-01-01T00:00:00Z</earliestDatestamp></Identify>")
        elif verb == "ListSets":
            body = "<ListSets><set><setSpec>col_1</setSpec><setName>Tezler</setName></set></ListSets>"
        elif verb == "ListRecords" and "resumptionToken" not in q:
            assert q["metadataPrefix"] == "oai_dc"
            body = f"<ListRecords>{R1}<resumptionToken>t2</resumptionToken></ListRecords>"
        elif verb == "ListRecords":
            records = state.get("page2", R2 + R3)
            body = f"<ListRecords>{records}<resumptionToken/></ListRecords>"
        elif verb == "GetRecord":
            body = ('<GetRecord><record><metadata><atom:entry xmlns:atom="http://www.w3.org/2005/Atom">'
                    '<atom:link rel="http://www.openarchives.org/ore/terms/aggregates" '
                    'href="https://arsiv.ornek.edu.tr/bitstream/1/tez.pdf" type="application/pdf"/></atom:entry>'
                    '</metadata></record></GetRecord>')
        elif request.url.path.endswith(".pdf"):
            return httpx.Response(200, content=state["pdf"])
        else:
            body = '<error code="badVerb">bad</error>'
        return httpx.Response(200, text=HEAD + body + "</OAI-PMH>", headers={"Content-Type": "text/xml"})
    return handler


def client_for(state, tmp_path):
    h = Http(cache_dir=tmp_path / "cache", min_interval=0)
    h.client = httpx.Client(transport=httpx.MockTransport(make_server(state)))
    return oai.OaiClient("https://arsiv.ornek.edu.tr/oai/request", http=h)


@pytest.fixture
def proj(tmp_data, monkeypatch):
    import kg.okg.store as st
    monkeypatch.setattr(st, "DATA_DIR", tmp_data / "data")
    monkeypatch.delenv("NEO4J_URI", raising=False)
    return create_project("Arşiv testi")


def test_identify_sets_records(tmp_path):
    c = client_for({}, tmp_path)
    assert c.identify()["name"] == "Örnek Açık Arşiv"
    assert c.list_sets() == [{"spec": "col_1", "name": "Tezler"}]
    recs = list(c.list_records(limit=10))
    assert [r.identifier for r in recs] == ["oai:arsiv:1", "oai:arsiv:2", "oai:arsiv:3"]
    assert recs[0].dc["subject"] == ["NATO; Dış politika", "Güvenlik"] and recs[2].deleted
    assert len(list(c.list_records(limit=1))) == 1


def test_helpers():
    assert oai.person_name("Yılmaz, Ayşe") == "Ayşe Yılmaz"
    assert oai.person_name("Demir, Mehmet (Danışman)") == "Mehmet Demir"
    assert oai.person_name("Ornek Kurumu") == "Ornek Kurumu"
    assert oai.split_subjects(["NATO; Dış politika", "NATO"]) == ["NATO", "Dış politika"]
    with pytest.raises(oai.OaiError):
        oai.check_base_url("ftp://x")


def test_harvest_metadata_graph_and_rerun(proj, tmp_path):
    state = {}
    res = oai.harvest(proj, "https://arsiv.ornek.edu.tr/oai/request", client=client_for(state, tmp_path))
    assert len(res["added"]) == 2 and res["deleted"] == 0
    docs = proj.documents()
    assert docs[0]["source"]["url"] == "https://arsiv.ornek.edu.tr/handle/1/1"
    assert "Bu tez Türkiye ile NATO" in proj.text(docs[0]["id"])
    onto = proj.ontology()
    assert {"Yayın", "Kişi", "Konu", "Kurum"} <= set(onto.type_names())
    assert onto.relation("YAZARI").uri == "http://purl.org/dc/elements/1.1/creator"
    g = proj.graph()
    names = {e["name"] for e in g["entities"].values()}
    assert {"Ayşe Yılmaz", "Mehmet Demir", "NATO", "Güvenlik", "Ornek Üniversitesi"} <= names
    preds = sorted(r["predicate"] for r in g["relations"].values())
    assert preds.count("KONUSU") == 4 and "KATKIDA_BULUNAN" in preds and "YAYINCISI" in preds
    guv = next(e for e in g["entities"].values() if e["name"] == "Güvenlik")
    assert len({m["doc_id"] for m in guv["mentions"]}) == 2  # iki kayıtta ortak konu → tek düğüm
    # tekrar çekmek çoğaltmaz; kayıt silinirse belge de silinir
    oai.harvest(proj, "https://arsiv.ornek.edu.tr/oai/request", client=client_for(state, tmp_path))
    assert len(proj.documents()) == 2 and len(proj.graph()["relations"]) == len(g["relations"])
    state["page2"] = rec("oai:arsiv:2", "2024-05-01", [], deleted=True)
    res = oai.harvest(proj, "https://arsiv.ornek.edu.tr/oai/request", client=client_for(state, tmp_path))
    assert res["deleted"] == 1 and len(proj.documents()) == 1
    assert "Ali Kaya" not in {e["name"] for e in proj.graph()["entities"].values()}


def test_harvest_fulltext(proj, tmp_path):
    import pymupdf
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), "Tam metin: Turkiye 1952 yilinda NATO'ya katildi. " * 12)
    state = {"pdf": pdf.tobytes(), "page2": ""}
    res = oai.harvest(proj, "https://arsiv.ornek.edu.tr/oai/request", fulltext=True, build_metadata_graph=False,
                      client=client_for(state, tmp_path))
    assert res["fulltext"] == 1
    doc = proj.documents()[0]
    assert doc["source"]["fulltext"] and "TAM METİN" in proj.text(doc["id"])


def test_oai_errors(tmp_path):
    h = Http(cache_dir=tmp_path / "c", min_interval=0)
    h.client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>değil</html>")))
    with pytest.raises(oai.OaiError):
        oai.OaiClient("https://x.org/oai", http=h).identify()
