import re
import sys
import importlib
from types import SimpleNamespace

import pytest

from kg.llm import FakeLLM

DOCS = {
    "a.txt": "Başlık: Transformatör arızaları\n\nBu çalışmada güç transformatörü ve kesici arızaları incelendi. "
             "Transformatör sıcaklığı PSCAD ile modellendi. 154 kV şebekede kısa devre analizi yapıldı.",
    "b.txt": "Başlık: Güneş santrali\n\nFotovoltaik sistemde MPPT algoritması ve inverter verimi PSCAD üzerinde incelendi.",
    "c.txt": "Başlık: Rüzgâr türbini\n\nRüzgâr türbininde jeneratör ve inverter kontrolü için MPPT yöntemi önerildi.",
    "d.txt": "Başlık: Koruma\n\nKesici ve röle koordinasyonu kısa devre akımlarına göre yapıldı; transformatör korundu.",
    "e.txt": "Başlık: Dağıtım\n\nDağıtım şebekesinde harmonik bozulma ve inverter etkisi ölçüldü.",
    "f.txt": "Başlık: Yeni\n\nBir inverter ve bir transformatör PSCAD ile test edildi.",
}
GOLD = {"transformatör": "Ekipman", "güç transformatörü": "Ekipman", "kesici": "Ekipman", "inverter": "Ekipman",
        "jeneratör": "Ekipman", "röle": "Ekipman", "PSCAD": "Yazılım / Araç", "MPPT": "Yöntem / Teknik",
        "kısa devre": "Arıza / Olay", "154 kV": "Birim / Değer", "harmonik bozulma": "Arıza / Olay"}


def responder(system, user, schema):
    """Dil modeli taklidi: her METİN bölümünde sözlükteki ifadeleri (metindeki yazılışıyla) döndürür."""
    docs = []
    for i, body in re.findall(r"=== METİN (\d+) ===\n(.*?)(?=\n\n=== METİN |\Z)", user, re.S):
        ents = []
        for k, label in GOLD.items():
            m = re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)", body, re.I)
            if m:
                ents.append({"text": m.group(), "label": label})
        ents.append({"text": "uydurma varlık", "label": "Ekipman"})  # metinde yok → reddedilmeli
        docs.append({"index": int(i), "entities": ents})
    return {"documents": docs}


@pytest.fixture
def proj(tmp_data, monkeypatch):
    import kg.okg.store as st
    monkeypatch.setattr(st, "DATA_DIR", tmp_data / "data")
    p = st.create_project("Elektrik", "test")
    ids = {}
    for name, text in DOCS.items():
        ids[name] = p.add_document(name, text.encode(), text)["id"]
    # a.txt için arşiv metadata grafında bir Yayın düğümü
    p.add_extraction(ids["a.txt"] + "#meta", [{"type": "Yayın", "name": "Transformatör arızaları",
                                                "mention": "Transformatör arızaları", "description": None}], [], [])
    return p, ids


def test_spans_windows_gliner_format():
    from kg.okg import ner
    t = "Ankara'da transformatör ve Transformatör. transformatörün"
    assert ner.find_spans(t, "transformatör") == [(10, 23)]  # birebir; ekli kelime eşleşmez
    assert ner.find_spans(t, "TRANSFORMATÖR") == [(10, 23), (27, 40)]  # birebir yoksa harf duyarsız
    long = " ".join(f"kelime{i}." for i in range(450))
    ws = ner.windows(long, 200)
    assert len(ws) == 3 and all(long[o:o + len(w)] == w for o, w in ws)
    ents = [{"start": 10, "end": 23, "label": "Ekipman"}, {"start": 0, "end": 6, "label": "Yer"}]
    ex = ner.to_gliner(t, ents)[0]
    assert ex["tokenized_text"][:3] == ["Ankara", "'", "da"]
    assert sorted(ex["ner"]) == [[0, 0, "Yer"], [3, 3, "Ekipman"]]
    assert ner.match_preset("Elektrik Mühendisliği") == "elektrik" and ner.match_preset("arkeoloji") is None


def test_pipeline_grouped_sozluk(proj):
    from kg.okg import ner
    p, ids = proj
    cfg = ner.load_config(p)
    cfg["labels"] = [{"name": n, "description": d} for n, d in ner.DOMAIN_PRESETS["elektrik"]["labels"]]
    ner.save_config(p, cfg)
    assert "Ekipman" in p.ontology().type_names() and any(r.name == "BAHSEDER" for r in p.ontology().relation_types)

    llm = FakeLLM(responder)
    res = ner.annotate(p, llm, doc_ids=[ids[n] for n in ("a.txt", "b.txt", "c.txt", "d.txt", "e.txt")], per_request=2)
    assert res["added"] == 5 and res["rejected"] == 5
    assert len(llm.calls) == 3  # 5 belge, istek başına 2 → 3 istek
    rows = {r["doc_id"]: r for r in ner.load_dataset(p)}
    a = rows[ids["a.txt"]]
    assert {(e["text"], e["label"]) for e in a["entities"]} >= {("Transformatör", "Ekipman"), ("PSCAD", "Yazılım / Araç"),
                                                              ("154 kV", "Birim / Değer"), ("güç transformatörü", "Ekipman")}
    assert all(a["text"][e["start"]:e["end"]] == e["text"] for e in a["entities"])
    # grafa yazıldı ve Yayın düğümüne bağlandı
    g = p.graph()
    pub = "Yayın::transformatör arızaları"
    assert any(r["subject"] == pub and r["predicate"] == "BAHSEDER" and r["object"] == "Yazılım / Araç::pscad" for r in g["relations"].values())
    assert p.document(ids["a.txt"])["ner"] == "llm"
    # ikinci çalıştırma: etiketli belgeler atlanır
    assert ner.annotate(p, llm, doc_ids=[ids["a.txt"]])["added"] == 0

    m = ner.train(p, kind="sozluk")
    assert m["model_kind"] == "sozluk" and m["n_test"] == 1 and m["test"]["micro"]["f1"] > 0.5
    out = ner.apply(p)  # yalnızca etiketlenmemiş f.txt
    assert out["docs"] == 1 and out["entities"] >= 3
    f = p.graph()["entities"]
    assert any(m_["doc_id"] == ids["f.txt"] + "#ner" for e in f.values() for m_ in e["mentions"])
    assert p.document(ids["f.txt"])["ner"] == "model"
    # belge silinince NER katkısı da gider
    p.delete_document(ids["f.txt"])
    assert not any(m_["doc_id"].startswith(ids["f.txt"]) for e in p.graph()["entities"].values() for m_ in e["mentions"])
    assert ner.clear_ner(p) == 5


class _FakeBatches:
    def __init__(self):
        self.created, self.polls = None, 0

    def create(self, requests):
        self.created = requests
        return SimpleNamespace(id="msgbatch_1")

    def retrieve(self, bid):
        self.polls += 1
        counts = SimpleNamespace(succeeded=0, errored=0, canceled=0, expired=0, processing=len(self.created))
        return SimpleNamespace(processing_status="ended" if self.polls > 1 else "in_progress", request_counts=counts)

    def results(self, bid):
        import json
        for r in self.created:
            ans = responder("", r["params"]["messages"][0]["content"], None)
            msg = SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(ans, ensure_ascii=False))])
            yield SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=msg))


def test_anthropic_batch_api(proj):
    from kg.okg import ner
    p, ids = proj
    cfg = ner.load_config(p)
    cfg["labels"] = [{"name": n, "description": d} for n, d in ner.DOMAIN_PRESETS["elektrik"]["labels"]]
    ner.save_config(p, cfg)
    batches = _FakeBatches()
    llm = SimpleNamespace(client=SimpleNamespace(messages=SimpleNamespace(batches=batches)), model_name="claude-test", effort="high")
    res = ner.annotate(p, llm, n=6, per_request=3, poll_seconds=0)
    assert res["added"] == 6 and len(batches.created) == 2 and batches.polls == 2
    params = batches.created[0]["params"]
    assert params["output_config"]["format"]["type"] == "json_schema" and params["model"] == "claude-test"
    assert not (ner.ner_dir(p) / "state.json").exists()


@pytest.fixture
def webc(tmp_data, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("APP_PASSWORD", "s3cret-pass")
    monkeypatch.setenv("APP_USERNAME", "admin")
    monkeypatch.setenv("LLM_BACKEND", "manual")
    monkeypatch.delenv("NEO4J_URI", raising=False)
    import kg.okg.store as st
    import kg.web.jobs as jobs_mod
    monkeypatch.setattr(st, "DATA_DIR", tmp_data / "data")
    monkeypatch.setattr(jobs_mod, "DATA_DIR", tmp_data / "data")
    sys.modules.pop("kg.web.app", None)
    app_mod = importlib.import_module("kg.web.app")
    c = TestClient(app_mod.app)
    c.auth = ("admin", "s3cret-pass")
    return c, app_mod


def test_ner_web(webc):
    c, app_mod = webc
    from test_web import _csrf, _new_project, _wait_job
    import kg.okg.store as st
    slug = _new_project(c, "Elektrik Arşivi")
    p = st.Project(slug)
    for name, text in DOCS.items():
        p.add_document(name, text.encode(), text)
    assert "Hızlı varlık modeli" in c.get(f"/p/{slug}/docs").text
    r = c.get(f"/p/{slug}/ner")
    assert r.status_code == 200 and "1. Varlık türleri" in r.text
    tok = _csrf(c, f"/p/{slug}/ner")
    r = c.post(f"/p/{slug}/ner/suggest", data={"csrf": tok, "domain": "elektrik mühendisliği", "how": "preset"}, follow_redirects=False)
    assert r.status_code == 303
    assert 'value="Ekipman"' in c.get(f"/p/{slug}/ner").text
    app_mod.get_llm = lambda: FakeLLM(responder)
    r = c.post(f"/p/{slug}/ner/annotate", data={"csrf": tok, "n": 5, "per_request": 5, "add_to_graph": "1"}, follow_redirects=False)
    job = r.headers["location"].split("/jobs/")[1].split("?")[0]
    assert _wait_job(c, job)["status"] == "done"
    page = c.get(f"/p/{slug}/ner").text
    assert "Veri seti: 5 belge" in page
    r = c.post(f"/p/{slug}/ner/train", data={"csrf": tok, "kind": "sozluk"}, follow_redirects=False)
    assert _wait_job(c, r.headers["location"].split("/jobs/")[1].split("?")[0])["status"] == "done"
    assert "Sözlük modeli" in c.get(f"/p/{slug}/ner").text
    r = c.post(f"/p/{slug}/ner/apply", data={"csrf": tok, "scope": "unlabeled", "threshold": 0.3}, follow_redirects=False)
    assert _wait_job(c, r.headers["location"].split("/jobs/")[1].split("?")[0])["status"] == "done"
    tr = c.post(f"/api/p/{slug}/ner/try", json={"csrf": tok, "text": "Yeni bir inverter PSCAD ile denendi."}).json()
    assert {e["text"] for e in tr["entities"]} >= {"inverter", "PSCAD"}
    assert c.post(f"/api/p/{slug}/ner/try", json={"csrf": "x", "text": "a"}).status_code == 403
    jl = c.get(f"/p/{slug}/ner/export/dataset.jsonl").text.splitlines()
    assert len(jl) == 5
    gl = c.get(f"/p/{slug}/ner/export/gliner.json").json()
    assert gl["train"] and gl["train"][0]["tokenized_text"] and "ekipman" in gl["labels"] and gl["label_map"]["Ekipman"] == "ekipman"
    assert "NER" in c.get(f"/p/{slug}/docs").text
    r = c.post(f"/p/{slug}/ner/model-upload", data={"csrf": tok}, files={"file": ("m.zip", b"bozuk", "application/zip")}, follow_redirects=False)
    assert "err=" in r.headers["location"]


def test_gliner_train_apply_offline(proj, tmp_path):
    """GLiNER kuruluysa: internetsiz küçük rastgele bir omurgayla eğitim → kaydet → yükle → uygula zinciri çalışır."""
    pytest.importorskip("gliner")
    pytest.importorskip("torch")
    from gliner import GLiNER, GLiNERConfig
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertModel, BertTokenizerFast
    from kg.okg import ner

    bb = tmp_path / "tiny-bert"
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(list(DOCS.values()) * 5, trainers.WordPieceTrainer(
        vocab_size=600, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    fast = BertTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                             sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(bb)
    BertModel(BertConfig(vocab_size=fast.vocab_size, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64)).save_pretrained(bb)
    base = tmp_path / "gliner-base"
    GLiNER.from_config(GLiNERConfig(model_name=str(bb), hidden_size=32, max_width=6)).save_pretrained(base)

    p, ids = proj
    cfg = ner.load_config(p)
    cfg["labels"] = [{"name": n, "description": d} for n, d in ner.DOMAIN_PRESETS["elektrik"]["labels"]]
    cfg["base_model"] = str(base)
    ner.save_config(p, cfg)
    ner.annotate(p, FakeLLM(responder), doc_ids=[ids[n] for n in ("a.txt", "b.txt", "c.txt", "d.txt", "e.txt")])
    m = ner.train(p, kind="gliner", epochs=1)
    assert m["model_kind"] == "gliner" and "test" in m and "base" in m
    assert (ner.ner_dir(p) / "model" / "gliner_config.json").exists()
    assert m["gliner_labels"]["Yazılım / Araç"] == "yazılım / araç"
    out = ner.apply(p, threshold=0.3)
    assert out["docs"] == 1


SAMPLE_JSON = """{
  "domain": "Atık gazlaştırma ve katı çıktı karakterizasyonu",
  "entity_types": [
    {"name": "Süreç", "description": "Termal işlem (örn. gazlaştırma, yakma).", "uri": null},
    {"name": "AnalizYöntemi", "description": "Deneysel analiz yöntemi (örn. XRD, XRF, SEM).", "uri": null},
    {"name": "Ürün", "description": "Gaz veya katı çıktı.", "uri": null, "prompt": "ürün veya çıktı"},
    {"name": "Atık", "description": "Beslenen atık.", "examples": ["tehlikeli atık", "çamur"]}
  ],
  "relation_types": [
    {"name": "ANALİZ_EDER", "description": "Analiz yöntemi numuneye uygulanır.", "source": ["AnalizYöntemi"], "target": ["Atık", "Ürün"], "uri": null},
    {"name": "ÜRETİR", "description": "Süreç ürün oluşturur.", "source": ["Süreç"], "target": ["Ürün"], "uri": null}
  ]
}"""


def test_labels_json_import(proj):
    from kg.okg import ner
    p, _ = proj
    parsed = ner.parse_labels_json(SAMPLE_JSON)
    assert parsed["domain"].startswith("Atık gazlaştırma") and len(parsed["labels"]) == 4
    atik = parsed["labels"][3]
    assert atik["description"] == "Beslenen atık. (örn. tehlikeli atık, çamur)"
    cfg = ner.load_config(p)
    cfg["labels"] = parsed["labels"]
    ner.save_config(p, cfg)
    assert ner.merge_relation_types(p, parsed["relation_types"], parsed["labels"]) == 2
    onto = p.ontology()
    assert {"ANALİZ_EDER", "ÜRETİR", "BAHSEDER"} <= {r.name for r in onto.relation_types}
    assert "AnalizYöntemi" in onto.type_names()
    lm = ner.gliner_label_map(ner.load_config(p))
    assert lm == {"Süreç": "süreç", "AnalizYöntemi": "analiz yöntemi", "Ürün": "ürün veya çıktı", "Atık": "atık"}
    cfg = ner.load_config(p)
    cfg["gliner_use_description"] = True
    assert ner.gliner_label_map(cfg)["Süreç"].startswith("Termal işlem")
    assert ner.gliner_label_map(cfg)["Ürün"] == "ürün veya çıktı"  # prompt önce gelir
    out = ner.labels_json(p, ner.load_config(p))
    assert {r["name"] for r in out["relation_types"]} == {"ANALİZ_EDER", "ÜRETİR"}
    assert ner.parse_labels_json('{"Kişi": "insan", "Yer": "konum"}')["labels"][1] == {"name": "Yer", "description": "konum"}
    with pytest.raises(ValueError):
        ner.parse_labels_json('{"domain": "x", "entity_types": []}')


def test_gliner_predict_maps_labels():
    from kg.okg import ner

    class M:
        def inference(self, texts, labels, threshold, batch_size):
            assert labels == ["analiz yöntemi"]
            return [[{"start": 0, "end": 3, "label": "analiz yöntemi", "text": "XRD", "score": 0.9}] for _ in texts]

    g = ner.GlinerModel(M(), "x", {"AnalizYöntemi": "analiz yöntemi"})
    assert g.predict(["XRD ile incelendi."], ["AnalizYöntemi"])[0][0]["label"] == "AnalizYöntemi"


def test_labels_import_web(webc):
    c, _ = webc
    from test_web import _csrf, _new_project
    slug = _new_project(c, "Gazlaştırma")
    tok = _csrf(c, f"/p/{slug}/ner")
    r = c.post(f"/p/{slug}/ner/labels/import", data={"csrf": tok, "with_relations": "1"},
               files={"file": ("turler.json", SAMPLE_JSON.encode(), "application/json")}, follow_redirects=False)
    assert r.status_code == 303 and "err" not in r.headers["location"]
    page = c.get(f"/p/{slug}/ner").text
    assert 'value="AnalizYöntemi"' in page and 'placeholder="analiz yöntemi"' in page and 'value="ürün veya çıktı"' in page
    assert "Atık gazlaştırma" in page
    assert "ANALİZ_EDER" in c.get(f"/p/{slug}/ontology").text
    exp = c.get(f"/p/{slug}/ner/labels.json").json()
    assert exp["entity_types"][2]["prompt"] == "ürün veya çıktı"
    r = c.post(f"/p/{slug}/ner/labels/import", data={"csrf": tok, "text": "{bozuk"}, follow_redirects=False)
    assert "err=" in r.headers["location"]
    # tablo kaydı prompt ve açıklama seçeneğini korur
    c.post(f"/p/{slug}/ner/labels", data={"csrf": tok, "name": ["Süreç", "Ürün"], "desc": ["a", "b"],
                                          "prompt": ["", "çıktı"], "use_desc": "1", "threshold": 0.4})
    import kg.okg.store as st
    from kg.okg import ner
    cfg = ner.load_config(st.Project(slug))
    assert cfg["gliner_use_description"] and cfg["labels"][1]["prompt"] == "çıktı" and "prompt" not in cfg["labels"][0]


def test_labels_merge_and_clear(webc):
    c, _ = webc
    from test_web import _csrf, _new_project
    import kg.okg.store as st
    from kg.okg import ner
    slug = _new_project(c, "Birleştirme")
    tok = _csrf(c, f"/p/{slug}/ner")
    cfgp = lambda: ner.load_config(st.Project(slug))
    c.post(f"/p/{slug}/ner/labels/import", data={"csrf": tok, "text": SAMPLE_JSON, "mode": "replace"})
    assert len(cfgp()["labels"]) == 4
    # var olanlara ekle: aynı ad güncellenir, yeni ad eklenir, eskiler kalır
    extra = '{"entity_types": [{"name": "atık", "description": "Yeni açıklama"}, {"name": "Mevzuat", "description": "Yönetmelik"}]}'
    c.post(f"/p/{slug}/ner/labels/import", data={"csrf": tok, "text": extra, "mode": "merge"})
    labels = cfgp()["labels"]
    assert [l["name"] for l in labels] == ["Süreç", "AnalizYöntemi", "Ürün", "Atık", "Mevzuat"]
    assert labels[3]["description"] == "Yeni açıklama" and labels[2]["prompt"] == "ürün veya çıktı"
    assert cfgp()["domain"].startswith("Atık gazlaştırma")  # birleştirmede alan korunur
    # hazır öneriyi ekle
    c.post(f"/p/{slug}/ner/suggest", data={"csrf": tok, "domain": "", "preset": "genel", "how": "preset", "mode": "merge"})
    names = [l["name"] for l in cfgp()["labels"]]
    assert names[:5] == ["Süreç", "AnalizYöntemi", "Ürün", "Atık", "Mevzuat"] and "Kişi" in names
    # değiştir
    c.post(f"/p/{slug}/ner/suggest", data={"csrf": tok, "domain": "", "preset": "genel", "how": "preset", "mode": "replace"})
    assert "Süreç" not in [l["name"] for l in cfgp()["labels"]]
    assert "Tümünü sil" in c.get(f"/p/{slug}/ner").text
    c.post(f"/p/{slug}/ner/clear", data={"csrf": tok, "what": "labels"})
    assert cfgp()["labels"] == []


def test_settings_and_selection(proj):
    from kg.okg import ner
    p, ids = proj
    cfg = ner.load_config(p)
    cfg["labels"] = [{"name": n, "description": d} for n, d in ner.DOMAIN_PRESETS["elektrik"]["labels"]]
    ner.save_config(p, cfg)
    s = ner.update_settings(p, "annotate", {"n": "3", "count_mode": "total", "per_request": "999", "selection": "kötü",
                                            "keyword": "inverter", "batch_api": False, "bilinmeyen": 1})
    assert s["n"] == 3 and s["count_mode"] == "total" and s["per_request"] == 100 and s["selection"] == "random"
    assert s["batch_api"] is False and "bilinmeyen" not in s
    # anahtar kelime süzgeci: yalnız "inverter" geçen belgeler (b, c, e, f)
    assert set(ner.select_docs(p, keyword="inverter")) == {ids[n] for n in ("b.txt", "c.txt", "e.txt", "f.txt")}
    assert ner.select_docs(p, selection="longest")[0] == ids["a.txt"]
    llm = FakeLLM(responder)
    assert ner.annotate(p, llm)["added"] == 3  # kayıtlı ayarlar: toplam 3, "inverter" süzgeci
    assert ner.annotate(p, llm)["added"] == 0  # hedefe ulaşıldı
    assert ner.annotate(p, llm, n=4)["added"] == 1  # hedef 4'e çıkınca yalnız 1 yeni
    assert len(ner.load_dataset(p)) == 4
    # veri setini küçült / belge çıkar
    assert ner.remove_from_dataset(p, keep=2) == 2 and len(ner.load_dataset(p)) == 2
    gone = ner.load_dataset(p)[0]["doc_id"]
    assert ner.remove_from_dataset(p, doc_ids=[gone]) == 1
    assert p.document(gone).get("ner") is None
    # uygulama ayarları: sınır ve "henüz işlenmemiş" kapsamı
    ner.annotate(p, llm, doc_ids=[ids["a.txt"], ids["d.txt"]])
    ner.update_settings(p, "train", {"kind": "sozluk", "test_ratio": "0"})
    m = ner.train(p)
    assert m["model_kind"] == "sozluk" and m["n_test"] == 0
    ner.update_settings(p, "apply", {"limit": 2, "scope": "new"})
    assert ner.apply(p)["docs"] == 2
    remaining = len([d for d in p.documents() if not d.get("ner")])
    assert remaining > 0 and ner.apply(p, limit=0)["docs"] == remaining


def test_ner_settings_web(webc):
    c, _ = webc
    from test_web import _csrf, _new_project
    import kg.okg.store as st
    from kg.okg import ner
    slug = _new_project(c, "Ayarlar")
    p = st.Project(slug)
    for name, text in DOCS.items():
        p.add_document(name, text.encode(), text)
    tok = _csrf(c, f"/p/{slug}/ner")
    r = c.post(f"/p/{slug}/ner/annotate", data={"csrf": tok, "_bools": "batch_api,add_to_graph", "action": "save",
                                                "n": "4", "count_mode": "total", "source": "upload", "keyword": "kesici",
                                                "per_request": "3", "add_to_graph": "1"}, follow_redirects=False)
    assert r.status_code == 303 and "ok=" in r.headers["location"]
    s = ner.get_settings(ner.load_config(p))["annotate"]
    assert s["n"] == 4 and s["count_mode"] == "total" and s["batch_api"] is False and s["add_to_graph"] is True
    page = c.get(f"/p/{slug}/ner").text
    assert 'name="n" value="4"' in page and 'value="kesici"' in page and "Ayarlar kaydedildi" not in page
    prev = c.get(f"/api/p/{slug}/ner/preview", params={"step": "annotate", "keyword": "kesici", "n": 4, "count_mode": "total"}).json()
    assert prev == {"available": 2, "will_process": 2, "in_dataset": 0}
    r = c.post(f"/p/{slug}/ner/train", data={"csrf": tok, "action": "save", "kind": "sozluk", "epochs": "7", "lr": "1e-5"},
               follow_redirects=False)
    t = ner.get_settings(ner.load_config(p))["train"]
    assert t["epochs"] == 7 and t["lr"] == 1e-5 and t["kind"] == "sozluk"
    c.post(f"/p/{slug}/ner/apply", data={"csrf": tok, "_bools": "skip_done", "action": "save", "limit": "5", "threshold": "0.7"})
    assert ner.get_settings(ner.load_config(p))["apply"]["limit"] == 5 and ner.load_config(p)["threshold"] == 0.7
    assert c.get(f"/api/p/{slug}/ner/preview", params={"step": "apply", "limit": 2}).json()["will_process"] == 2
