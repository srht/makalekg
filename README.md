# kg-bone-brain — Kemik–Beyin Ekseni Knowledge Graph

Kemik–beyin ekseni literatürünü **iddia merkezli** bir knowledge graph'a dönüştüren, uzman olmayan
kullanıcının soru sormasına ve sistemin **otonom hipotez** üretmesine izin veren boru hattı.
Tasarım belgesi: [SPEC.md](SPEC.md).

Tohum makale: Cai ve ark., *Tibial bone compression promotes recovery after brain injury through
osteocyte PIEZO1*, Nature Neuroscience (2026), doi:10.1038/s41593-026-02422-w.

## Değişmez ilkeler ve kodda nerede uygulandıkları

| İlke | Uygulama |
|---|---|
| Kanıt ve hipotez asla karışmaz | Kanıt: `SUPPORTS/CONTRADICTS` + Claim'den türetilen `INCREASES/DECREASES/REQUIRES/MEDIATES/PROTECTS`. Hipotez: yalnızca `HYPOTHESIS`, yalnızca `kg/infer/writer.py` yazar. Kural motoru, PyKEEN ve ajan yalnızca kanıt kenarlarını okur; ajanın önerdiği her yol halkası gerçek bir kanıt kenarına karşı doğrulanır (`validate_path`). Hipotez durumu yalnızca kanıt grafından türetilir (`infer/literature.py`). |
| Her kenarın kaynağı vardır | `Provenance` modeli: `source_doi`, `figure`, `source_sentence` (≤ 300 karakter, boş olamaz), `extraction_model`, `extraction_confidence`, `verified_by_human`. Kaynak cümlesi metinde birebir geçmeyen çıkarım reddedilir. |
| Belirsizlik açık yazılır | Çıkarım promptu: bilinmeyen alan `null`, sayı uydurma yok; yapılandırılmış çıktı Pydantic ile doğrulanır. Öz-doğrulama `supported/partial/unsupported`; `unsupported` → `data/review_queue.jsonl`. |
| Tür ve bağlam korunur | `Experiment.design/species`; gen düğüm anahtarı türe özgüdür (`Gene:mouse:MGI:…` ≠ `Gene:human:HGNC:…`), `ORTHOLOG_OF` ile bağlanır. |
| Kullanıcı uzman değildir | `score.describe_experiment` her kanıtı düz Türkçe özetler ("Fig. 5 h, fare, n = 9–10, körlemesiz, tek laboratuvar"); `kg ask` ve hipotez kartları bunu kullanır. |

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'            # PyKEEN için: pip install -e '.[embed]'
cp .env.example .env               # ANTHROPIC_API_KEY, NEO4J_PASSWORD, CONTACT_EMAIL ...
docker compose up -d               # Neo4j 5 → http://localhost:7474
```

LLM arka ucu `.env` ile seçilir: `LLM_BACKEND=anthropic` (varsayılan model `claude-opus-5-5`,
yapılandırılmış JSON çıktısı, adaptive thinking, sunucu tarafı `fallbacks: "default"`) veya
`LLM_BACKEND=openai_compat` (yerel model sunucusu, `OPENAI_COMPAT_BASE_URL`).
Neo4j olmadan denemek için `GRAPH_BACKEND=memory` (graf `data/graph.json`'a yazılır) ya da her komutta
`--backend memory`.

## Kullanım

```bash
# Faz 1 — tek makale (PDF'i data/seed.pdf olarak ekleyin; JATS açık erişimde varsa o kullanılır)
kg ingest --phase 1 --pdf data/seed.pdf
#  → spotcheck.html, reports/inconsistencies.md, reports/phase1_acceptance.md
#  spotcheck.html'i tarayıcıda açın, 10 kenarı işaretleyin, JSON'u indirin:
kg spotcheck --results spotcheck_results_<run>.json
kg accept --phase 1

# Faz 2 — genişleme (referanslar + atıf yapanlar + anahtar kavram sorguları, ≤ 200 makale/set)
kg ingest --phase 2 --set bone_brain
kg ingest --phase 2 --set cholesterol_tracing
kg unresolved                       # normalize edilemeyen varlıklar + bulanık eşleşme önerileri

# Faz 3 — çıkarım
kg infer                            # kural zincirleri → HYPOTHESIS kenarları + reports/hypotheses/*.md
kg infer --agent --literature       # + LLM ajanı + Europe PMC literatür kontrolü
kg infer --literature --ingest-literature   # bulunan makaleleri boru hattına ekleyip durumu güncelle
kg ask "Kemiğe yük bindirmek beyin hasarında neden işe yarıyor olabilir?"

# Faz 4 — değerlendirme
kg infer --embed                    # PyKEEN RotatE
kg backtest --cutoff 2024           # reports/backtest.md: Precision@20, Refuted@20, rastgele çizgi, kalibrasyon
```

Boru hattı idempotenttir: ham metin `data/raw/`, çıkarımlar `data/extractions/`, HTTP yanıtları
`data/cache/` altında önbelleklenir; graf yazmaları `MERGE` ile yapılır. Çıkarımı yeniden yaptırmak
için `--force`. Her çalıştırma bir `IngestRun` kaydı bırakır.

## Dizin yapısı

```
config/   scoring.yaml (güven katsayıları) · rules.yaml (zincir kalıpları) · sources.yaml (tohumlar, sorgular)
kg/
  schema.py      Pydantic modelleri (düğümler, kenarlar, LLM çıkarım şeması, katı JSON şema dönüştürücü)
  discover.py    Europe PMC / Semantic Scholar / Crossref (meta veri, fon kaynağı, retraction)
  fetch.py       JATS / PDF / özet + bölümleme (figure legend'ler ayrı)
  extract.py     LLM çıkarımı + birebir kaynak cümle kontrolü + öz-doğrulama
  normalize.py   UniProt (MGI/HGNC/RGD xref) · HGNC · MeSH · Cell Ontology · ChEBI · bulanık öneri · ortologlar
  graph.py       depo arayüzü, salt-okunur snapshot, bellek-içi depo
  load.py        Neo4j deposu (MERGE) + yükleyici (REPLICATES, ORTHOLOG_OF)
  llm.py         Anthropic / OpenAI uyumlu yerel model / test için sahte model
  score.py       E, C, S formülleri + düz Türkçe kanıt düzeyi
  report.py      hipotez kartları
  pipeline.py    ingest orkestrasyonu
  infer/         rules.py · embed.py (PyKEEN) · agent.py (salt-okunur) · literature.py · writer.py · run.py
  validate/      spotcheck.py · consistency.py · backtest.py · acceptance.py
cli.py           ingest | infer | ask | spotcheck | backtest | accept | unresolved
data/            ham metin, json çıkarımlar, review_queue.jsonl (git'e girmez)
reports/         hipotez kartları, tutarsızlık, backtest ve kabul raporları (git'e girmez)
```

## Notlar ve SPEC'e göre kararlar

- **PROTECTS** yüklemi kanıt yüklemlerine eklendi: SPEC'teki kural örneği (`A -INCREASES-> B -PROTECTS-> C`) bunu kullanıyor.
- Varlık→varlık kenarları `stance` taşır: `affirmed` (iddia edilen ilişki deneyle destekleniyor) veya
  `negated` (etki yok / çürütüldü). Zincirler yalnızca `affirmed` kenarlardan kurulur; çelişkili yön
  ×0.3 cezası `negated`/ters yönlü kenarlardan hesaplanır.
- Spot-check hata oranı = yanlış / işaretlenen; "kısmen" ayrıca raporlanır. %10'u aşarsa o ingest
  partisinin kenarları `needs_review` olur ve zincirlerde kullanılmaz.
- Ajanın Neo4j erişimi READ erişim modu + yazma anahtar kelimesi filtresiyle sınırlıdır; Neo4j
  Enterprise kullanıyorsanız `NEO4J_READ_USER` için `reader` rolünde ayrı bir kullanıcı tanımlayın.
- `industry` fonlu makaleler skoru düşürmez, kartta uyarı satırı çıkar; `tracer_type` NBD/BODIPY ise
  "floresan grup molekülün davranışını değiştirebilir" notu çıkar.
- Tutarsızlık kontrolleri (n uyuşmazlığı, dışlama çelişkisi, protokol parametresi çelişkisi)
  "olası" bulgulardır; tohum makaledeki bilinen iki tutarsızlığın listede olduğunu insan doğrular.
- Backtest eşikleri SPEC'te açık soru olarak bırakıldı; rapor yalnızca rastgele çizgiyle karşılaştırır.

## Test

```bash
pytest
```

Testler ağ, Neo4j ve API anahtarı olmadan çalışır: sahte LLM, bellek-içi depo ve açıkça "SYNTHETIC
TEST FIXTURE" olarak işaretlenmiş sentetik bir makale metni kullanır (gerçek veri değildir).
