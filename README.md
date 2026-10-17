# Knowledge Graph Kurucu

Herhangi bir alandaki makale, tez, rapor ve dokümanlardan **ontoloji tabanlı** varlık ve ilişki çıkarımı yapıp
knowledge graph kuran web uygulaması.

## Nasıl çalışır

1. **Proje aç.** Her proje kendi ontolojisi, belgeleri ve grafıyla ayrı durur.
2. **Ontoloji:** hangi varlık türlerini (Kişi, Kurum, Devlet, Gen, Kavram…) ve hangi ilişki türlerini
   (ÇALIŞIR, ÜYESİDİR, ARTIRIR…) arayacağını tanımla. Üç yol var:
   - **Hazır şablon:** schema.org (standart), genel, akademik makale/tez, biyomedikal, uluslararası ilişkiler.
   - **Belgeden öner:** örnek bir belge ver; model okuyup bu alana uygun bir ontoloji önerir, sen düzenlersin.
   - **Standart ontoloji dosyası:** OWL / RDF / Turtle / JSON-LD (ör. schema.org, FOAF, Dublin Core, Protégé'de
     hazırladığın ontoloji). Dosyadaki sınıflardan ve ilişkilerden istediklerini seçersin; özgün URI'ler korunur.
3. **Belgeler:** PDF, Word (.docx), .txt, .md, .html yükle. Sistem her belgeyi ontolojiye göre işler.
   **Kurumsal arşivden çekme (OAI-PMH):** DSpace, EPrints gibi açık erişim arşivlerinin OAI-PMH adresini ver
   (DSpace'te genelde `…/oai/request`); "Bağlan" ile setler (topluluk/koleksiyon) listelenir. Kayıtların Dublin Core
   (`oai_dc`) metadatası tarih aralığı ve kayıt sınırıyla çekilir; her kayıt bir belge olur (başlık, yazarlar, konular,
   özet; istenirse DSpace ORE üzerinden tam metin PDF). **Metadata grafı** dil modeli olmadan kurulur:
   Yayın —YAZARI→ Kişi (dc:creator), —KATKIDA_BULUNAN→ Kişi (dc:contributor, ör. danışman), —KONUSU→ Konu (dc:subject),
   —YAYINCISI→ Kurum (dc:publisher); türler Dublin Core/FOAF/SKOS URI'leriyle ontolojiye eklenir. "Soyad, Ad"
   biçimindeki adlar "Ad Soyad"a çevrilir, ortak yazar/konu tek düğümde birleşir. Aynı kayıt tekrar çekilirse
   güncellenir; arşivde silinmişse belge de silinir. Kayıtlar istenirse ontolojiye göre de işlenir; kısa kayıtlar
   tek istemde gruplanır (kopyala–yapıştır modunda az sayıda yapıştırma).
4. **Graf:** etkileşimli ağ görünümü. Bir düğüme/oka tıklayınca ayrıntısı ve **hangi belgedeki hangi cümleden
   geldiği** görünür. İki varlık arasındaki bağlantı zincirini bulabilirsin. Düğüm yerleşimi ilk açılışta bir kez
   hesaplanıp `layout.json`'a kaydedilir; sonraki açılışlar anında yüklenir (yeni eklenen düğümler komşularının yanına yerleşir).
   **Ara** sayfası (her proje sayfasının üstündeki kutu da): birkaç kelime yaz, kelimelerin geçtiği ilişkiler
   kaynak cümleleriyle listelenir. Kelimeler varlık adlarında, ilişki adlarında, türlerde ve cümlelerde aranır;
   büyük/küçük harf, Türkçe karakter ve ek farkı sorun olmaz. Birden çok kelimede
   kelimelerin yan yana geçtiği sonuçlar (ör. "yapay zeka") önce, ayrı ayrı geçtikleri sonra listelenir. Bir varlığa tıklayınca yalnızca onun ilişkileri
   listelenir; sonuçlar CSV olarak indirilebilir (`/api/p/<proje>/search?q=...` JSON verir).
5. **Açık knowledge graph (Wikidata):** varlıkları Wikidata kayıtlarıyla eşle, sonra eşlenen varlıklar arasında
   Wikidata'nın bildiği ilişkileri (ör. bir ülkenin bir örgüte üyeliği) grafa ekle. Bunlar belgelerden gelenlerden
   ayrı tutulur ve kesikli çizgiyle gösterilir. Yanlış eşlemeyi panelden kaldırabilir veya başka adayı seçebilirsin.
6. **Dışa aktar:** CSV (Excel), JSON, RDF/Turtle (standart ontoloji URI'leri ve Wikidata `owl:sameAs` ile).
   Neo4j ayarlıysa graf her işlemden sonra oraya da yazılır (`project` özelliğiyle ayrılır).

### Güvenilirlik kuralları

Model yalnızca ontolojideki türleri seçebilir (yanıt şeması ontolojiden üretilir). Ardından her öneri denetlenir ve
geçmeyen **reddedilir** (belge sayfasında sebebiyle görünür):
- varlığın belgede geçtiği biçim metinde gerçekten olmalı,
- ilişkinin kaynak cümlesi metinde birebir geçmeli,
- ilişkinin özne/nesnesi ontolojideki tür kısıtına uymalı (ör. ÜYESİDİR: Devlet → Uluslararası örgüt).

Aynı ad ve türdeki varlıklar belgeler arasında tek düğümde birleşir. Bir belge yeniden işlendiğinde önce o belgenin
katkıları silinir, çift kayıt oluşmaz.

## Hızlı varlık modeli (GLiNER) — çok sayıda kayıt için

Binlerce DSpace kaydını tek tek dil modeline göndermek yerine (Belgeler → "Hızlı varlık modeli"):

1. **Varlık türleri:** alanı yaz (ör. "elektrik") → hazır tipik türler gelir (elektrik, bilgisayar, tıp, hukuk,
   uluslararası ilişkiler, eğitim, ekonomi, inşaat, genel) ya da dil modelinden öneri istenir. Liste düzenlenebilir.
   Liste **JSON olarak yüklenebilir** (ontoloji biçimi: `domain`, `entity_types[{name, description, uri}]`,
   `relation_types`; türe isteğe bağlı `prompt` = GLiNER'in göreceği kısa etiket, `examples` = örnek listesi).
   Açıklamalar dil modelinin etiketlemesinde kullanılır; GLiNER kısa etiket metnini görür (boşsa ad, ör.
   "AnalizYöntemi" → "analiz yöntemi"; istenirse açıklamanın tamamı verilir).
2. **Eğitim verisi:** N belge (varsayılan 200) gruplar hâlinde dil modeline gönderilir (istek başına ~10 kayıt).
   Anthropic API'de **Message Batches** kullanılır (standart fiyatın yarısı; paket kimliği saklanır, iş yarıda kalırsa
   aynı paket beklenir). Model varlıkları metindeki yazılışıyla döndürür; sistem her geçişin karakter aralığını
   metinde birebir doğrular (bulunamayan öneriler atılır). Veri seti `ner/dataset.jsonl` olarak saklanır,
   JSONL veya GLiNER eğitim biçiminde indirilebilir.
3. **Eğitim:** [GLiNER](https://github.com/urchade/GLiNER) (`urchade/gliner_multi-v2.1`, çok dilli, Apache-2.0)
   veri setiyle ince ayarlanır. Veri setinin %20'si testte kullanılır; eğitimsiz temel model ile eğitilmiş model
   aynı test belgelerinde kesinlik/duyarlılık/F1 ile karşılaştırılır. GLiNER kurulu değilse bağımlılıksız
   **sözlük modeli** eğitilir (görülen ifadeleri tanır, yeni adları genelleyemez). GPU'da eğitmek için:
   `scripts/train_gliner.py` + arayüzden model .zip yükleme.
4. **Uygulama:** model etiketlenmemiş belgelerin hepsinden varlık çıkarır (dil modeli yok, ücretsiz). Varlıklar grafa
   `<belge>#ner` katkısı olarak yazılır ve arşiv kaydının Yayın düğümüne `BAHSEDER` ilişkisiyle bağlanır.

Komut satırı: `python -m kg.okg.ner_cli run <proje> --domain elektrik --n 200` (ayrıca `labels`, `annotate`,
`train`, `apply`, `status`). Docker imajı GLiNER'i (torch CPU) `WITH_GLINER=1` ile kurar (varsayılan; imaja ~1 GB ekler,
`WITH_GLINER=0` ile kapatılır). Temel model ilk kullanımda Hugging Face'ten indirilir ve `data/hf` altında kalır.
CPU'da 200 belgelik eğitim yaklaşık 15–60 dakika sürer ve 4–6 GB bellek ister; eğitim sırasında diğer işler sırada bekler.

## Dil modeli: API'siz (kopyala–yapıştır) veya API

- `LLM_BACKEND=manual` (varsayılan): API gerekmez. Sistem modele gidecek metni (istem) gösterir; "İstemi kopyala"ya
  basıp Claude.ai gibi bir sohbete yapıştırırsın, yanıtı geri yapıştırırsın. Yanıt şemaya uymazsa hata gösterilir ve
  aynı adım tekrar beklenir. Belge başına genelde **1 kopyala–yapıştır** (uzun belgelerde birkaç). Verilen yanıtlar
  saklanır; aynı istem tekrar sorulmaz.
- `LLM_BACKEND=anthropic` + `ANTHROPIC_API_KEY`: her şey otomatik.
- `LLM_BACKEND=openai_compat` + `OPENAI_COMPAT_BASE_URL`: yerel model sunucusu.

## Dokploy'da çalıştırma

1. **Create Service → Compose**, kaynak bu GitHub deposu ve dal; Compose Path `./docker-compose.yml`.
2. **Environment**: `APP_PASSWORD=<şifre>` (kullanıcı adı `admin`), `NEO4J_PASSWORD=<en az 8 karakter>`.
   İsteğe bağlı: `LLM_BACKEND=anthropic` ve `ANTHROPIC_API_KEY=...`, `CONTACT_EMAIL=...`.
3. **Advanced → Isolated Deployments** açık.
4. **Domains**: Service `web`, Container Port `8000`, HTTPS açık.
5. **Deploy**, sonra domain'i aç.

Veriler kalıcı birimlerde: `kg_data` (projeler, belgeler, graflar), `neo4j_data`.

### Sorun giderme: "dependency neo4j failed to start / is unhealthy"

Dokploy'da compose servisinin **Logs** sekmesinden `neo4j` konteynerini seçip son satırlara bakın. Sık sebepler:

- **Şifre 8 karakterden kısa.** Neo4j 5 açılmaz (`Invalid value for password`). `NEO4J_PASSWORD`'ü uzatıp yeniden dağıtın.
- **Şifreyi ilk kurulumdan sonra değiştirdiniz.** `NEO4J_AUTH` yalnızca veritabanı ilk oluşturulurken uygulanır;
  sonradan değiştirilen şifre eski birimde geçmez ve web arayüzü "authentication failure" verir. Ya eski şifreye
  dönün ya da (veri yoksa) Dokploy → Volumes'tan `neo4j_data` birimini silip yeniden dağıtın.
- **Bellek yetmiyor** (log'da `OutOfMemoryError` ya da konteyner kendiliğinden yeniden başlıyor). Varsayılanlar
  düşük tutuldu (heap 512M, page cache 256M); sunucu küçükse `NEO4J_HEAP=384M`, büyükse `NEO4J_HEAP=1G` verebilirsiniz.


**Yerel deneme:** `cp .env.example .env` (şifreleri doldur), sonra
`docker compose -f docker-compose.yml -f docker-compose.local.yml up --build` → http://localhost:8000.
Docker olmadan: `pip install -e '.[web]'` ve `GRAPH_BACKEND=memory kg web`.

## Kod yapısı (web aracı)

```
kg/okg/ontology.py    ontoloji modeli, hazır şablonlar, ontolojiden JSON şema üretimi
kg/okg/standards.py   OWL/RDF/Turtle/JSON-LD içe aktarma (rdflib)
kg/okg/documents.py   PDF/DOCX/TXT/MD/HTML → metin, parçalama
kg/okg/extractor.py   çıkarım istemi + deterministik doğrulamalar, ontoloji önerisi
kg/okg/store.py       proje deposu (data/projects/<proje>/), belge katkılarıyla idempotent graf, Neo4j aynası
kg/okg/wikidata.py    Wikidata eşleme ve zenginleştirme
kg/okg/oai.py         OAI-PMH istemcisi (DSpace oai_dc, ORE tam metin), metadata grafı
kg/okg/search.py      grafta kelime araması (ilişki/varlık puanlama, vurgulama)
kg/okg/ner.py         hızlı varlık modeli: alan önerileri, dil modeliyle etiketleme (Batch API), GLiNER/sözlük eğitimi, uygulama
kg/okg/ner_cli.py     aynı boru hattının komut satırı
scripts/train_gliner.py  GLiNER'i GPU'lu bir makinede eğitmek için bağımsız betik
kg/web/               FastAPI arayüzü, iş kuyruğu, kopyala–yapıştır aracısı, şablonlar
kg/web/static/vendor/ vis-network 9.1.9 (MIT/Apache-2.0; CDN'e bağımlı olmamak için depoda)
```

---

## Ek: Kemik–beyin ekseni örnek boru hattı (komut satırı)

Projenin ilk sürümü, tek bir konu (kemik–beyin ekseni, tasarım: [SPEC.md](SPEC.md)) için iddia merkezli bir
boru hattıdır: kanıt/hipotez ayrımı, kanıt gücü puanlaması, hipotez üretimi, backtest. Web arayüzünden kaldırıldı;
komut satırından çalışmaya devam eder.

### Komut satırı

```bash
# Faz 1 — tek makale (PDF'i data/seed.pdf olarak ekleyin; JATS açık erişimde varsa o kullanılır)
kg ingest --phase 1 --pdf data/seed.pdf
#  → reports/spotcheck.html, reports/inconsistencies.md, reports/phase1_acceptance.md
#  reports/spotcheck.html'i tarayıcıda açın (ya da web arayüzündeki Spot-check sayfasını kullanın), 10 kenarı işaretleyin, JSON'u indirin:
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

### Dizin yapısı

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
  service.py     CLI ve web'in ortak servis katmanı
  web/           FastAPI arayüzü: app.py, jobs.py (arka plan iş kuyruğu), templates/, static/
cli.py           ingest | infer | ask | spotcheck | backtest | accept | unresolved | web
Dockerfile, docker-compose.yml (Dokploy), docker-compose.local.yml (yerel port yayını)
data/            ham metin, json çıkarımlar, review_queue.jsonl (git'e girmez)
reports/         hipotez kartları, tutarsızlık, backtest ve kabul raporları (git'e girmez)
```

### Notlar ve SPEC'e göre kararlar

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

### Test

```bash
pytest
```

Testler ağ, Neo4j ve API anahtarı olmadan çalışır: sahte LLM, bellek-içi depo ve açıkça "SYNTHETIC
TEST FIXTURE" olarak işaretlenmiş sentetik bir makale metni kullanır (gerçek veri değildir).
