# Kemik–Beyin Ekseni Knowledge Graph — Sistem Spec

2026-10-15 · Serhat · Kaynak: Claude Docs belgesi "Kemik–Beyin Ekseni Knowledge Graph — Sistem Spec"

## Amaç ve ilkeler

Sistem, kemik–beyin ekseni literatürünü iddia merkezli bir knowledge graph'a dönüştürür, uzman olmayan bir
kullanıcının soru sormasına ve sistemin otonom hipotez üretmesine izin verir. Başlangıç makalesi: Cai ve ark.,
*Tibial bone compression promotes recovery after brain injury through osteocyte PIEZO1*, Nature Neuroscience
(2026), doi:10.1038/s41593-026-02422-w.

Kapsam: bu makale, kaynakçası, ona atıf yapan çalışmalar ve PIEZO1, HSP70, IL-1R2, APOL11a, BDNF, PF4, osteokin
ve TBI/inme anahtar kavramlarından genişleyen 1–2 atlama.

Değişmez ilkeler:

1. **Kanıt ve hipotez asla karışmaz.** Makalelerden çıkarılan ilişkiler `SUPPORTS`/`CONTRADICTS`, sistemin
   ürettikleri `HYPOTHESIS` kenarlarıdır. Bir hipotez, ancak bir makale onu test ettiğinde kanıta dönüşür.
2. **Her kenarın kaynağı vardır.** Makale DOI'si, şekil/panel ve kaynak cümle (en fazla 300 karakter) zorunludur.
   Kaynaksız kenar yazılmaz.
3. **Belirsizlik açık yazılır.** Çıkarım modeli emin değilse `extraction_confidence` düşük işaretlenir; sayı
   uydurulmaz, bilinmeyen alan boş kalır.
4. **Tür ve bağlam korunur.** Fare in vivo, hücre kültürü ve insan bulguları ayrı tutulur; fare geni ile insan
   geni ayrı düğümdür, ortolog kenarıyla bağlanır.
5. **Kullanıcı uzman değildir.** Her cevap kanıt düzeyini düz Türkçe ile söyler ("sadece farede, tek
   laboratuvar, körlemesiz").

## Mimari

Sistem tek yönlü bir boru hattı ile kanıt grafını besler; çıkarım katmanı grafı yalnızca okur ve hipotezleri
ayrı bir çıktı olarak üretir.

```
keşif → tam metin → bölümleme → LLM çıkarımı → öz-doğrulama → normalizasyon → yükleme (kanıt grafı)
                                                                                  │ salt-okunur
                         literatür kontrolü ← hipotez (rapor + HYPOTHESIS) ← çıkarım katmanı
                                 │ yeni makale
                                 └──────────────→ keşif (boru hattına geri)
```

Bir hipotez, ancak literatür kontrolü onu test eden yeni bir makale bulup bu makale boru hattından geçtiğinde
kanıt grafına girer; sistem kendi tahminini kanıt sanamaz.

## Graf şeması

Şema iddia (Claim) etrafında kurulur: varlıklar arası her ilişki bir deneyden gelir ve deneyin kanıt
nitelikleri kenarda taşınır.

**Düğüm tipleri**

| Düğüm | Zorunlu özellikler | Kimlik kaynağı |
|---|---|---|
| Paper | doi, title, year, journal, lab/affiliation, retracted, funding_source (akademik / endüstri / karma) | DOI |
| Experiment | figure, panel, design (in vivo / in vitro / ex vivo), species, sex, n_per_group, blinded, randomized, tracer_type (³H, ¹³C, döteryum, NBD, BODIPY, yok) | paper DOI + figure + panel |
| Claim | text (tek cümle, normalize), polarity (+/−), status (evidence / hypothesis / refuted) | hash(normalize text) |
| Gene / Protein | symbol, species, uniprot_id | MGI / HGNC / UniProt |
| Intervention | name, type (mekanik, ilaç, gen, antikor), dose, schedule | serbest + ChEBI varsa |
| Model | name (CCI-TBI, MCAO, domuz CCI), severity | serbest |
| CellType | name | Cell Ontology |
| Outcome | measure (pole test, MWM, MAP2, Nissl, sağkalım), direction | serbest |
| Disease | name | MeSH |

**Kenar tipleri**

| Kenar | Yön | Not |
|---|---|---|
| REPORTS | Paper → Experiment | |
| SUPPORTS / CONTRADICTS | Experiment → Claim | p, effect_size, source_sentence zorunlu |
| SUBJECT / OBJECT | Claim → varlık | iddianın özne ve nesnesi |
| INCREASES / DECREASES / REQUIRES / MEDIATES | varlık → varlık | yalnızca bir Claim'den türetilir, `claim_id` taşır |
| ORTHOLOG_OF | Gene → Gene | fare ↔ insan |
| CITES | Paper → Paper | |
| REPLICATES | Experiment → Experiment | farklı laboratuvar şartı |
| HYPOTHESIS | varlık → varlık | sadece çıkarım katmanı yazar; `path`, `score`, `rationale`, `created_at` |

Her kenar ortak alanlar taşır: `source_doi`, `figure`, `source_sentence`, `extraction_model`,
`extraction_confidence` (0–1), `verified_by_human` (bool).

## Veri toplama ve bilgi çıkarımı

Boru hattı idempotent çalışır: her makale DOI ile anahtarlanır, tekrar çalıştırmak çift kayıt üretmez.

1. **Keşif.** Europe PMC REST API ve Semantic Scholar Graph API ile tohum makalenin referansları ve atıf
   yapanlar çekilir; anahtar kavram sorgularıyla 1–2 atlama genişletilir. Varsayılan üst sınır 200 makale.
   Retraction durumu Crossref/Retraction Watch'tan kontrol edilir.
2. **Tam metin.** Önce Europe PMC açık erişim XML (JATS), yoksa PDF (`pymupdf`). Kapalı erişimde yalnızca özet
   işlenir ve `fulltext=false` işaretlenir.
3. **Bölümleme.** Metin bölümlere ayrılır; şekil açıklamaları (figure legends) ayrı tutulur, çünkü n ve p
   değerleri büyük ölçüde oradadır.
4. **LLM çıkarımı.** Her bölüm için sabit JSON şemasıyla (Pydantic) Claim, Experiment ve varlıklar çıkarılır.
   Prompt kuralları: yalnızca metinde açıkça yazanı çıkar; kaynak cümleyi birebir kopyala; bilinmeyen alan
   `null`; fare/insan/hücre ayrımını koru.
5. **Öz-doğrulama.** İkinci bir LLM çağrısı her çıkarımı kaynak cümlesiyle karşılaştırır ve
   `supported / partial / unsupported` etiketler. `unsupported` olanlar grafa girmez, `review_queue.jsonl`
   dosyasına yazılır.
6. **Normalizasyon.** Gen/protein adları MGI, HGNC ve UniProt'a; hastalıklar MeSH'e eşlenir. Eşlenemeyen varlık
   `unresolved=true` ile tutulur, yeni düğüm açılmadan önce bulanık eşleşme önerisi üretilir.
7. **Yükleme.** Neo4j'e `MERGE` ile yazılır; her çalıştırma bir `ingest_run` kaydı bırakır (tarih, model,
   makale sayısı, reddedilen çıkarım sayısı).

Örnek çıkarım çıktısı (tek iddia):

```json
{
  "claim": "Osteocyte-specific Piezo1 deletion abolishes the motor and cognitive benefit of tibial loading after TBI",
  "polarity": "+",
  "subject": {"type": "Gene", "symbol": "Piezo1", "species": "mouse"},
  "object": {"type": "Outcome", "measure": "pole test, MWM"},
  "experiment": {"figure": "Fig. 3", "panel": "d-h", "design": "in vivo", "species": "mouse", "sex": "male", "n_per_group": [14, 14, 14, 17], "blinded": false},
  "source_sentence": "DCTAL ameliorated motor and cognitive deficits in TBI-Piezo1fl/fl mice, but not in TBI-Piezo1Sost-Cre/ERT2 mice",
  "extraction_confidence": 0.9
}
```

## Otonom çıkarım katmanı

Çıkarım katmanı yalnızca `HYPOTHESIS` kenarı ve hipotez raporu yazar; kanıt kenarlarına dokunamaz. Üç üretici
paralel çalışır, adayları tek bir kuyrukta birleştirir.

1. **Kural tabanlı zincirler.** 2–3 adımlı yollar aranır: `A -INCREASES-> B -PROTECTS-> C` ve `A -[*]-> C` yoksa
   aday üretilir. İzinli zincir kalıpları `rules.yaml` dosyasında tanımlıdır (örn. "müdahale → molekül →
   koruyucu etki", "molekül A bloke edilince etki kayboluyor + molekül A başka modelde de var").
2. **Bağlantı tahmini.** PyKEEN ile (RotatE veya ComplEx) kanıt kenarları üzerinde embedding eğitilir; skoru
   eşiğin üstündeki eksik kenarlar aday olur. Model asla `HYPOTHESIS` kenarlarıyla eğitilmez.
3. **LLM ajanı.** Grafı salt-okunur Cypher ile gezen bir ajan; kullanıcı sorusu veya periyodik görev ("en az test
   edilmiş ama en güçlü bağlı molekülleri bul") ile tetiklenir. Ajan her adayı yol (path) ve gerekçesiyle döner.

Her aday için zorunlu adımlar:

1. **Tekilleştirme.** Aynı (özne, ilişki, nesne) üçlüsü varsa skor güncellenir, yeni kenar açılmaz.
2. **Literatür kontrolü.** Ajan Europe PMC'de "bu test edilmiş mi?" araması yapar. Destekleyen veya çürüten
   makale bulunursa boru hattına eklenir ve hipotezin durumu `tested_supported`, `tested_refuted` ya da
   `untested` olur.
3. **Rapor.** Her hipotez düz Türkçe bir kart üretir: iddia, zincir (makale + şekil), en zayıf halka, güven
   skoru, "bunu test etmek için gereken deney" önerisi.

Yasaklar: ajan grafa yazma yetkisine sahip değildir (yazma ayrı bir servisten geçer); kendi ürettiği hipotezi
başka hipotezin kanıtı olarak kullanamaz; insan klinik önerisi üretmez.

## Güven puanlama

Her kanıt kenarı 0–1 arası bir kanıt gücü (E) taşır; bir hipotezin zinciri en zayıf halkası kadar güçlüdür.
Katsayılar başlangıç değeridir ve `scoring.yaml` dosyasından ayarlanır.

Kanıt gücü: `E = w_design · w_species · w_n · w_blind · w_repl · c_extract`

| Faktör | Değerler |
|---|---|
| w_design | RCT insan 1.0 · gözlemsel insan 0.7 · in vivo hayvan 0.5 · ex vivo 0.35 · in vitro 0.25 |
| w_species | insan 1.0 · domuz/primat 0.8 · kemirgen 0.6 · hücre hattı 0.4 |
| w_n | n ≥ 20: 1.0 · 10–19: 0.8 · 5–9: 0.6 · n ≤ 4: 0.4 |
| w_blind | körlemeli 1.0 · körlemesiz veya belirtilmemiş 0.7 |
| w_repl | bağımsız laboratuvar replikasyonu: yok 0.6 · 1 tane 0.85 · 2+ tane 1.0 · çürüten çalışma varsa ×0.5 |
| c_extract | çıkarım güveni; `verified_by_human` ise 1.0 |

Zincir güveni (L adımlı yol): `C = min_i(E_i) · Π_j p_j · 0.8^(L-1)`

Uyumsuzluk cezaları (p): halkalar arasında tür farkı ×0.6, doku/hücre tipi farkı ×0.7, doz veya rejim belirsiz
×0.8, ilişki yönü çelişkili ×0.3.

Nihai hipotez skoru, embedding tahmini (P) varsa birleşik, yoksa yalnızca C: `S = 0.6 · C + 0.4 · P`

Raporda skor yanında her zaman en düşük E'li halka ve nedeni gösterilir (örn. "Fig. 5h, fare, n = 9–10,
körlemesiz").

## Doğrulama

Sistemin güvenilirliği üç katmanda ölçülür; uzman olmayan kullanıcı ilk ikisini kendisi yapabilir.

1. **Çıkarım doğruluğu (spot-check).** Her ingest sonrası rastgele 10 kenar, kaynak cümlesiyle yan yana
   `spotcheck.html` sayfasında gösterilir; kullanıcı "doğru / kısmen / yanlış" işaretler. Hata oranı %10'u aşarsa
   çıkarım promptu düzeltilir ve o partinin kenarları `needs_review` olur.
2. **Sayısal tutarlılık.** Otomatik kontroller: n değerleri şekil açıklamasıyla eşleşiyor mu, aynı deney için metin
   ve açıklama çelişiyor mu (örn. dışlama bildirimi, protokol parametreleri). Çelişkiler `inconsistencies`
   raporuna yazılır; tohum makalede bu tür en az iki tutarsızlık bilindiği için bu bir kabul testi olarak
   kullanılır.
3. **Backtesting.** Graf belirli bir kesim tarihine (örn. 2024-01-01) kadar olan makalelerle kurulur, hipotezler
   üretilir, sonra kesim sonrası yayımlanmış makalelerde bu hipotezlerin test edilip edilmediğine bakılır.

| Metrik | Tanım | Hedef (ilk sürüm) |
|---|---|---|
| Precision@20 | İlk 20 hipotezden sonradan desteklenenlerin oranı | rastgele çizgiden anlamlı yüksek |
| Refuted@20 | İlk 20 hipotezden sonradan çürütülenlerin oranı | raporlanır, düşük olmalı |
| Rastgele çizgi | Aynı tip düğüm çiftlerinden rastgele 20 hipotezin destek oranı | karşılaştırma için |
| Kalibrasyon | Skor dilimleri ile gerçekleşme oranı arasındaki ilişki | monoton artan |

Sayısal hedef eşikleri ilk backtest sonucuna göre belirlenecek; açık soru olarak bırakılmıştır.

## Teknoloji ve dizin yapısı

Python 3.11+, Neo4j 5 (Docker), Pydantic şemaları, PyKEEN, LLM erişimi Anthropic API veya yerel model (OpenAI
uyumlu uç nokta, `.env` ile seçilir). Tüm ayarlar YAML; gizli anahtarlar yalnızca `.env`.

```
kg-bone-brain/
  docker-compose.yml        # neo4j
  .env.example
  config/
    scoring.yaml            # güven katsayıları
    rules.yaml              # izinli zincir kalıpları
    sources.yaml            # tohum DOI, sorgu terimleri, üst sınırlar
  kg/
    schema.py               # Pydantic modelleri
    discover.py             # Europe PMC / Semantic Scholar
    fetch.py                # JATS / PDF
    extract.py              # LLM çıkarımı + öz-doğrulama
    normalize.py            # MGI / HGNC / UniProt / MeSH
    load.py                 # Neo4j MERGE
    infer/
      rules.py
      embed.py              # PyKEEN
      agent.py              # salt-okunur Cypher ajanı
    score.py
    validate/
      spotcheck.py
      consistency.py
      backtest.py
  cli.py                    # ingest | infer | ask | spotcheck | backtest
  data/                     # ham metin, json çıkarımlar, review_queue.jsonl
  reports/                  # hipotez kartları (markdown)
```

Örnek sorgular:

```cypher
// Zayıf kanıta dayanan iddialar
MATCH (e:Experiment)-[s:SUPPORTS]->(c:Claim)
WHERE e.blinded = false AND all(n IN e.n_per_group WHERE n <= 10)
RETURN c.text, e.figure, e.species, e.n_per_group ORDER BY c.text;

// Bir molekül için bağımsız laboratuvar sayısı
MATCH (g:Gene {symbol:'Hspa1a'})<-[:SUBJECT|OBJECT]-(c:Claim)<-[:SUPPORTS]-(:Experiment)<-[:REPORTS]-(p:Paper)
RETURN c.text, count(DISTINCT p.lab) AS labs ORDER BY labs DESC;

// En yüksek skorlu test edilmemiş hipotezler
MATCH (a)-[h:HYPOTHESIS {status:'untested'}]->(b)
RETURN a.name, b.name, h.score, h.weakest_link ORDER BY h.score DESC LIMIT 20;
```

## İkinci tohum seti: kolesterol izleme

İkinci set, damar tıkanıklığına yol açan kolesterolün izini süren çalışmalardan oluşur. Bilerek birbirini sınayan
makaleler seçildi: biri LDL nedenselliğini, biri HDL nedenselliğini çürüten genetik kanıtı, üçü izleme yöntemini
fareden insana, biri plak görüntülemesini temsil eder.

| Makale | Rol | Grafa katkısı |
|---|---|---|
| Ference ve ark., Eur Heart J 2017 | LDL nedenselliği konsensüsü | insan, yüksek kanıt uç noktası |
| Voight ve ark., Lancet 2012 | HDL için Mendelyen randomizasyon | `CONTRADICTS` testi: gözlemsel destek, genetik destek yok |
| Zhang, Rader ve ark., Circulation 2003 | ³H-kolesterol makrofaj RCT tahlili (fare) | `tracer_type` = ³H, yöntem kökü |
| Cuchel ve ark., J Lipid Res 2017 | aynı yöntem insanda | fare → insan zinciri |
| Annema & Tietge, Nutr Metab 2012 | makrofaj RCT çalışmaları derlemesi | genişleme merkezi (hub) |
| Waksman ve ark. (LRP), Lancet 2019 | NIRS-IVUS lipid çekirdek yükü ve olaylar | `funding_source` = endüstri |

Kurallar: `funding_source` = endüstri olan makaleler skoru otomatik düşürmez, hipotez kartında ayrı bir uyarı
satırıyla gösterilir. `tracer_type` = NBD veya BODIPY olan deneylerde kartta "floresan grup molekülün davranışını
değiştirebilir" notu çıkar.

(`config/sources.yaml` bu seti içerir.)

## Fazlar ve kabul kriterleri

Her faz bir öncekinin kabul kriterleri geçmeden başlamaz.

| Faz | Kapsam | Kabul kriteri |
|---|---|---|
| 1. Tek makale | Tohum makaleden şema + çıkarım + Neo4j yükleme | Ana iddiaların (Piezo1 gerekliliği, serum transferi, HSP70/APOL11a/IL-1R2, domuz) her biri kaynak cümleli Claim olarak var; spot-check hata oranı ≤ %10; bilinen iki tutarsızlık `inconsistencies` raporunda |
| 2. Genişleme | Referanslar + atıf yapanlar, 1–2 atlama, ≤ 200 makale | Normalizasyon oranı ≥ %90; tekrar çalıştırmada çift kayıt yok; retraction kontrolü çalışıyor |
| 3. Çıkarım | Kural zincirleri + LLM ajanı + skor + hipotez kartları | Her hipotezin yolu, en zayıf halkası ve literatür kontrol sonucu var; hiçbir HYPOTHESIS kenarı kanıt sorgusunda dönmüyor |
| 4. Değerlendirme | PyKEEN + backtest | Backtest raporu Precision@20'yi rastgele çizgiyle karşılaştırıyor |

**Claude Code için başlatma talimatı**

```markdown
Bu repodaki SPEC.md dosyasını oku ve Faz 1'i uygula.
- Tohum: doi 10.1038/s41593-026-02422-w (PDF data/seed.pdf olarak eklendi).
- Neo4j'i docker-compose ile ayağa kaldır, şemayı kg/schema.py'de Pydantic ile tanımla.
- Çıkarımda kaynak cümleyi birebir kopyala, bilmediğin alanı null bırak, uydurma.
- Faz 1 bitince: spotcheck.html üret, inconsistencies raporunu çıkar,
  kabul kriterlerini tek tek kontrol edip sonucu bana raporla. Faz 2'ye benim onayım olmadan geçme.
```
