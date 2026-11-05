# TEBDİL: girdiden sonuca hesaplama haritası

**Durum: yeni model hesabı ve deneyler kullanıcı kararıyla durduruldu.**
Başlangıçta bitmesine izin verilen `2309ea0` koşusu da sonraki açık kullanıcı
talimatıyla SIGKILL kullanılarak kesildi (çıkış 137). Hiçbir yürütme istisnası kalmadı.
Yeniden çalıştırma, devam ettirme, daha fazla token, yeni prompt, mini-model deneyi
veya controller eğitimi yok. Kısmi kayıtlar tamamlanmış bir deney değildir.
Bu belge mevcut kod ve kayıtların açıklamasıdır; yeni bir deney planı değildir.

## Önce üç ayrım

1. **Recurrent depth ≠ üretilen token.** Bir sonraki tokenı seçmeden önce aynı
   24 katmanlı blok dört kez çalışır. Dört depth, dört düşünce veya dört token değildir.
2. **Hidden state ≠ metinsel CoT.** Hidden state sayısal model aktivasyonudur;
   CoT modelin ürettiği metindir. Aktivasyon normunun değişmesi, düşüncenin
   iyileştiğini veya CoT'un iç hesabı sadakatle açıkladığını göstermez.
3. **Gerçek model çalıştırmak ≠ yöntemin işe yaradığını kanıtlamak.** Checkpoint,
   state ve token eşleşmesi yürütmeyi doğrular; başarı için tamamlanmış cevap,
   bağımsız doğrulayıcı, uygun karşılaştırma ve maliyet hesabı gerekir.

## 1. Hangi model ve hangi sorular?

- `ByteDance/Ouro-1.4B-Thinking`, revision
  `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`; tokenizer/config/ağırlık kimlikleri
  ham raporlarda SHA-256 ile bağlıdır. Ağırlıklar eğitilmiyor veya değiştirilmiyor.
- Son reasoning koşulları: BF16, Apple MPS, eager attention, batch 1,
  dört recurrent depth, son depth'ten okuma, greedy seçim, **KV cache kapalı**.
- Yeni reasoning havuzu altı development ve on iki reserved instance içerir.
  Şu ana kadarki reasoning pilotları yalnızca ilk iki development instance'ını
  tekrar kullanır: dört değişkenli CNF ve altı odalı anahtar/yol problemi.
  Bunlar held-out sonuçlar değildir; on iki seçili nokta da on iki bağımsız soru değildir.
- CNF'de herhangi bir geçerli atama, planlamada herhangi bir en kısa yasal yol
  kabul edilir. Tek bir altın cevaba metin eşleştirmesi yapılmaz.

Kaynak: [adapter](../src/tebdil/models/ouro_adapter.py),
[model config](../vendor/ouro_1_4b_thinking/config.json),
[görev/verifier](../src/tebdil/tasks/reasoning.py),
[görev protokolü](reasoning_tasks_protocol.md).

## 2. Soru modele tam olarak nasıl giriyor?

Yalnızca soru metni chat template'e verilir. `enable_thinking=True`, input'un
sonuna `assistant\n<think>\n` açılışını koyar. Bu açılış, modelin ürettiği bir token
dizisi gibi sayılmaz. CNF'nin orijinal promptu 262, planlamanınki 281 tokendır.
Çözüm, BFS sonucu, doğruluk etiketi veya gelecekteki çıktı modele verilmez.
Tarihsel arithmetic koşuları farklı/False-mode protokollerle kaydedilmiştir;
onları bu yeni True-mode kayıtlarla aynı deney gibi birleştirmiyoruz.

Kaynak: [`encode_reasoning_prompt`](../src/tebdil/evaluation/reasoning.py),
[True-mode kayıt runner'ı](../scripts/verify_prompt_started_reasoning.py).

## 3. Tek bir çıktı tokenı nasıl hesaplanıyor?

`k`, şimdiye kadar üretilmiş token sayısı olsun:

```text
orijinal prompt IDs + gerçekten üretilmiş output IDs [0:k]
  → embedding, causal attention mask, pozisyonlar 0…L−1
  → aynı 24 katmanlı blok + RMSNorm  [depth 1]
  → aynı 24 katmanlı blok + RMSNorm  [depth 2]
  → aynı 24 katmanlı blok + RMSNorm  [depth 3]
  → aynı 24 katmanlı blok + RMSNorm  [depth 4]
  → son input pozisyonunun vocabulary logits'i
  → argmax (greedy) → gerçek output token ID
  → ID'yi input dizisine ekle; sonraki token için baştan hesapla
```

Hidden-state tensörü `[1, L, 2048]` boyutundadır. Son input pozisyonu bir sonraki
tokenı tahmin eder. Tek bir "recurrent call", tek katman değil, 24 katmanlık ortak
stack'in ve RMSNorm sınırının bir yürütümüdür. Olağan token üretimi dört call yapar.
Gate hesaplanabilir fakat bu koşullarda son depth zorlanır; adaptif erken çıkış
veya hesaplama tasarrufu iddia etmiyoruz.

**Önemli:** cache kapalı olduğu için bir önceki forward'ın hidden state'i veya
KV cache'i sonraki output tokenına taşınmaz. Taşınan şey üretilmiş **token dizisi**dir.
Bir defalık latent müdahale greedy tokenı değiştirmezse, sonraki tokenın input'u
da aynı kalır. Bunun için her tokena müdahale etmek veya başka bir taşıma mekanizması
ayrı bir yöntem olurdu; şu an yapılmış gibi anlatılamaz.

Kaynak: [`native_generate`](../scripts/complete_ouro_baseline.py),
[`initialize/transition/readout_logits`](../src/tebdil/models/ouro_adapter.py),
[vendored native forward](../vendor/ouro_1_4b_thinking/modeling_ouro.py).

## 4. Gerçek hidden state'leri nasıl kaydettik?

Passive recorder, her token için aynı prefix'in dört depth'inde son pozisyonun
input, RMSNorm öncesi stack çıkışı ve normalize çıkışını gözler. Bir depth'in
normalize çıkışı sonraki depth'in input hash'iyle eşleşmelidir. State'ler model
forward'ından gelir; sentetik tensor veya uydurma PCA koordinatı değildir.

Her noktada hash/scalar kaydı vardır. Yalnızca önceden sabitlenmiş
`k=0,127,255,511,1023,2047` noktalarında gerçek son-pozisyon vektörleri de saklanır.
Bütün tokenların bütün full-prefix state'lerini saklamıyoruz. Normlar ve cosine,
aynı prefix içindeki mekanik tanılardır; anlamsal ilerleme ölçütü değildir.
Tekrar eden sekiz-token pencereleri de yalnızca heuristic aday bayrağıdır.
İlk bayraklar görev girdisini yeniden düzenlerken çıktı; doğrulanmış stall değildi.

Kaynak: [passive protokol](passive_reasoning_protocol.md),
[observer](../src/tebdil/evaluation/monitoring.py),
[gerçek kayıt bulguları](../reports/passive_reasoning_verification.md).

## 5. TEBDİL hesabı: nereye gidip neyi geri getiriyoruz?

Son sensitivity probe, aynı input prefix'inde çalışır:

1. Depth 1 sonrasında A, değiştirilmemiş kopyadır. B, aynı state'in normu ayarlı
   Gaussian pertürbasyonlu kopyasıdır (relative norm 0.25); ikisi depth 2'ye yürütülür.
2. **B bağımsız bir metinsel çözüm veya başka bir model değildir.** Aynı ağırlıklar
   kullanılır. Rastgele latent çeşitlilik, anlamlı alternatif strateji kanıtı değildir.
3. Depth 3 hesabında A'dan B yönüne geçici visit yapılır; visit'te gerçek recurrent
   fonksiyon çalıştırılır; visit'in **update'i** A'ya taşınır.
4. A'nın güncellenmiş state'i depth 4'e yürütülür ve gerçek tokenı okunur.
   Bu, B'yi kalıcı olarak A'nın yerine koymak değildir.

`F`, sabit prefix/pozisyonlarda stack **ve RMSNorm'u** içeren geçiş olsun:

```text
visit = A + clipped(alpha × (B − A))
ΔA = F(A) − A
Δvisit = F(visit) − visit
A_next = A + (1 − beta) × ΔA + beta × Δvisit
```

Bu son satır probe'un full-scope, gatesiz/projeksiyonsuz koşuludur. Alpha ve beta
0.65'tir; visit offset'i bütün-prefix Frobenius normunda `0.1 × ||A||` ile sınırlanır.
Clipping olduğunda gerçek alpha, nominal alpha değildir. BF16 dönüşümleri yuvarlama
getirir. Normalize geçiş farkını "saf residual update" diye tanımlamıyoruz.

Kontroller:

| Yöntem | Ne yapıyor? |
|---|---|
| S0 | Normal recurrence; token/state eski native kayıtla birebir eşleşmeli |
| M0 | State'leri önce karıştırıp geçiş yapar; alpha×beta için kendi clipping hesabı vardır |
| M1 | A/B'nin normal update'lerini karıştırır; visit'te fonksiyon hesaplamaz |
| E1 | Visit'te gerçek update hesaplar, A'ya geri taşır |
| Output-matched random | E1'in S0'a göre displacement normuna eşlenmiş rastgele düzeltme; semantik donor değildir |

M1'deki lineer update karışımı ile visit update'i arasındaki fark, kodda curvature
diye kaydedilir. Bu bir Hessian hesabı veya anlam/doğruluk ölçüsü değildir. Eşit
effective alpha ve gatesiz/projeksiyonsuz durumda E1−M1 ayrımı bu farkla ilişkilidir;
clipping, scope, projection ve dtype etkileri yokmuş gibi genel formül kullanılamaz.

Kaynak: [branching](../src/tebdil/core/branching.py),
[visit/return](../src/tebdil/core/excursions.py),
[kontroller](../src/tebdil/core/controls.py),
[sabit-prefix runner](../scripts/probe_reasoning_cot.py).

### Somut bir kayıt: planlama, k=1023

Bu örnek yeni inference değil, mevcut [ham probe raporundan](../reports/reasoning_cot_probe_v1/report.json):

| Girdi/hesap | Gerçek kayıt |
|---|---|
| Prefix | 281 prompt + 1023 output = 1304 token |
| Müdahale scope'u | Full prefix; hidden boyutu `[1,1304,2048]` |
| Nominal alpha / beta | 0.65 / 0.65 |
| Clipping sonrası effective alpha | **0.2077687150**; nominal 0.65 uygulanmış değil |
| S0 sonraki token | ID 3875, `" Then"` |
| E1 sonraki token | ID 3875, `" Then"` |
| E1−S0 vocabulary logit L2 | 17.687645; doğru cevap veya olasılık ölçüsü değil |
| Son normalize state hash'i, S0 | `737b76a2787f…` |
| Son normalize state hash'i, E1 | `9a1c3bf111f0…` |
| Bütün paylaşılan karşılaştırmanın işi | 16 call × 1304 = 20,864 recurrent token position |

Yani state ve logit gerçekten değişmiş, fakat greedy seçimin kazananı değişmemiş.
Bir sonraki token için input aynı kaldığından bu tek müdahaleden devam eden farklı
bir metinsel düşünce iddia edemiyoruz. Full hash'ler ham rapordadır; kısaltmalar
latent koordinat veya semantic yakınlık göstermez. Bu bir seçilmiş açıklama
örneğidir; toplam sonuç yine bütün on iki noktayı içerir.

## 6. Hangi karar gerçekten self-directed?

Bu yeni reasoning probe'unda **controller yoktur**. İki soru, altı konum ve müdahale
parametreleri önceden sabittir. CONTINUE/EXCURSION/EXIT öğrenilmiş bir policy'nin
bu sorularda doğrulanmış kararları değildir. Tarihsel controller tabloları ayrı,
exploratory ve provenance sınırlamalıdır; bunlar reasoning-rescue kanıtı yapılamaz.

## 7. Cevap nasıl puanlanıyor?

Modelin düşüncesinde geçen bir sayı veya rota cevap sayılmaz. Önce native EOS,
kapanmış `</think>` ve uygun final JSON gerekir. Verifier yalnızca üretim bittikten
sonra çalışır; inference kararlarına gold bilgi sızdırmaz.

- `budget_exhausted`: output limiti doldu; doğru/yanlış final kararı değil.
- `invalid_format`: tamamlandı ama boundary/final biçimi uygun değil.
- `incorrect`: uygun biçimli, tamamlanmış final çözüm koşulları sağlamıyor.
- `correct`: uygun final, bağımsız görev koşullarını sağlıyor.

CoT'taki uydurma `3→0` kenarı bir process hatasıdır; olmayan bir final cevabı
yanlış diye puanlama yetkisi vermez. Değişen token da kendi başına başarı değildir.

Kaynak: [boundary splitter](../src/tebdil/evaluation/reasoning.py),
[task verifier](../src/tebdil/tasks/reasoning.py).

## 8. Maliyet neden büyük; neyi sayıyoruz?

Her yeni token için büyüyen **full prefix** dört kez işlenir. Kayıtlı native helper
vocabulary logits'ini bütün input pozisyonları için de hesaplar (`logits_to_keep=0`),
oysa token seçimi yalnızca son pozisyonu kullanır. Bu doğruluk/provenance odaklı,
optimize edilmemiş yürütümdür. Cache veya daha dar readout'a geçmek şu an yapılmaz;
aynı davranışı koruduğu ayrıca incelenmeden sessizce değiştirilemez.

Call sayacı RMSNorm sınırının gerçek yürütümlerini sayar. Recurrent token positions
`Σ(call sayısı × gerçek prefix uzunluğu)` şeklinde toplanır; FLOPs değildir.
Duvar saati model hesabı yanında hash, observer, CPU transferi ve journal I/O'yu
da içerir. Hangi bileşenin ne kadar zaman aldığını profillemedik; darboğaz yüzdesi
uyduramayız. Bu süreyi optimize GPU throughput'u olarak sunamayız.

Önceki passive pair 16,384 call yaptı. Sensitivity probe'un gerçek yeni işi 192
call'dur: her konumda toplam 16, referans/branch/visit/kontrol işleri dahil.
Paylaşılan referans hesabı olduğu için bunlar ayrı ayrı compute-matched arm'lar
değildir. Budget follow-up'ta eski 16,384 call yeniden yapılmış gibi yazılmaz;
terminal replay ve gerçek yeni suffix üretimi ayrıca sayılır.

## 9. Şimdiye kadar hangi bulgulara sahibiz?

- Gerçek checkpoint ve native/adapter sınırları üzerinde sınırlı yürütüm kanıtı var.
- İki orijinal reasoning baseline'ı 2048 tokenda tamamlanmadı.
- On iki sabit context'te E1/M0/M1/random, hidden state/logitleri değiştirdi ama
  greedy next-token değişimi hepsinde **0/12** kaldı. Yeni conditional CoT penceresi
  çalıştırılmadı. Genel etkisizlik veya görev uygunsuzluğu sonucu çıkmaz.
- Sabit bütçe follow-up'ının kapanış durumu ve ham kayıtları
  [ayrı doğrulama raporundadır](../reports/reasoning_budget_verification.md).
- Doğrulanmış **stuck→TEBDİL→çözüm** örneği, held-out rescue oranı veya bu
  reasoning koşullarında doğrulanmış learned controller sonucu yok.

## 10. Compute yeniden düşünülmeden açıklığa kavuşması gerekenler

Bu liste çalışma izni değil, birlikte kod/kayıt üzerinden cevaplayacağımız sorulardır:

1. Müdahalenin nesnesi hangi prefix'teki, hangi depth'teki hangi tensor? Normalize
   state üzerinde update taşımanın teorik anlamı nedir?
2. Donor neden yararlı bir alternatif olsun? Rastgele sapma ile farklı akıl yürütme
   stratejisini nasıl ayıracağız?
3. Müdahale tokenı değiştirmiyorsa etkisi neden devam etsin? Tek/tekrarlı müdahale
   ve cache tasarımı hangi farklı yöntemleri tanımlar?
4. Modelin task/prompt/decoding koşullarında tamamlanmış baseline üretmesini
   doğrulamadan “zor soru”, “stuck” veya “rescue” diyebilir miyiz?
5. Tetikleme zamanı nasıl gold ve gelecek çıktı görmeden seçilecek?
6. Hangi sonuç başarı sayılacak; rescue kadar bozulan doğru cevapları ve bütün
   referans/donor/monitor maliyetlerini nasıl koruyacağız?
7. Token bütçesine ek olarak açık süre/bellek sınırı ve durdurma kaydı ne olacak?

Yeni experiment ancak bu süreç anlaşılıp **kullanıcı açıkça compute'u yeniden
başlatmaya karar verdikten sonra** gündeme gelir. O zamana kadar kaynak okuma,
mevcut artifact denetimi, açıklama ve statik/recorded Pages doğrulaması yapılabilir.
