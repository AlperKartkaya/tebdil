# TEBDİL — Cooperative Latent Excursions for Recurrent Reasoning

**Araştırma tasarımı · Çok fazlı geliştirme yol haritası · Teknik mimari · Deney protokolü · Coding-agent master prompt**  
**Sürüm:** 1.0 · **Hazırlanma tarihi:** 27 Ekim 2026  
**Çalışmanın türü:** Açık kaynak, tekrarlanabilir, deneysel bilgisayar bilimi / ML lisans bitirme projesi  
**Durum:** Araştırma hipotezi ve uygulama spesifikasyonu. Bu belgede anlatılan deneylerin gerçekleştirildiği veya beklenen sonuçların elde edildiği iddia edilmez.

> **Ana araştırma sorusu:** Bir recurrent/looped language model (LoopLM), verimsiz veya yanlış bir latent hesaplama rotasında ilerliyorsa, bağımsız bir rotanın latent komşuluğunu **geçici olarak ziyaret ederek** orada hesaplanan güncelleme yönünü kendi rotasına taşıdığında, aynı hesaplama bütçesiyle daha sık doğru sonuca ulaşabilir mi?
>
> **Bir cümlede sistem:** Tek bir açık ağırlıklı LoopLM'nin paylaşılan ağırlıklarıyla birden çok hidden-state trajectory üret; gerektiğinde bir dalı diğerinin latent bölgesine doğru geçici olarak taşı; recurrent transition'ı orada çalıştır; yalnızca yararlı olabilecek **yerel güncelleme bilgisini** orijinal dala geri aktar; sonuçları eşit hesaplama maliyetli güçlü baseline'larla karşılaştır.

---

## 0. Bu doküman nasıl kullanılacak?

Bu dosya **tek yetkili proje spesifikasyonu** ve kodlama ajanına verilecek ana talimattır. Kodlama ajanı:

1. Önce **bölüm 1–9**'u okuyarak kavramları, bilimsel iddiayı ve deneysel sınırları anlayacak.
2. **Bölüm 10–17**'deki mimariyi, deney koşullarını ve ölçüm düzenini gerçekleştirecek.
3. **Bölüm 18–24**'teki fazları sırasıyla uygulayacak; bir fazın kabul kriterlerini geçmeden sonraki fazı tamamlanmış saymayacak.
4. **Bölüm 25–29**'daki kaynaklar, riskler, raporlama ve ajan çalışma kurallarına uyacak.
5. Özellikle **bölüm 30 — Coding Agent Master Instruction**'ı yürütme görevi kabul edecek.

Bu proje “güzel görünen latent-space animasyonu” üretmek için yapılmıyor. Öncelikli çıktı **doğrulanabilir bir mekanizma, sağlam kontrol deneyleri ve bilimsel olarak yorumlanabilir sonuçlar**. Görselleştirme önemli; ancak analizin yerini alamaz.

### Yapılacak / yapılmayacak

| Kapsamda | İlk sürümde kapsam dışında |
|---|---|
| Tek model, paylaşılan ağırlıklar, birden fazla recurrent-state dalı | Her dal için farklı bir LLM eğitmek |
| Hidden-state perturbation, excursion ve return | Doğal dilde “başka açıdan düşün” prompt'u vermek |
| Açık LoopLM'ye inference-time müdahale | Üretim ortamına agent dağıtmak |
| Başarısız trajectory'nin kurtarılmasını ölçmek | “Modelin gerçek düşüncesini okuduk” iddiası |
| Eşit compute, doğru negatif kontroller ve ablation'lar | Sadece rastgele birkaç başarılı örnek göstermek |
| Sonraki fazda adaptif seçim ve küçük controller | İlk günden RLHF veya büyük ölçekli fine-tuning |
| Model davranışını doğrulayan küçük formal problem aileleri | İzin gerektiren sistemlere yönelik gerçek siber istismar |

---

## 1. Nihai vizyon ve başarı tanımı

### 1.1 Nihai ürün

**TEBDİL** dört parçalı bir açık kaynak araştırma platformu olacak:

1. **Instrumented LoopLM Runtime:** LoopLM'nin recurrent geçişlerinde hidden state yakalayan, dallandıran, dışarıdan kontrollü müdahale kabul eden, yeniden sürdüren ve çıktıları kaydeden inference katmanı.
2. **Cooperative Latent Excursion Engine:** Bağımsız dallar, ziyaret edilecek latent bölgenin seçilmesi, ziyaret sırasında transition hesaplanması ve geri dönüş için kurallar.
3. **Scientific Evaluation Suite:** Matematiksel/doğrulanabilir görevler, bağımsız verifier'lar, compute-matched baseline'lar, tekrarlar, belirsizlik aralıkları ve mekanizma ablation'ları.
4. **TEBDİL Observatory:** Recurrent step ve branch bazında latent uzaklıkları, cevap olasılıkları, müdahale anı, kurtarma oranları, maliyetler ve vaka replay'leri gösteren arayüz.

### 1.2 Nihai bilimsel hedef

Önceden seçilmiş değerlendirme görevlerinde, müdahale yapılmayan bağımsız bir dalın başarısız kaldığı durumlarda, **oracle bilgisi kullanmadan** seçilen bir excursion'ın, aynı hesaplama bütçesindeki bağımsız sampling, daha fazla recurrence, state mixing ve update mixing kontrol gruplarına karşı **başarısız trajectory'leri daha yüksek oranda kurtarabildiğini** göstermek.

İddia, yalnızca tek bir kazanan örnekten değil; ayrılmış test kümesindeki tekrarlı karşılaştırmalardan ve maliyet analizinden destek almalı.

**İki farklı başarı düzeyi:**

- **Asgari başarılı bitirme projesi:** Gerçek LoopLM'de müdahale sağlam biçimde çalışıyor; kontrollü veri ve baseline'lar üzerinden hangi koşullarda işe yarayıp yaramadığına dair güvenilir **pozitif veya negatif** sonuç üretiliyor.
- **İddialı bilimsel başarı:** TEBDİL'in seçici excursion'ı test kümelerinde, maliyet eşitliğinde rakipleri aşabiliyor ve yararın nedenini açıklayan nedensel analizler sunuluyor.

**Başarısız hipotez = başarısız proje değildir.** Doğru kontrol deneyleriyle elde edilmiş güçlü bir negatif sonuç araştırma çıktısıdır. Bununla birlikte iyi bir demo ile araştırma sonucunu karıştırmayacağız.

### 1.3 Test edilebilir hipotezler

- **H1 — Feasibility:** Aynı pretrained recurrent model, iki veya daha çok tutarlı latent branch üzerinde ayrı ayrı çalıştırılabilir; gizli durum ziyaretleri sayısal ve anlamsal olarak ölçülebilir.
- **H2 — Recovery:** Belirli başarısız branch'lerde excursion-and-return, doğrudan mixing ve norm-matched random perturbation'dan daha fazla doğruya dönüş sağlayabilir.
- **H3 — Mechanism:** Fayda varsa, yalnızca branch'lerin durumlarını karıştırmaktan değil, **ziyaret edilen yerde recurrent transition hesaplamaktan** doğan nonlineer bileşenin katkısı vardır.
- **H4 — Selectivity:** Adaptif olarak **ne zaman, kimi ve ne kadar** ziyaret edeceğini seçen düşük maliyetli yöntem, her seferinde kör müdahaleden daha avantajlı olabilir.
- **H5 — Efficiency:** Aynı toplam model hesaplamasında elde edilen doğruluk / güvenilirlik artışı, yöntemin ek maliyetini karşılayabilir.

Bu hipotezleri kesin bulgular gibi sunma. Her biri yanlışlanabilir olacak şekilde deney tasarla.

---

## 2. Temel kavramlar: “mekân”, “tebdil”, “kurtarma”

### 2.1 Mekân nedir?

“Mekân” sözcüğünü **dilsel bağlam** veya **fiziksel ortam** anlamında kullanmıyoruz. İlk sürüm için:

> **Mekân = belirli bir model, sabit bir input ve aynı recurrent step altında, modelin geçerli hidden-state tensörlerinin oluşturduğu yüksek boyutlu hesaplama durum uzayı.**

Kavramsal olarak:

\[
H_t\in\mathbb{R}^{B\times L\times d}
\]

Burada \(B\) batch, \(L\) token dizisi uzunluğu, \(d\) hidden boyutudur. Bu tensör **semantic embedding** ile aynı nesne değildir: devam eden hesaplamanın durumudur. İçinde attention'ın bağlam ilişkileri, residual bilgi ve sonraki katmanların kullanacağı başka nicelikler bulunabilir.

Bir modelin bütün hidden state geometrisini tek bir PCA grafiğinden çıkarsayamayız. “Başka bölgede” olmak, seçtiğimiz tam-dimensiyonlu metrikte, aynı recurrent step'teki iki farklı duruma sahip olmak demektir.

### 2.2 Tebdil nedir?

> **Tebdil = Bir trajectory'nin mevcut durumundan geçici olarak başka bir latent komşuluğa gidip aynı recurrent transformation'ı orada çalıştırması ve elde edilen güncelleme yönünü kendi başlangıç durumuna geri taşıması.**

Bu bir **context reset**, yeni prompt veya iki state'in basit aritmetik ortalaması değildir.

### 2.3 Kurtarma nedir?

Kurtarma tanımını sonuçtan bağımsız biçimde deney protokolünde sabitle:

- **Başlangıç hatası:** Referans branch A, sabit decode politikasıyla yanlış / geçersiz cevap üretiyor.
- **Excursion sonrası:** Başlangıçtaki token prefix ve problem korunurken branch A' üzerinde uygulanan müdahale sonucunda doğru / geçerli cevap üretiliyor.
- **Net recovery:** Aynı örnek üzerinde hatalı→doğru geçişler eksi doğru→hatalı geçişler, ilgili paydalarla birlikte raporlanıyor.
- **Oracle recovery:** A'nın yanlış, B'nin doğru olduğu **sonradan etiketlenen** çiftler üzerinde mekanizma testi. Bu, mekanizmanın kapasitesini değerlendirir, deploy edilebilir algoritma skoru değildir.
- **Non-oracle recovery:** Donor veya tetikleyici **cevap doğruluğu veya gold label görmeden** seçilir. Asıl nihai araştırma metriği budur.

“Yanlış attractor basin'den kaçış” ancak attractor/basin'e ilişkin ek deneysel kanıt varsa kullanılabilir. Dört step'lik modelde yalnızca yanlış cevaptan doğru cevaba geçiş görmüş olmak attractor kanıtı değildir.

---

## 3. Matematiksel model: TEBDİL algoritması

### 3.1 Gerçek model bağımlılığı

Basit anlatımda \(F_\theta(H)\) kullanacağız. Fakat gerçek LoopLM'de transition şu biçimde daha doğru modellenir:

\[
H_{t+1}=F_{\theta,t}(H_t;X, M, P, C_t)
\]

- \(t\): recurrent step; bazı modellerde `current_ut` gibi koşullandırma olabilir.
- \(X\): input / embedding yapısı.
- \(M\): attention mask.
- \(P\): positional / RoPE bilgisi.
- \(C_t\): varsa cache ve diğer yürütme durumu.

**En önemli implementasyon ilkesi:** A, B ve ziyaret edilen state karşılaştırılırken aynı recurrent step, aynı token pozisyonları, aynı mask ve anlamlı şekilde aynı yan koşullar kullanılmalı. Cache tutarlılığı bozulursa gizli durum müdahalesinin etkisi karıştırıcı değişkenlerle ölçülmüş olur.

### 3.2 İki dal

Aynı input ve checkpoint için, ilk ortak recurrent geçişten sonra iki bağımsız durum oluştur:

\[
H_1^A=H_1,\qquad H_1^B=H_1+\epsilon
\]

\(\epsilon\) normu sınırlandırılmış, kontrollü bir perturbation. Alternatifler sonradan: dropout, farklı projeksiyon altuzayları ve learned branching. İlk fazda model eğitimi yok.

A ve B eşit step sayısıyla ayrı ilerler. Modelin ağırlıkları aynıdır, her branch için yalnızca **aktivasyon durumu** ayrıdır.

### 3.3 Mekân ziyareti

A için normal update:

\[
V_t^A=F_{\theta,t}(H_t^A)-H_t^A.
\]

B'nin bulunduğu yere doğru normalize edilmiş ziyaret yönü:

\[
D_t=H_t^B-H_t^A,
\qquad
\widehat D_t=\frac{D_t}{\|D_t\|_F+\varepsilon}.
\]

Ziyaret:

\[
H_t^{\mathrm{visit}}=H_t^A+\rho_t\widehat D_t.
\]

Burada \(\rho_t\), ziyaret mesafesi için **açıkça sınırlandırılmış bir norm bütçesi**. Kullanıcıya sezgisel gelen parametrizasyonla da test yapılabilir:

\[
H_t^{\mathrm{visit}}=(1-\alpha)H_t^A+\alpha H_t^B,
\qquad \alpha\in[0,1].
\]

Bu son form norm kontrollü olmak zorunda değildir; ikisini ayrı ablation olarak değerlendir.

Ziyaret edilen durumdaki update:

\[
V_t^{\mathrm{visit}}
=F_{\theta,t}(H_t^{\mathrm{visit}})-H_t^{\mathrm{visit}}.
\]

Geri dönüş:

\[
\boxed{
H_{t+1}^{A'}=
H_t^A+(1-\beta)V_t^A+\beta V_t^{\mathrm{visit}}
}
\qquad \beta\in[0,1].
\]

B'nin ana trajectory'si değiştirilmez. **Geri taşıdığımız nesne, B'nin cevabı veya bütün state'i değil, ziyaret edilen noktadaki bir transition yönüdür.**

### 3.4 Mekanizma için kritik kontrol: Doğrusallık ve eğrilik

Update mixing kontrolü:

\[
V_t^{\mathrm{mix}}
=(1-\alpha)V_t^A+\alpha V_t^B.
\]

Ziyaret katkısının basit mixing'den ayrışan kısmı:

\[
\boxed{
C_t=V_t^{\mathrm{visit}}-
[(1-\alpha)V_t^A+\alpha V_t^B]
}
\]

Eğer \(F_{\theta,t}\) ilgili bölgede affine ise \(C_t=0\). Bu durumda excursion'ın matematiksel yeniliği büyük ölçüde sıradan update mixing'e indirgenir. **Dolayısıyla \(\|C_t\|\)'yi raporlamak zorunludur; ancak büyük \(\|C_t\|\) tek başına fayda kanıtı değildir.** Gerekirse \(C_t\)'nin decoder doğruluğuna nedensel etkisini incele.

### 3.5 Önerilen en basit ilk algoritma

```python
# PSEUDOCODE — Ouro'nun gerçek API'si değildir.
# F_t: Aynı current_ut, mask, position ve kontrol edilmiş cache
# koşullarında tek recurrent-core geçişi.

A, B = fork_after_first_recurrent_step(input, seed)

for t in allowed_recurrent_steps:
    if t != visit_step:
        A = F_t(A)
        B = F_t(B)
        continue

    F_A = F_t(A)
    F_B = F_t(B)          # baseline ve curvature kontrolünde de gerekli

    D = B - A
    visit = A + bounded_step(D, distance_budget)
    F_visit = F_t(visit)

    update_A = F_A - A
    update_visit = F_visit - visit
    A = A + (1.0 - beta) * update_A + beta * update_visit
    B = F_B

answer_A = decode_from_final_state(A)
answer_B = decode_from_final_state(B)
```

Bu kod gerçek implementasyon talimatıdır **ama çalışır kütüphane API'si iddiasında bulunmaz**. Yukarıda kullanılan fonksiyonlar ajan tarafından, gerçek Ouro kaynak kodu incelendikten sonra uygulanacaktır.

### 3.6 İki kritik tasarım seçimi

**State müdahalesi nereye uygulanacak?** İlk fazda belirli recurrent step sonundaki **tam hidden-state tensörü** ile başla. İleride token-position selection ve feature/subspace selection çalış. Tek token'a etki etmek ile bütün prompt tensörünü değiştirmek farklı deneylerdir; her koşul kaydedilmeli.

**Ne zaman uygulayacağız?** Ouro'nun 4 step varsayımı nedeniyle ilk müdahale örneği ortak step 1 sonrasında branch oluşturup step 2 veya 3'te ziyaret olabilir. Son step'te excursion yapıp “devam eden trajectory'yi kurtardık” deme; en az bir anlamlı devam adımı olmalı.

---

## 4. En büyük bilimsel riskler ve korunma yöntemleri

1. **Gürültü ≠ yararlı alternatif düşünce.** Gaussian noise yalnızca farklı bir hidden state üretir; farklı *anlamsal* problem çözme stratejisi üretildiği kanıtlanmaz. Diversity ve bağımsız çözüm farkları ölçülmeli.
2. **Hidden-state ortalamaları dağılım dışına çıkabilir.** Müdahale normlarını sınırlamak, norm/RMS ve token-başına istatistikleri izlemek, NaN/Inf veya logits çökmesi görürsek deneyi geçersiz saymak gerekir.
3. **Aynı model, aynı prompt, aynı deterministik işlem → aynı trajectory.** Gerçek dal çeşitliliği kontrollü perturbation veya başka mekanizma gerektirir.
4. **Araç parametresi olarak verilen “loop sayısı” her zaman ara state'e devam imkânı sağlamaz.** Modelin kendi kaynak kodu hook/adapter düzeyinde incelenecek.
5. **Modelin recurrent uygulaması step'e bağlı olabilir.** \(F_t\)'yi evrensel, konumdan bağımsız tek bir sabit fonksiyon gibi uygulamak hatalı olabilir.
6. **Recurrent state ile token-level KV cache aynı şey değildir.** Branch'ler kendi cache ve prefix tutarlılığını sağlamalı; ilk safhada `use_cache=False` ve kısa input tercih et.
7. **Tek-tokendeki daha iyi olasılık, nihai reasoning başarısı değildir.** Pilot metrik ile gerçek sequence-level sonucu ayrı raporla.
8. **Compute unfairness:** Ziyaret noktasında ek \(F_t\) değerlendirmesi maliyet yaratır; daha fazla hesaplama yapan yöntem doğal olarak avantaj kazanabilir.
9. **Oracle donor seçimi skoru şişirir.** Gold cevap yalnızca mekanizma kapasitesi testi ve analiz için; deploy edilebilir yöntemde yasak.
10. **Tuning/test leakage:** Müdahale katsayılarını final test üzerinde seçme. Dataset, seed ve görev şablonları train/validation/test düzeyinde ayrılmalı.
11. **Ad hoc “stuck” tanımı:** İki step'te cosine benzerliği yüksek çıkması başarısız attractor tespiti değildir.
12. **Ağırlık-uzayı merging ile karıştırma:** Bu proje model parametrelerini fine-tune edip toplamakla ilgili değildir; ana çalışma inference-time hidden-state müdahalesidir.
13. **CoT/AI alignment iddiaları:** İç latent state'e müdahale etmek, modelin gerçek niyetini okuma veya alignment doğrulama anlamına gelmez; safety uzantısı ayrı deneydir.

---

## 5. Güncel literatürle ilişki ve iddia sınırları

Bu proje 2026 itibarıyla aktif ve rekabetçi bir literatüre giriyor. Çalışmaları baştan okuyup bir `literature_matrix.md` üret:

| Referans | İlgili mevcut katkı | Bizim test edeceğimiz fark |
|---|---|---|
| [Ouro / Looped Language Models](https://ouro-llm.github.io/) ve [Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking) | Shared-weight latent recurrent inference, değişken geçiş sayısı, açık model | Recurrent step içinden kontrollü branch/excursion müdahalesi |
| [Parallel Test-Time Scaling for Latent Reasoning Models — ACL 2026](https://aclanthology.org/2026.acl-long.2069/) | Latent noise/dropout ile çoklu trajectory ve latent reward ile seçim | Sadece sona kadar bağımsız gitmek yerine **devam ederken** cross-branch transition kullanımı |
| [GRAM — Generative Recursive Reasoning, 2026](https://ahn-ml.github.io/gram-website/) | Çoklu latent reasoning rotaları | Excursion-and-return ile özgül failed-trajectory recovery hipotezi |
| [RecursiveMAS — 2026](https://recursivemas.github.io/) | Learned latent link ile farklı agent'lar arası recurrent işbirliği | Aynı ağırlığı paylaşan tek LoopLM'nin dallarında, önce **training-free**, yerel transition örnekleme |
| [Interlat — ACL 2026](https://aclanthology.org/2026.acl-long.1248/) | Agent'lar arası latent haberleşme | Haberleşmenin kendisinden çok başka konumda **model transition'ının hesaplanması** |
| [STARS — ICML 2026](https://proceedings.mlr.press/v306/yang26t.html) | Daha kararlı recurrent dinamikler | Tek trajectory stabilizasyonu değil, yanlış/başarısız rotayı dış dal yardımıyla düzeltme |
| [What Makes Recurrence Effective in Looped Language Models? — 2026](https://arxiv.org/abs/2609.36636) | Recurrent yerleşim, horizon ve history-state conditioning etkileri | Sabit checkpoint'te online, cross-trajectory müdahale |
| [GTS — Gaussian Thought Sampler, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.341/) | Öğrenilmiş latent trajectory sampling | Sampler iyi olsa bile excursion'ın seçme/örnekleme ötesindeki ek faydası |

**Özgünlük iddiasını böyle sınırla:** “Birden fazla latent düşünce” ve “hidden-state paylaşımı” yapılmış. “Başka bir dalın bölgesinde transition hesaplayıp yerel update'ini asıl dala geri taşımanın, failed trajectory recovery üzerindeki etkisi” bir **araştırma hipotezi**. Literatürdeki tam öncüller yeniden taranmadan ve deneysel ek fayda gösterilmeden “ilk” veya “kanıtlanmış” olarak sunulmaz.

Ajan, literatür matrisiyle çelişen yeni bir doğrudan öncül keşfederse sonuçları saklamayacak; projeyi **reproduction + rigorous ablation + yeni failure-recovery değerlendirmesi** olarak yeniden konumlandıracak.

---

## 6. Model, cihaz, bağımlılıklar

### 6.1 Birincil model

**Birincil aday:** [`ByteDance/Ouro-1.4B-Thinking`](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking). Thinking varyantı reasoning görevleri için uygundur. Karşılaştırma amacıyla [`ByteDance/Ouro-1.4B`](https://huggingface.co/ByteDance/Ouro-1.4B) ayrıca denenebilir.

Yayımlanmış model kartlarına göre yaklaşık 1.4B parametre, 24 paylaşılan layer, varsayılan 4 recurrent step ve 2048 hidden width mevcuttur. Resmî `modeling_ouro.py` içindeki recurrent döngü, `hidden_states_list` ve `current_ut` akışı başlangıç noktasıdır. **Pinlenen checkpoint revision'ını ve dosyanın SHA/commit kimliğini raporla.** Model kartları belirli `transformers` sürüm koşulları içeriyor; kurulumda güncel kod ile uyumluluğu test et ve gerçekten çalışan kombinasyonu kilitle.

Bir model kartı `trust_remote_code=True` gerektirebilir. Uzaktan gelen kodu güvenilen bağımsız çalışma ortamında incelemeden çalıştırma. Üçüncü taraf kodunu kendi kopyasında pinle, modifikasyonları patch olarak tut.

### 6.2 Runtime

- Python 3.11 veya modelin doğrulanmış sürümüne uygun Python.
- PyTorch; Hugging Face `transformers`; `safetensors` ve gerekli tokenizer bağımlılıkları.
- NumPy, pandas veya polars, SciPy, scikit-learn, pytest, pydantic, psutil.
- Reprodüksiyon için `pyproject.toml`, `uv.lock`/eşdeğer lockfile; model revision sabitleme.
- Görselleştirme için ilk etapta Plotly/Streamlit **veya** React + TypeScript. İkisini birden ilk faza koyma.

### 6.3 Donanım

FP16/BF16 1.4B ağırlıklar tek başına kabaca 2.8 GB tutar; buna activation, attention, ek hidden branch ve model runtime maliyeti eklenir. Bu rakam toplam VRAM tahmini değildir. İlk denemede **12–24 GB GPU VRAM bulunabilirse tercih et**; kısa prompt ve küçük batch'lerle daha az VRAM mümkün olabilir. Ortama göre gerçek peak GPU belleği ölç; başarıyı modelin boyutuna dair varsayıma bağlama.

Tek makinede CPU benzeri **time slicing** mümkündür: ağırlıklar bir kez yüklenir, her dalın state'i ayrıdır. Bu işlem **hesaplamayı ücretsiz paralelleştirmez**. GPU üzerinde batch execution karşılaştırması ayrıca yapılır. İlk öncelik hız değil doğruluk ve tekrar üretilebilirliktir.

### 6.4 Fallback stratejisi

Ouro gerçek recurrent step müdahalesiyle çalıştırılamazsa, üst üste API `generate()` çağrılarıyla bunu taklit etme. Önce doğrulanmış saf PyTorch küçük recurrent toy model üzerinde algoritmayı çalıştır; sonra Ouro adapter'ını düzelt. Başka modele geçilirse sebebi ve teknik uyumsuzluğu `decision_log.md`'ye yaz.

---

## 7. Kod mimarisi: API tasarımı

### 7.1 Temel arayüzler

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class TransitionContext:
    step: int
    token_ids_hash: str
    mask_hash: str
    positions_hash: str
    cache_mode: str

@dataclass
class BranchState:
    branch_id: str
    recurrent_step: int
    hidden: "Tensor"                  # [batch, tokens, hidden]
    context: TransitionContext
    # varsa cache gibi yan durumlar adapter düzeyinde ayrı ve güvenli tutulur

@dataclass(frozen=True)
class ExcursionConfig:
    alpha: float
    beta: float
    max_relative_norm: float
    visit_step: int
    token_scope: str                  # full | last_token | selected
    donor_strategy: str               # fixed | random | diversity | adaptive

class RecurrentAdapter(Protocol):
    def initialize(self, input_ids: "Tensor") -> BranchState: ...
    def transition(self, state: BranchState) -> BranchState: ...
    def readout_logits(self, state: BranchState) -> "Tensor": ...
    def validate_compatibility(self, a: BranchState, b: BranchState) -> None: ...
```

**Doğrudan Ouro sınıflarını TEBDİL algoritmasının içine sızdırma.** Recurrent adapter, modelin bütün `current_ut`, mask, rotary embedding ve cache ayrıntılarını sahiplenmeli. Böylece mekanizma toy modelde ve mümkünse ikinci gerçek LoopLM'de değişmeden çalışır.

### 7.2 Zorunlu modüller

- `ouro_adapter.py`: checkpoint yükleme, stepwise transition, ara state ve logit readout.
- `branching.py`: aynı başlangıç state'inden seed'li, kontrollü çoklu dallar üretme.
- `perturbations.py`: Gaussian, norm-clipped Gaussian, structured perturbation; seed, norm, scope raporları.
- `excursions.py`: visit state, update fields, return operator, curvature component.
- `scheduler.py`: tek ağırlık seti üzerinde branch time-slicing ve isteğe bağlı batching.
- `controls.py`: baseline ve negatif kontrollerin aynı API ile çalıştırılması.
- `tasks/`: görev üretme, veri ayrımı, prompt şablonları ve doğrulayıcılar.
- `metrics.py`: task accuracy, recovery, regret, diversity, numeric stability, compute.
- `runner.py`: tüm konfigürasyon kombinasyonlarını deterministik yönetme.
- `traces.py`: JSONL/Parquet olay kaydı, tensor metadata, replay desteği.
- `analysis/`: analiz notebook yerine tekrarlanabilir `.py` script'leri.
- `ui/`: deneylerden elde edilen sabit veya canlı event verisiyle çalışacak görselleştirme.

### 7.3 Önerilen repo ağacı

```text
tebdil/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── uv.lock                  # doğrulanan ortamda üret
├── configs/
│   ├── phase1_smoke.yaml
│   ├── phase2_recovery.yaml
│   ├── phase3_ablations.yaml
│   └── phase4_adaptive.yaml
├── src/tebdil/
│   ├── __init__.py
│   ├── cli.py
│   ├── models/
│   │   ├── interface.py
│   │   ├── toy_adapter.py
│   │   └── ouro_adapter.py
│   ├── core/
│   │   ├── state.py
│   │   ├── branching.py
│   │   ├── excursions.py
│   │   ├── scheduler.py
│   │   ├── controls.py
│   │   └── policies.py
│   ├── tasks/
│   │   ├── arithmetic.py
│   │   ├── logic.py
│   │   ├── graph.py
│   │   └── validators.py
│   ├── evaluation/
│   │   ├── runner.py
│   │   ├── metrics.py
│   │   ├── bootstrap.py
│   │   └── trace.py
│   └── analysis/
│       ├── accuracy_depth.py
│       ├── rescue.py
│       ├── curvature.py
│       └── compute_frontier.py
├── tests/
│   ├── test_toy_transition.py
│   ├── test_adapter_equivalence.py
│   ├── test_branch_isolation.py
│   ├── test_excursion_invariants.py
│   ├── test_controls.py
│   ├── test_task_verifiers.py
│   └── test_no_label_leakage.py
├── scripts/
├── reports/
│   ├── literature_matrix.md
│   ├── decision_log.md
│   ├── phase1_report.md
│   └── experiment_registry.md
├── results/                   # büyük dosyalar Git dışı
├── ui/
└── docs/
    ├── architecture.md
    ├── methods.md
    ├── reproduction.md
    └── limitations.md
```

Ajan, mevcut repo yapısı varsa buna uyacak; yukarıdaki ağacı körlemesine yeniden oluşturmayacak. Model ağırlıklarını ve büyük aktivasyon tensorlerini Git'e commit etmeyecek.

---

## 8. Reproducibility ve deney protokolü

### 8.1 Her çalıştırmanın manifest'i

Her `run_id` için minimum:

```yaml
run_id: string
created_at_utc: iso8601
git_commit: sha
model_id: ByteDance/Ouro-1.4B-Thinking
model_revision: pinned_commit_sha
software_lock_hash: sha256
hardware: gpu_name_and_memory
seed: integer
task_family: logic
split: train_or_validation_or_test
problem_id: string
prompt_hash: sha256
decode_policy: greedy
recurrent_steps: 4
branch_count: 2
intervention: targeted_excursion
intervention_step: 2
alpha: 0.5
beta: 0.5
norm_clip: 0.1
cache_policy: disabled
precision: bf16
result: populated_by_runner
```

### 8.2 Ayrımlar

- `train/dev`: öğrenilen controller için; ilk fazlarda yalnızca yöntem kurma.
- `validation`: alpha/beta, perturbation büyüklüğü, trigger ve norm seçimi.
- `test`: bütün kararlar sabitlendikten sonra **tek ana raporlama geçişi**.

Sadece random instance ayırma yeterli olmayabilir. Mümkün olduğunca görev üretim **şablonlarını** ve problem büyüklüklerini de ayrıştır. Aynı şablonun farklı sayıları test setinde bulunuyorsa açıkça belirt.

### 8.3 Determinizm ve tutarlılık

- Ağırlıklar ilk fazlarda `eval()` ve frozen.
- Randomness seed kontrollü.
- Decoding önce greedy/temperature 0; sampling sonraki ablation.
- `use_cache=False` ile eşdeğerlik doğrulanmadan KV-cache optimizasyonu açma.
- Aynı prompt, tokenizer, `bos/eos`, position IDs, dtype ve decode ayarları.
- GPU kaynaklı düşük seviye nondeterminizm varsa tolerans tanımla.
- Logits farkı için `max_abs_diff`, `relative_diff`, top-k ve softmax KL/JSD raporla.

### 8.4 Testlerin önceliği

**Doğru müdahale yapmadan önce modelin standart davranışını birebir tekrar üretmek zorunludur.** En sık görülen hata, recurrent block'u ayırırken normalization, position embedding, cache veya step counter'ı ihmal edip farklı modeli çalıştırmaktır.

---

## 9. Değerlendirme görevleri ve gold doğrulama

### 9.1 Üç görev ailesi

**A. Kısa aritmetik ve sembolik işlemler:** Tek ve birkaç adımlı hesaplamalar, verilen ifade için sayı/etiket cevabı. Hızlı smoke test ve logits takibi için uygun.

**B. Mantık / küçük constraint satisfaction:** Boolean değişkenler, küçük CNF veya izinli kombinasyonlar. SAT solver veya kapsamlı brute force ile gold bul. Cevabın yalnızca formatını değil, verilen atamanın kuralları gerçekten sağlayıp sağlamadığını kontrol et.

**C. Küçük graf/path problemleri:** Kısa weighted graph, kısıtlı yol, basit reachable/unreachable veya minimum maliyet. Gold için Dijkstra/BFS/uygun solver.

Birincil bilimsel değerlendirme, çok uzun CoT üretimini gerektirmeyen **kısa, makinece doğrulanabilir cevaplar** ile başlamalı. Modelden çok uzun düşünce açıklaması istemek latent recurrence etkisini görünür CoT ve token-generation bütçesiyle karıştırabilir.

### 9.2 Pilot ve büyütme

- **Smoke:** 5–10 kısa görev; aynı checkpoint ile standart forward eşdeğerliği.
- **Feasibility pilot:** 20–30 farklı instance; branch ayrışması, müdahale aralığı, stabilite.
- **Recovery pilot:** 50–100 kontrollü instance; makul sayıda A-failure ve farklı B rotası olup olmadığı.
- **Main benchmark:** Birden çok görev ailesi ve şablon ayrımı; sayı, çalışma gücüne ve pilotta görülen hata sıklığına göre önceden kayda geçirilmeli.

Çok az yanlış cevap veren kolay görevleri kullanıp recovery deneme; çok zor ve hiçbir branch'in doğru cevap üretemediği görevlerde de mekanizma çalışabilirlik sinyali elde etmek zor. **Problem difficulty sweep** ile uygun karışımı bul.

### 9.3 Çıktı değerlendirme seviyeleri

- **Level 0:** Sabit prefix için gold answer token log-probability; hızlı ama görev başarısı değildir.
- **Level 1:** Belirli recurrent state'ten başlayan kısa deterministic continuation ile nihai cevap doğruluğu.
- **Level 2:** Her token'da yeni recurrent müdahale uygulanabilen tam decoding; cache doğruluğu ve maliyeti daha zordur.

Level 0'ı *final accuracy* gibi sunma. Faz 1 için Level 0 yeterlidir; Faz 2 sonunda en az Level 1 olmalıdır.

---

## 10. Karşılaştırma sistemleri: Rakipler güçlü olmalı

| ID | Yöntem | Zorunlu gerekçe |
|---|---|---|
| `S0` | Standart tek trajectory, varsayılan recurrent depth | Model referansı |
| `S1` | Aynı modele izinli ek recurrent step | “Daha fazla düşünme” alternatifi |
| `P0` | Bağımsız N latent branch; seçim kuralı sabit | Paralel latent sampling alternatifi |
| `P1` | Independent branches + **oracle best-of-N** | Yalnızca üst sınır, deploy karşılaştırması değil |
| `M0` | Direct hidden-state averaging | En basit mixing |
| `M1` | Direct update-vector mixing | Mekanizmanın doğrusal bileşeni |
| `R0` | Norm-matched random perturbation | Sadece gürültünün faydası |
| `E0` | Rastgele donor ile excursion | Ziyaret etmenin kendisinin katkısı |
| `E1` | Sabit seçilmiş donor ile excursion | Ana mekanizmanın ilk versiyonu |
| `E2` | Diversity-triggered / adaptive excursion | Nihai aday yöntem |
| `C0` | Ziyaret noktasına direkt taşınıp oradan devam | “Geri dönmek” gerçekten gerekli mi? |

### 10.1 FLOP ve hesaplama bütçesi

`F_t(H)` bir **tam recurrent-block evaluation** birimidir; farklı sequence length ve batch boyutlarında bu birim tek başına eşit FLOP demek değildir. Bu nedenle:

- Önce `num_recurrent_block_calls` say.
- Sonra token sayısı ve model tabanlı FLOP tahmini yap; varsa profiler FLOP sayımıyla doğrula.
- End-to-end GPU zamanı (`cuda event`, warmup sonrası), peak VRAM, enerji mümkünse ölç.
- Extra excursion call maliyetini saklama.
- Aynı toplam bütçeyle control'lara ilave branch, ekstra recurrence veya bağımsız sample hakkı ver.
- **Pareto frontier:** Accuracy / recovery karşısında FLOP ve wall-clock grafikleri yayınla.

Örnek: A ve B için birer `F_t` ile birlikte bir de `F_t(H_visit)` çağırmak **o noktada üç ayrı transition** demektir. Sadece iki transition çalıştıran baseline'ı karşısına koyup “TEBDİL daha iyi” demek geçersizdir.

---

## 11. Metrikler ve analiz

### 11.1 Ana metrikler

**Task accuracy / exact validity:** Gold validator sonucu.

**Conditional rescue rate:**

\[
\mathrm{RescueRate}=
\frac{\#\{i:A_i\ \text{wrong},\ A_i'\ \text{correct}\}}
{\#\{i:A_i\ \text{wrong}\}}
\]

**Degradation rate:**

\[
\mathrm{DegradationRate}=
\frac{\#\{i:A_i\ \text{correct},\ A_i'\ \text{wrong}\}}
{\#\{i:A_i\ \text{correct}\}}
\]

**Paired accuracy difference:** Aynı prompt/seed üzerinde yöntemin doğruluğu eksi baseline doğruluğu. Rescue tek başına güvenilir ana skor değildir: doğru çözümleri bozmanın maliyetini görmeliyiz.

**Compute-normalized quality:** Aynı recurrent FLOP veya aynı GPU süresi altında accuracy/recovery.

### 11.2 Mekanistik metrikler

- \(\|H_A-H_B\|_F\), cosine similarity ve token başına ayrışma.
- \(\|H_{t+1}-H_t\|\) ile recurrence update norm'u.
- RMS / norm / finite-value kontrolleri; layer veya token pozisyonu bazında.
- \(\|C_t\|_F\) ve \(\|C_t\|/\|V_A\|\): nonlineer excursion bileşeni.
- Final logits KL/JSD, answer log-prob değişimi.
- Dalların top-k token veya final cevap çeşitliliği.
- Müdahale büyüklüğüne göre accuracy, degradation ve instability.
- Müdahale anına (`visit_step`) ve branch separation'a göre duyarlılık.

### 11.3 İstatistiksel analiz

- Seed'ler ve problem örnekleri için paired analysis.
- Accuracy farkı için bootstrap confidence intervals; geçerliyse McNemar testi gibi paired ikili test.
- Çoklu karşılaştırma / hiperparametre aramasını açıkça raporla; test setine seçilen “en iyi” konfigürasyonu taşı.
- Heterojen görev ailelerinin sonuçlarını yalnızca tek mikro-ortalamaya sıkıştırma.
- Başarılı vakalar kadar olumsuz ve etkisiz vakaları da göster.

**Ön kayıt:** Ana baseline, ana metrik ve tuning politikası main test çalıştırılmadan önce `experiment_registry.md`'ye yazılmalı.

---

## 12. Faz 0 — Literatür ve teknik keşif

### Amaç

Gerçek Ouro sürümünün kontrol edilebilir bir recurrent-state geçişine ayrılıp ayrılamadığını doğrulamak; deneyin yanlış API varsayımlarına dayanmamasını sağlamak.

### Görevler

1. Ouro model kartını, `configuration_ouro.py` ve `modeling_ouro.py` dosyalarını pinli revizyonla oku.
2. Forward içindeki `for current_ut in range(total_ut_steps)` bölgesini, normalizasyonun konumunu ve erken çıkış/readout mantığını haritala.
3. KV cache, position IDs ve attention mask'in branch/fork sırasında nasıl ayrışması gerektiğini incele.
4. `RecurrentAdapter` için **tam olarak hangi tensörün** recurrent state sayılacağını seç ve gerekçelendir.
5. Recurrent transition ile decode readout'un giriş/çıkış şekillerini küçük testlerle doğrula.
6. Kaynaklar ve doğrudan ilgili öncüller için `reports/literature_matrix.md` oluştur.
7. Uyumluluk ve potansiyel sayısal sınırlamaları `reports/decision_log.md`'ye yaz.

### Teslimatlar

- Çalışan environment ve dependency lock.
- En az bir kısa input için orijinal model forward vs instrumented adapter karşılaştırması.
- `docs/architecture.md` ve `phase0_compatibility_report.md`.

### Faz kapısı

**GO:** Stepwise inference modeli, müdahale olmadan standart forward'a kabul edilen tolerans içinde eşdeğer.  
**NO-GO:** Hidden state'i sürdürmek normalizasyon/attention/cache açısından yanlış sonuç üretiyor ve çözülemiyor. Bu durumda toy adapter ile matematiği kanıtla, Ouro entegrasyonunu başarılmış sayma.

---

## 13. Faz 1 — Teknik feasibility: İlk gerçek latent excursion

### Amaç

**Araştırmanın ilk gerçek başarısı:** Aynı açık LoopLM içinde iki farklı latent trajectory oluşturup birinin update yönünü diğerinin latent bölgesinde hesaplamak ve orijinal rotaya geri taşımak.

### Alt adımlar

**1A — Standard equivalence**

- `use_cache=False`, kısa prompt, `eval()`, sabit dtype/seed.
- Her recurrent step'in state'ini ve logits'ini kaydet.
- Normal forward ile adım adım yeniden kurulan forward'ın çıktısını karşılaştır.
- Early-exit sabitlemesi yap; testin recurrent step sayısı konusunda değişken olmamalı.

**1B — Branch isolation**

- Step 1 state'ini A/B olarak klonla.
- B'ye norm-sınırlı perturbation ver.
- A ve B'yi ayrık storage ile yürüt, aliasing/in-place mutation testi ekle.
- Perturbation sıfırken sonuç A==B ve baseline ile eşleşmeli.

**1C — Excursion operator**

- İlk olarak \(\alpha=0,\beta=0\) kimlik kontrolü.
- Ardından makul küçük grid: \(\alpha\in\{0.25,0.5,0.75,1.0\}\); \(\beta\in\{0,0.25,0.5,1.0\}\), norm clipping açık.
- Update mixing ve curvature computation implement et.
- Seed, perturbation tensor normu ve state istatistiklerini logla.

**1D — Output effects**

- Gold answer log-prob ve logits ayrışmasını ölç.
- NaN/Inf, eşiği aşan RMS veya çöken çıktı varsa başarısız koşul olarak kaydet.
- 20–30 kısa problemde bütün pilot koşullarını çalıştır.

### Zorunlu testler

```text
[ ] adapter disabled == original forward
[ ] alpha=0 ya da beta=0 kimlik davranışı
[ ] A=B durumunda excursion etkisiz (numerical tolerance)
[ ] B state değiştirildiğinde A etkilenmiyor (izolasyon)
[ ] same-step / same-token-position assertions çalışıyor
[ ] zero-noise seed tekrarları yeniden üretilebilir
[ ] tensor shape, dtype ve device doğrulaması var
[ ] NaN/Inf ve norm eşiği alarmı var
[ ] tüm intervention'lar olay log'una yazılıyor
[ ] batch=1 çalışırken farklı batch'e ihtiyaç yok
```

### Faz kapısı

**GO:** Müdahalesiz eşdeğerlik doğrulandı, müdahale kontrollü ve sonlu sonuç üretiyor, bazı nonzero müdahaleler logits/cevap dağılımını ölçülebilir biçimde değiştiriyor.  
**NO-GO:** Müdahale yalnızca bozuk cache/normalizasyon yüzünden değişim üretiyor veya bütün makul normlarda çıktı çöküyor. İlerlemeyi sahtelemeyip adapter / müdahale noktasını yeniden tasarla.

> Faz 1 çıktısı modelin daha akıllı olduğunun kanıtı değildir; **kontrollü ve güvenilir latent intervention aracı**dır.

---

## 14. Faz 2 — Başarısız trajectory'yi gerçekten kurtarabiliyor muyuz?

### Amaç

TEBDİL'in yanlış bir referans rotayı doğruya çevirebilme kapasitesini ölçmek ve bunu **doğru rotalara verdiği zarar** ile birlikte raporlamak.

### Protokol

1. Zorluk kontrollü, bağımsız gold-validator'lı problem bankası oluştur.
2. A ve B dallarını farklı perturbation seed'leriyle çalıştır.
3. **Offline etiketleme:** Her branch'in standart devamı doğru mu yanlış mı belirle.
4. Test kümesindeki hem yanlış A hem doğru A vakalarını kaydet; yalnızca kurtarılabilir alt kümeyi raporlamaya indirgeme.
5. Mekanizma kapasitesi için A yanlış/B doğru çiftlerinde **oracle donor** analizi yap; bunu ana sonuç olarak sunma.
6. Aynı çiftlerde independent, random perturb, direct state mixing, update mixing ve excursion çalıştır.
7. Mekanizma çalışırsa **donor doğruluğu bilinmeyen** donor seçimini dene.
8. Doğru A'nın bozulduğu koşulları ayrı birincil metrik olarak ölç.

### Donor seçenekleri

- Sabit B (ilk kontrollü deney).
- Rastgele B (negatif/kolay baseline).
- Hidden-state çeşitliliği yüksek B (label-free).
- Ara logits belirsizliği farklı B (label-free, model calibration sınırlı).
- Oracle-correct B (yalnızca upper bound ve nedensel kapasite).

### Beklenen araştırma çıktıları

- Rescue/degradation/overall accuracy tablosu.
- Başarı ve başarısızlık örnekleri için state ve logits replay.
- Same-compute independent sampling ile karşılaştırma.
- Donor'un doğru olup olmadığı bilinmeden fayda görülüp görülmediği.

### Faz kapısı

**GO:** Yeterli sayıda mekanizma vakasıyla birlikte kontrollü iyileşme veya tekrarlanabilir negatif sonuç. Sonraki fazı özellikle hangi mekanizmanın etkili olduğuna göre seç.

**Pivot:** Hiçbir makul girişimde rescue gerçekleşmiyorsa önce hidden-state müdahalesinin dağılım dışına çıkıp çıkmadığını ve verilen problem bankasında modelin neden başarısız olduğunu analiz et; katman/feature seçimi veya daha iyi branching dene. “Adaptive controller başarısızlığı çözecek” varsayımıyla ilerleme.

---

## 15. Faz 3 — Mekanizma analizi ve güçlü ablation

### Amaç

“TEBDİL başka bir noktada hesaplama yaparak gerçek avantaj elde ediyor mu, yoksa daha fazla hesaplama ve gürültü mü işe yarıyor?” sorusunu yanıtla.

### Zorunlu ablation matrisi

| Boyut | Test değerleri / koşulları |
|---|---|
| Ziyaret mesafesi | \(\alpha\) grid, ayrıca norm-trust-region |
| Geri dönüş | \(\beta\) grid, 0 ve 1 kontrolleri |
| Müdahale zamanı | Uygun recurrent step'ler; en az bir devam adımı |
| Token scope | Whole sequence, last token, seçilmiş token grupları |
| Donor | Rastgele, benzer, farklı, oracle (analysis only) |
| Sampling | Gaussian noise, norm-clipped, dropout uygunsa |
| Return method | Direct state, state mixing, update mixing, excursion |
| Compute | Aynı recurrent calls/FLOP/latency bütçeli rakipler |
| Model | Önce Thinking, mümkünse base kontrol |

### Curvature analizi

Ziyaret farkı \(C_t\)'yi her vakada hesapla. Şu bağlantıları incele:

- Başarı artışı \(\|C_t\|\) ile ilişkili mi?
- Fayda **\(C_t\) teriminin yönünden** geliyor mu, yoksa yalnızca büyük normdan mı?
- \(C_t\)'yi kaldırınca başarının kaybolduğu bir ablation yapılabiliyor mu?
- Aynı normda rastgele başka yöne perturbation, yararı taklit edebiliyor mu?

**Önemli:** Tüm nonlineerliklerin “faydalı bilgi” taşıdığını iddia etme. Nedensel test yap.

### Faz kapısı

**GO:** Yöntemin neden/bazen ne zaman işe yaradığını açıklayan güçlü ablation ve eşit-compute tabloları. Özgün mekanizma için yeterli sinyal yoksa sonuç “cooperative excursion, tested settings'de random/mixing'i aşmadı” olabilir; bunu dürüstçe raporla.

---

## 16. Faz 4 — Adaptive TEBDİL: Ne zaman, kimi, ne kadar ziyaret etmeli?

### Amaç

Gold sonucu bilmeden, görevin ortasında excursion'ın yararlı olabileceği **zamanları ve donor'ları** seçen bir policy üretmek.

### 16.1 Basit, eğitimsiz başlangıç policy'si

Controller'ın erişebileceği sinyaller:

- Son iki recurrence arasındaki normalleştirilmiş update norm'u.
- Branch A/B arasındaki açı veya uzaklık.
- Ara logits entropy/top-k margin (varsa).
- Önceki aynı instance içindeki faydalı veya zararlı ziyaret sinyalleri; **ground truth içeremez**.
- Kalan recurrent compute bütçesi.

İlk controller bir threshold sistemi olsun:

```python
# label-free ve causal özelliklerle karar: örnek policy
if budget_remaining < extra_visit_cost:
    return "no_visit"
if divergence_is_too_small or instability_flag:
    return "no_visit"
if stagnation_proxy and suitable_donor_exists:
    return "visit"
return "continue_normal"
```

**Stagnation proxy ≠ gerçekten yanlışta takıldık**. Bunu “erken risk / düşük ilerleme göstergesi” olarak raporla.

### 16.2 Donor seçimi

- **Random donor**: temel.
- **Most-diverse donor**: uzaklığın fazla olması yarar getiriyor mu?
- **Mid-diverse donor**: çok uzak state'lerin OOD riski var mı?
- **Nearest useful donor proxy**: logit margin veya entropy gibi etiket gerektirmeyen vekil sinyal.

### 16.3 Jev-like düşük maliyetli controller — koşullu uzantı

Bu adım ancak heuristik controller benchmark'ı kurulursa yapılır. Bir hızlı karar modeli:

`(branch summaries, state distances, entropy, remaining budget) -> {no_visit, visit_B, visit_C, intensity_bin}`

çıktısı verebilir. Ağır backbone inference'ından daha ucuz olmalı; aksi takdirde gerekçesi kaybolur.

**Öğrenilmiş policy training kuralları:** Train split dışında oracle truth yok; validation seçimi yalnızca tuning için; final test dokunulmaz. Policy'nin kendi inference maliyeti bütçeye eklenir. Başlangıçta 2 branch ve bir ziyaret sınırı korunur.

### Faz kapısı

**GO:** Label-free adaptive policy, makul güven aralığında compute-matched random/fixed excursion'a kıyasla daha iyi Pareto noktası sunuyor veya seçici olmayan ziyaretlerden daha az degradation yaratıyor.

**NO-GO:** Heuristik controller yarar sağlamıyorsa küçük policy modeli yalnızca gösteriş için ekleme; mekanizma çalışması ve negatif sonuçla güçlü bir proje tamamlanabilir.

---

## 17. Faz 5 — Genellenebilirlik, güvenilirlik ve benchmark

### Amaç

Tek checkpoint ve tek görev ailesindeki şanslı sonucu genelleyebilmek.

### Testler

- Yeni problem şablonları / görev ailesi.
- Yeni seed'ler ve örnekleme şiddetleri.
- Ouro Base ve Thinking karşılaştırması (uyumlu oldukları doğrulanınca).
- Donanım / dtype değişimlerinde duyarlılık (mümkünse).
- Aynı compute maliyetli bağımsız branch sayısı artışı.
- Bootstrap confidence intervals ve Pareto eğrileri.

### Genişleme kriteri

Yalnızca Faz 3–4'te anlamlı sinyal varsa başka model ailesi (ör. farklı re-read-loop mimarisi) ekle. Farklı mimariler hidden-state aktarımının tanımını değiştireceğinden adapter uyumluluğunu sıfırdan doğrula; checkpoint farklılığını mekanizma katkısıyla karıştırma.

### Opsiyonel safety uzantısı

Farklı trajectory'lerden birinin belirlenmiş bir action/answer constraint'ini ihlal edip etmediğini **formal validator** ile ölçebiliriz. Ama bunu genel alignment tespiti olarak sunma. Ana tez boyunca ilk odak **reasoning recovery** kalmalı.

---

## 18. Faz 6 — Observatory, demo, açık kaynak yayın

### Nihai görsel deneyim

Kullanıcı bir görev seçer, iki latent branch'in recurrent hesaplamasını görür, excursion'ı açık/kapalı çalıştırır ve:

1. Her step'te state ayrışmasını.
2. Ziyaret yapılan yeri, katsayıları ve norm sınırını.
3. Doğrudan mixing'e karşı farkı (\(C_t\)).
4. Modelin gold answer olasılığını veya final doğruluğunu.
5. Kurtarma/degradation durumunu.
6. Extra recurrent calls, FLOP, süre ve VRAM'i.
7. Bir örneğin seed ve konfigürasyonla yeniden oynatılmasını görür.

**Önerilen arayüz:** Solda problem ve branch'ler; ortada latent trajectory haritası (PCA/UMAP yalnızca *görselleştirme*, gerçek metrikleri taşıyan çizgiler ayrı); sağda metrik, compare ve intervention paneli. Veriler gerçek `results/` trace dosyalarından gelmeli; varsayılan sahte skor gösterme.

**Önemli:** PCA/UMAP'te iki noktanın yan yana görünmesi, tam-dimensiyonlu latent uzayda yakın olduklarının ispatı değildir. Görsellerin altına dönüşüm yöntemi ve explained variance gibi sınırlılıkları ekle.

### Açık kaynak teslim paketleri

- Çalıştırılabilir baseline ve deney runner.
- Varsayılan config'ler ve seed'li görev veri üreticisi.
- Model ağırlıkları dışarıdan indirilen, açık lisansla uyumlu araçlar.
- Belirli üretim sonuçlarını yeniden oluşturmak için `reproduce.sh`.
- `README.md` yöntem özeti, demo GIF/MP4, performans grafikleri.
- `METHODS.md` ve `LIMITATIONS.md`.
- `reports/final_report.md` ve `reports/final_results.json`.
- Sunumda gösterilecek önceden kaydedilmiş replay + mümkünse canlı çalışma.

---

## 19. Önerilen yürütme sırası ve bağımlılık çizelgesi

| Sıra | Milestone | Önkoşul | Done olduğunda kanıt |
|---|---|---|---|
| M0 | Kaynak/model doğrulama | Yok | Versiyon pinleri + mimari raporu |
| M1 | Recurrent adapter eşdeğerliği | M0 | Test sonuçları, max logit diff |
| M2 | İki bağımsız trajectory | M1 | Ayrışma ve seed testleri |
| M3 | Excursion operator | M2 | Intervention/control testleri |
| M4 | İlk cevap etkileri | M3 | Pilot veri ve stabilite raporu |
| M5 | Rescue ve degradation | M4 | Gold validator'lı sonuçlar |
| M6 | Eşit-compute ablation | M5 | Pareto ve paired analiz |
| M7 | Label-free adaptive controller | M6; yarar sinyali | Validation/test ayrımlı sonuç |
| M8 | Genelleme ve tekrarlama | M6/M7 | Benchmark raporu |
| M9 | Observatory + yayın | En az M6 | Demo, reproducibility, final report |

**Kritik yol:** M0 → M1 → M2 → M3 → M4 → M5 → M6. Controller ve UI bunu engellememeli.

---

## 20. Karar ağacı: Hangi sonuçta ne yapacağız?

### A. Model adapter'ı eşdeğer değil

- `current_ut` koşullandırması, norm uygulaması, attention mask ve cache'i adım adım denetle.
- Orijinal Ouro koduna minimal instrumentation patch uygula.
- Kısa prompt, batch 1, deterministic decode ile sınırla.
- Eşdeğerlik sağlanmadan benchmark raporlama.

### B. İki branch oluşuyor ama anlamlı karar farkı yok

- Perturbation büyüklüğü ile çıktı etkisi arasındaki eğriyi çıkar.
- Belirli token scope veya recurrent step seçimini ablate et.
- Norm arttığında sadece bozulan durum varsa in-distribution branching için dropout/structured perturbation gibi yöntemleri incele.
- “İki trajectory üretildi” yerine gerçekten farklı çözüm davranışı olup olmadığını raporla.

### C. Rescue var ama random noise da aynı faydayı sağlıyor

- Excursion'ın katkısına ilişkin iddiayı daralt.
- Random ve state/update mixing baseline'larını daha güçlü yap.
- Eğrilik/curvature analizini kontrol et.
- Yeni strateji olarak donor selection veya intervention timing'i deneyebilirsin; fakat seçimi validation'da yap.

### D. Oracle donor işe yarıyor ama non-oracle başarısız

- Mekanizma bir kapasite gösteriyor, deploy edilebilir karar kuralı henüz yok.
- Label-free proxy ve maliyet duyarlı seçim çalış.
- Başarı yoksa “oracle upper bound” dürüst ve ayrı raporlanmalı.

### E. Kurtarma artıyor ama doğru cevaplar da bozuluyor

- Net paired accuracy ve degradation raporla.
- No-op seçeneği olan seçici controller dene.
- Güvenilirlik / maliyet sınırı olmadan başarı iddiası verme.

### F. TEBDİL basit update mixing'i yenemiyor

- Matematiksel katkının sınırını kabul et.
- Daha güçlü hedefli ziyaret, history-guided branch seçimi veya learned gate ancak mekanizma analizinden sonra önerilebilir.
- Negatif bulguyu yayınlanabilir / yeniden üretilebilir biçimde tamamla.

---

## 21. Kalite ve mühendislik ilkeleri

- **Read before edit:** Kaynak kodu anlamadan ağır patch yazma.
- **Minimal patches:** Orijinal modelin davranışı değişmeden test edilebilmeli; instrumentation hook ve adapter tercih et.
- **Test-first:** Tensor şekilleri, state aliasing, identity limits, compute count ve cache tutarlılığı için testleri önce oluştur.
- **Fixed interface:** Algoritma model-spesifik `forward()` ayrıntılarına bağlı olmayacak.
- **Finite-checks:** Her önemli aşamada finite-state ve norm kontrolü.
- **No data leakage:** Final test verisini hiperparametre seçiminde kullanma.
- **No invented results:** Deney çalışmadıysa raporda yalnızca plan ve teknik engel yaz.
- **Save evidence:** Run manifest, config, logs, grafiklerin veri kaynakları ve tam üreten script.
- **Respect licenses:** Model ve veri lisanslarını not et.
- **No excessive output files:** Büyük aktivasyonları Git'te tutma; isteğe bağlı örneklem kaydet.
- **No unnecessary training:** Başlangıç hipotezi inference-only. Eğitim yalnızca açık ablation ile eklenebilir.
- **Safe benchmark:** Sentetik/doğrulanabilir görevler; dış sistemlerde yetkisiz eylem yok.

---

## 22. Kayıt düzeni ve komut satırı sözleşmesi

Aşağıdaki CLI komutları **mevcut değil, kodlama ajanının geliştirmesi gereken hedef arayüzdür**:

```bash
# Donanımı ve checkpoint uyumluluğunu incele
python -m tebdil.cli inspect-model --model ByteDance/Ouro-1.4B-Thinking

# Model eşdeğerliği
python -m tebdil.cli validate-adapter --config configs/phase1_smoke.yaml

# Faz 1 smoke/feasibility
python -m tebdil.cli run --config configs/phase1_smoke.yaml

# Faz 2 rescue matrisi
python -m tebdil.cli run --config configs/phase2_recovery.yaml

# Faz 3 ablations
python -m tebdil.cli run --config configs/phase3_ablations.yaml

# Sonuçlar ve paired analiz
python -m tebdil.cli analyze --runs results/ --output reports/

# Tüm unit / integration testleri
pytest -q
```

CLI, her çalıştırmanın `run_id`'sini, run manifest'i ve output path'ini yazmalı. Başarısız koşullar dâhil bütün deneyler kayda girmeli. Çoklu seed çalıştırmalar kaldıkları yerden sürdürülebilmeli; eksik deneyler sessizce geçmiş gibi işaretlenmemeli.

### Örnek faz konfigürasyonu

```yaml
project: TEBDIL
phase: 1
model:
  repo: ByteDance/Ouro-1.4B-Thinking
  revision: PIN_BEFORE_RUN
  precision: bf16
  recurrent_steps: 4
  use_cache: false
  trust_remote_code: after_review
branches:
  count: 2
  fork_after_step: 1
  perturbation: normalized_gaussian
  perturbation_relative_norm: 0.01
intervention:
  method: excursion_return
  visit_step: 2
  alpha: 0.50
  beta: 0.50
  token_scope: full
  max_relative_norm: 0.10
  fallback_if_unstable: skip_intervention
execution:
  seed: 42
  decoding: greedy
  max_input_tokens: 96
  max_output_tokens: 24
  save_full_activations: false
evaluation:
  task_family: arithmetic
  split: pilot
  num_problems: 20
  record_compute: true
  record_logits: true
```

Buradaki `PIN_BEFORE_RUN` placeholder'ı gerçek commit/hash ile değiştirilmeli. `after_review` de bir boolean değil **kod denetim iş akışı niyeti**; gerçek loader'ın kabul ettiği parametreye dönüştürülmeden kullanılmamalı.

---

## 23. Rapor şablonları

### Faz raporunda zorunlu alanlar

```markdown
# Phase N Report

## Objective and hypothesis
## Source revisions, dependencies, hardware
## What was actually implemented
## Tests and equivalence checks
## Dataset and split policy
## Experiments and compute budgets
## Results with uncertainty
## Failure cases and negative findings
## Artifacts and reproduction commands
## Known limitations
## Go/no-go decision with evidence
## Next step justified by results
```

### Nihai bitirme raporu

1. Problem: sequential latent recurrence'ın olası verimsiz rotaları.
2. Literatür ve özgünlük iddiasının sınırlı tanımı.
3. Matematiksel operator, invariants ve maliyet.
4. Ouro entegrasyonu ve reprodüksiyon.
5. Görevler, oracle/non-oracle ayrımı.
6. Baseline'lar ve compute matching.
7. Ana sonuçlar, güven aralıkları, ablation.
8. Mekanistik yorum: curvature, divergence, intervention timing.
9. Genelleme, sınırlamalar, negatif vakalar.
10. Demo, kaynak kodu, future work.

---

## 24. Bilimsel değerlendirme için “kanıt seviyesi” sözlüğü

İfadeleri buna göre kullan:

| Gözlem | Kullanılabilecek ifade | Kullanılamayacak iddia |
|---|---|---|
| Hidden-state ve logits değişti | “Müdahale hesaplamayı etkiledi.” | “Model daha iyi düşündü.” |
| Bazı yanlış cevaplar düzeldi | “Seçilmiş örneklerde recovery gözlendi.” | “Genel olarak başarılı.” |
| Oracle-correct donor ile başarı | “Upper-bound mekanizma kapasitesi.” | “Deploy edilebilir controller.” |
| Same-compute testte iyileşme | “Kontrol edilen bütçede kazanım.” | “Her modelde kesin üstünlük.” |
| \(\|C\|>0\) | “Nonlineer update farkı var.” | “Faydanın nedeni kanıtlandı.” |
| Müdahaleyle başarılı fakat bazen bozuyor | “Trade-off var.” | “Her zaman güvenilir.” |
| PCA'da başka bölge | “2D projeksiyonda ayrışma.” | “Latent attractor'dan kesin çıkış.” |

---

## 25. Ana go/no-go kriterleri (proje yönetimi)

**G0 — Teknik doğruluk:** Stepwise adapter standart Ouro hesaplamasını müdahalesiz eşdeğerlikle tekrarlıyor mu? Hayırsa bilimsel deneylere geçme.

**G1 — Sayısal müdahale:** Branching/excursion açıkken finite ve kontrol edilebilir bir işlem var mı? Hayırsa state norm, token scope veya adapter düzelt.

**G2 — Görev etkisi:** Sadece logits değil, nihai doğruluk değişebiliyor mu? Hayırsa yöntemin etki noktası ve model seçimi yeniden değerlendir.

**G3 — Karşılaştırmalı fayda:** Aynı compute bütçesinde doğrudan mixing veya random perturb'dan öne çıkıyor mu? Hayırsa mekanizma hipotezini daralt.

**G4 — Non-oracle yarar:** Gold labels olmadan seçici trigger/donor avantaj sağlıyor mu? Hayırsa “oracle upper bound” ile tam sistemi karıştırma.

**G5 — Tekrarlanabilirlik:** Sonuçlar yeni seed ve problem şablonlarında korunuyor mu? Hayırsa genelleme iddiasından kaçın.

Bu kapıları onaylayan gerçek kanıtlar `reports/` dizininde bulunacak. Yapılmayan deneyin kutusu işaretlenmeyecek.

---

## 26. Önceki alternatiflerden farkı

Bu doküman şu iki ayrı araştırmayı birbirine karıştırmaz:

- **LATENT//SWAP:** Branch'ler arasında doğrudan hidden-state veya feature paylaşma genel fikri.
- **TEBDİL / Cooperative Latent Excursion:** Başka bir latent konumda **recurrent update** hesaplayıp kendi konumuna geri taşıma ve bunun failed-trajectory recovery etkisini test etme.

İlişkili ama ayrı bir diğer fikir **weight arithmetic** (ör. \(\theta_3=\theta_2+\theta_1-\theta_0\)) idi. O, iki eğitim rejiminin parametre güncellemelerini birleştirmeyi hedefler. **Bu projenin yöntemine dahil edilmeyecek; future-work olarak korunabilir.**

---

## 27. İleri uzantılar — yalnızca kanıt oluşursa

### 27.1 Daha fazla dal ve cooperative excursions

İki daldan 3–4 dala geç. Başlangıçta tüm dalları eşit çalıştırmak yerine budget-aware scheduler uygula. Bir dal yanlış bölgedeyse diğerleri donor adayı olabilir; fakat hangi donor'un faydalı olacağı gold label görmeden seçilmeli.

### 27.2 Local geometry-guided visitation

Basit cosine uzaklığı yerine estimated local curvature, Jacobian-vector product (maliyet pahalı olabilir) veya geçmiş faydalı ziyaretlerin ampirik sinyali ile donor/mesafe seç. Yeni ağır hesaplamalar bütün baseline'lara eşit bütçe karşılaştırmasında dahil edilmeli.

### 27.3 Selective feature transport

Tam hidden-state update yerine seçilmiş feature altuzayını transfer et. İlk denemede norm-clipped projected update, sonra gerekliyse küçük learned adapter. Öğrenilmiş temsil, önceki eğitim kaynaklarına ve dağılım dışına çıkma riskine göre ayrıca kontrol edilmeli.

### 27.4 Safety-recovery ayrı değerlendirmesi

Reasoning recovery çalıştıktan sonra kısıtlı action selection üzerinde modelin yanlış/güvensiz kararının TEBDİL ile değişip değişmediğini test edebilirsin. Formal validator yalnızca dış eylem özelliğini ölçer; modelin “alignment”ının genel kanıtı değildir.

---

## 28. Projenin açık bilim taahhüdü

Bu projenin sonunda şu dört sonuçtan **hangisi doğru çıkarsa onu** yayımla:

1. TEBDİL aynı compute bütçesinde anlamlı biçimde daha iyi.
2. Bazı görev/durumlarda avantajlı, bazılarında dezavantajlı; yararın belirli koşulları var.
3. Donor seçiminde oracle erişimi gerekmedikçe avantaj görünmüyor.
4. TEBDİL basit mixing/perturbation'ı geçemiyor; teorik olarak ilginç ama pratikte ek maliyetine değmiyor.

Model iyileşmesini “ispatlamaya” çalışmak yerine **hipotezi yanlışlayabilecek bir düzenek kur**. Sonucun güvenilirliği, pozitif olup olmamasından daha önemlidir.

---

## 29. Kaynaklar ve başlangıç linkleri

Aşağıdaki bağlantılar 27 Ekim 2026 itibarıyla proje başlangıcı için doğrulanan ana referanslardır. Repository'ler yeni commit'lerle değişebileceği için **kullanılan exact revision her koşulda pinlenmeli**.

1. **Ouro project:** https://ouro-llm.github.io/
2. **Ouro-1.4B-Thinking model card:** https://huggingface.co/ByteDance/Ouro-1.4B-Thinking
3. **Ouro source / model implementation:** https://huggingface.co/ByteDance/Ouro-1.4B-Thinking/blob/main/modeling_ouro.py
4. **Ouro configuration:** https://huggingface.co/ByteDance/Ouro-1.4B-Thinking/blob/main/configuration_ouro.py
5. **Ouro base model:** https://huggingface.co/ByteDance/Ouro-1.4B
6. **Parallel Test-Time Scaling for Latent Reasoning Models (ACL 2026):** https://aclanthology.org/2026.acl-long.2069/
7. **GRAM — Generative Recursive Reasoning:** https://ahn-ml.github.io/gram-website/
8. **RecursiveMAS:** https://recursivemas.github.io/
9. **RecursiveMAS code:** https://github.com/RecursiveMAS/RecursiveMAS
10. **Interlat — Enabling Agents to Communicate Entirely in Latent Space:** https://aclanthology.org/2026.acl-long.1248/
11. **STARS — Stabilizing Recurrent Dynamics:** https://proceedings.mlr.press/v306/yang26t.html
12. **What Makes Recurrence Effective in Looped Language Models?:** https://arxiv.org/abs/2609.36636
13. **GTS — Gaussian Thought Sampler:** https://aclanthology.org/2026.emnlp-main.341/
14. **Ouro mimarisinin açık yeniden uygulama ekosistemi:** https://github.com/rkstgr/LoopLM

Kaynakları README'de yalnızca isim olarak listeleme; hangi deneysel iddiamıza karşılık geldiklerini literatür matrisinde açıkça işle.

---

# 30. CODING AGENT MASTER INSTRUCTION

> **Bu bölüm, projenin geliştirilmesini devralacak kodlama ajanına doğrudan görev tanımıdır. Belgenin tamamını okuyup burada tanımlanan sınırlara uy.**

## Rolün

Sen bilimsel doğruluğu önceleyen kıdemli bir ML araştırma mühendisi ve PyTorch/Transformer sistem geliştiricisisin. Görevin **TEBDİL: Cooperative Latent Excursions for Recurrent Reasoning** projesini uçtan uca geliştirmek: teorik mekanizmayı açık kaynak LoopLM'nin **gerçek recurrent states** üzerinde uygulamak, doğrulamak, sağlam kontrol deneyleri yürütmek, verileri analiz etmek ve tekrar üretilebilir açık kaynak çalışma sunmak.

## Başlangıçta yapman gerekenler

1. Repo'nun mevcut durumunu incele; dosyaları, çalışma ortamını, test altyapısını ve GPU imkanlarını tespit et. Mevcut işe zarar verme.
2. Bu dokümanın **Nihai vizyon**, **Matematiksel model**, **Riskler**, **Fazlar** ve **Go/no-go** bölümlerini görev kapsamı kabul et.
3. Model kartı ve kaynak dosyalarını **doğrudan oku**. Özellikle `modeling_ouro.py` içindeki recurrence, norm/readout, rotary/attention/cache koşullarını doğrula; bu belgedeki kavramsal pseudocode'u çalışır Ouro API'si zannetme.
4. Model/repo revision, bağımlılık ve lisans bilgilerini kaydet.
5. `reports/experiment_registry.md` içine ilk fazın hedefini, metriklerini ve müdahale kontrollerini yaz.
6. M0 ve M1'i gerçekten çalıştırıp test etmeden mekanizmanın başarıyla kurulduğunu iddia etme.

## Uygulama sıralaması (zorunlu)

**Adım A:** Recurrent adapter + exact-equivalence testleri. Müdahale kapalıyken standart inference davranışını koru.

**Adım B:** İki branch üret, state isolation ve stochastic perturbation'ı doğrula.

**Adım C:** TEBDİL excursion-and-return'ü tanımlanan matematikle uygula; \(\alpha=0,\beta=0,A=B\) gibi identity ve negatif testleri geçir.

**Adım D:** 20–30 kısa gold-validator'lı problemde pilot. Tam trace, finite-check, logits etkisi, maliyet ve sonuçları kaydet.

**Adım E:** Nihai cevap doğruluğuna geçen Faz 2 recovery benchmark'ı. Gold label / oracle ve non-oracle ayrımını koru. Degradation metriklerini unutma.

**Adım F:** Faz 3 güçlü baseline, cost matching, ablation ve curvature analysis.

**Adım G:** İlk sonuçları yorumla. Eğer pilot desteklemiyorsa seçici controller geliştirmeden önce mekanizma/pivot kararı ver.

**Adım H:** Yalnızca gerekçeli ise Faz 4 label-free controller, ardından farklı görevlerde Faz 5 değerlendirme.

**Adım I:** En az M6 tamamlandıktan sonra Observatory, README ve raporları tamamla.

## Her kod değişikliğinde uyulacak kurallar

- Mevcut kaynak dosyalarını okuyup küçük, anlamlı ve test edilebilir commit'ler üret.
- GPU olmayan CI'da toy adapter ve unit testler çalışabilsin; GPU entegrasyon testlerini ayrı işaretle.
- Model kodunda sessiz davranış değişikliği yapma. Patch'lerin kaynak checkpoint revizyonuna bağlanmasını sağla.
- Unverified library API veya model method isimleri uydurma.
- Seed, model revision, software hash, task version ve compute sayımını sakla.
- Gerçek deneyi çalıştırmadan grafik veya metrik uydurma.
- Benchmark'taki başarısız deneyleri silme, atlama veya seçerek raporlama.
- Üretim kalitesinde hata kontrolleri ve dtype/device/shape assert'leri ekle.
- Bilimsel katkı için her güçlü iddiaya karşılık gelen kontrol grubu oluştur.

## Görev teslim formatın

Her fazın sonunda **aynı repo içinde** şu çıktıları bırak:

1. Nelerin uygulandığı ve hangi kaynak kod revizyonu kullanıldığı.
2. Çalışan komutlar ve gerçek test çıktıları.
3. Reprodüksiyon için config, seed, run manifest.
4. Sonuçların tablo/şekil veya ham kaydı.
5. Başarısızlıklar, teknik engeller ve sınırlamalar.
6. Kanıta dayalı `GO`, `PIVOT` veya `NO-GO` kararı.
7. Sonraki faz için sadece deney sonucundan gerekçelendirilmiş adımlar.

## Kesin başarı iddiası kuralları

- “Latent trajectory rescue” diyebilmek için **A yanlışken A' doğru** ve aynı paired instance üzerinde gold validator kanıtı olmalı.
- “Daha iyi” diyebilmek için baseline ve hesaplama bütçesi karşılaştırması olmalı.
- “Özgün” diyebilmek için doğrudan önceki literatüre göre fark ve ablation gösterilmeli.
- “Attractor'dan kaçtı” diyebilmek için yalnızca PCA görüntüsü değil, uygun dinamik sistem/kararlılık analizi olmalı.
- “CoT-independent” diyebilirsin çünkü mekanizma görünür CoT gerektirmiyor; bu ifade genel AI safety garantisine dönüştürülemez.

## Nihai komut

**Öncelikle doğru deney düzeneğini inşa et.** İyi bir bilimsel çalışma, beklediğimiz sonucun doğru çıkmasına bağlı değildir. TEBDİL'i gerçekten çalıştır, aynı compute maliyetinde ciddi baseline'lara karşı test et, nerede işe yaradığını ve nerede yaramadığını ölç, metodolojiyi açıkça belgeleyip herkesin yeniden üretebileceği hâle getir.

**Projenin nihai sorusu değişmeyecek:**

> *Can a recurrent language model recover from an unsuccessful latent reasoning trajectory by temporarily computing in another trajectory's neighborhood, then transferring the resulting update back, more effectively than independent sampling, direct mixing, or extra recurrent computation under matched compute budgets?*

---

**Belge sonu — TEBDİL / LATENT//SWAP araştırma hattı.**
