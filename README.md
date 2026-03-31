# NexusCore Proto-AGI (agi-ssi-llm-mix-superai)

Bu proje, kendi kendine öğrenebilen, içsel bir değerlendirme sistemine (ConstitutionGuard) ve sürekli hafızaya (ExperienceReplayBuffer, Episodic/Semantic Memory) sahip bağımsız bir Proto-AGI çekirdeği geliştirme çalışmasıdır. 

## Özellikler

- **Continual Trainer:** Öğrendiklerini unutmadan ('catastrophic forgetting') yeni bilgiler öğrenmeye devam eden eğitim döngüsü.
- **Dual Head Architecture:** Dil modeli (Language Head) mantıklı bir şekilde dil üretebilirken aynı zamanda Action seçimi yapar (Delegate, SandBox, Search vs.).
- **Meta-Cognition Loop:** Model verdiği kararların güven (confidence) oranını değerlendirip yansıtır.
- **Sandbox Environment:** Tehlikeli operasyonlar öncesi güvenli bir karantina ortamında (simüle) işlemi deneyerek sonuçları analiz eder.

## Kurulum ve Eğitim
```bash
python3 train_real.py --epochs 30 --d_model 128 --n_layers 3 --n_heads 4 --batch_size 32
```

Model, sentetik veri setleriyle `fast_mode` ile hızlıca optimize edilmektedir. Eğitim arka planda devam eden kesintisiz bir deneyim modelidir.
