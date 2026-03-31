"""
data_pipeline.py — Dataset Loading & Batching (Proto-AGI)

Gerçek text verilerini yükler, tokenize eder ve eğitim için batch üretir.

Veri kaynakları:
  1. TextFileDataset:   .txt dosyalarından yükleme
  2. InstructionDataset: Instruction-response çiftleri (JSON)
  3. DataPipeline:       Ana pipeline — load → tokenize → batch → iterator

Harici bağımlılık: yok (saf Python + PyTorch).
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import random
from typing import List, Dict, Optional, Tuple, Iterator, Any
from pathlib import Path
from dataclasses import dataclass, field

from tokenizer import BPETokenizer, CharTokenizer, TokenizerConfig, pad_sequences


# ───────────────────────────── Config ─────────────────────────────

@dataclass
class DataConfig:
    """Data pipeline konfigürasyonu."""
    max_seq_len: int = 256          # Maksimum sequence uzunluğu
    batch_size: int = 16            # Batch boyutu
    shuffle: bool = True            # Veriyi karıştır
    num_workers: int = 0            # DataLoader worker sayısı
    train_split: float = 0.9        # Eğitim/Doğrulama oranı
    seed: int = 42                  # Reproducibility


# ───────────────────────────── Text File Dataset ──────────────────

class TextFileDataset(Dataset):
    """
    .txt dosyalarından text yükler.
    Her satır veya paragraph bir örnek.
    
    Kullanım:
        dataset = TextFileDataset("corpus.txt", tokenizer, max_len=256)
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_len: int = 256,
        chunk_mode: str = "line",  # "line" | "paragraph" | "sliding"
        stride: int = 128,        # sliding window stride
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunk_mode = chunk_mode
        self.samples: List[List[int]] = []
        
        self._load_file(file_path, stride)
    
    def _load_file(self, file_path: str, stride: int):
        """Dosyayı yükle ve tokenize et."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if self.chunk_mode == "line":
            # Her satır bir örnek
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            for line in lines:
                ids = self.tokenizer.encode(line)
                if len(ids) > 3:  # En az birkaç token
                    self.samples.append(ids[:self.max_len])
        
        elif self.chunk_mode == "paragraph":
            # Her paragraph (boş satırla ayrılmış) bir örnek
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if para:
                    ids = self.tokenizer.encode(para)
                    if len(ids) > 3:
                        self.samples.append(ids[:self.max_len])
        
        elif self.chunk_mode == "sliding":
            # Sliding window — tüm dosyayı token'lara çevir, window ile böl
            all_ids = self.tokenizer.encode(content, add_special=False)
            for i in range(0, len(all_ids) - self.max_len, stride):
                window = all_ids[i:i + self.max_len]
                self.samples.append(window)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        return: {
            'input_ids':  (max_len,)   — input token'lar
            'target_ids': (max_len,)   — shifted target (next-token prediction)
        }
        """
        ids = self.samples[idx]
        
        # Padding
        pad_id = self.tokenizer.pad_id
        padded = ids + [pad_id] * max(0, self.max_len - len(ids))
        padded = padded[:self.max_len]
        
        # Input = tokens[:-1], Target = tokens[1:] (next-token prediction)
        input_ids = padded[:-1]
        target_ids = padded[1:]
        
        # Padding kısımlarını -100 yap (loss'ta ignore edilir)
        target_ids = [
            t if t != pad_id else -100 for t in target_ids
        ]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
        }


# ───────────────────────────── Instruction Dataset ────────────────

class InstructionDataset(Dataset):
    """
    Instruction-Response çiftlerinden eğitim verisi.
    
    JSON formatı:
    [
        {"instruction": "Hava durumu nedir?", "response": "Bugün güneşli..."},
        {"instruction": "2+2 kaç?", "response": "4'tür."}
    ]
    
    Modeli hem dil üretmeyi hem doğru action seçmeyi öğretir.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples: List[Dict[str, Any]] = []
        
        self._load_file(file_path)
    
    def _load_file(self, file_path: str):
        """JSON dosyasını yükle."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            instruction = item.get('instruction', '')
            response = item.get('response', '')
            action = item.get('action', 0)  # Opsiyonel action label
            
            # Instruction + response birleştir
            full_text = f"{instruction} <SEP> {response}"
            ids = self.tokenizer.encode(full_text)
            
            if len(ids) > 3:
                self.samples.append({
                    'ids': ids[:self.max_len],
                    'action': action,
                    'instruction_len': len(self.tokenizer.encode(instruction))
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        ids = sample['ids']
        
        pad_id = self.tokenizer.pad_id
        padded = ids + [pad_id] * max(0, self.max_len - len(ids))
        padded = padded[:self.max_len]
        
        input_ids = padded[:-1]
        target_ids = padded[1:]
        target_ids = [t if t != pad_id else -100 for t in target_ids]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'action_label': torch.tensor(sample['action'], dtype=torch.long),
        }


# ───────────────────────────── Synthetic Data Generator ───────────

class SyntheticDataGenerator:
    """
    Zengin sentetik veri üretici — çeşitli Türkçe text kalıpları.
    
    10 kategori, 50+ şablon, 200+ kelime ile çeşitli text üretir.
    Model bu zengin kalıpları öğrenerek gerçek dil kapasitesini geliştirir.
    """
    
    def __init__(self, tokenizer, seed: int = 42):
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        
        # ── Zengin kelime havuzları ──
        self.nouns = [
            "kedi", "köpek", "kuş", "balık", "at", "tavşan", "aslan", "kartal",
            "ev", "araba", "gemi", "uçak", "tren", "bisiklet", "köprü", "kule",
            "su", "ateş", "toprak", "hava", "rüzgar", "yağmur", "kar", "fırtına",
            "güneş", "ay", "yıldız", "gezegen", "galaksi", "evren", "dünya", "deniz",
            "ağaç", "çiçek", "orman", "dağ", "nehir", "göl", "ada", "vadi",
            "bilgi", "zaman", "uzay", "enerji", "güç", "hız", "mesafe", "ışık",
            "kitap", "kalem", "masa", "sandalye", "kapı", "pencere", "duvar", "tavan",
            "insan", "çocuk", "doktor", "öğretmen", "mühendis", "bilim insanı", "sanatçı", "müzisyen",
            "hafıza", "karar", "eylem", "düşünce", "duygu", "hayal", "rüya", "umut",
            "algoritma", "sistem", "ağ", "nöron", "katman", "fonksiyon", "değişken", "parametre",
            "strateji", "analiz", "çözüm", "problem", "soru", "cevap", "fikir", "plan",
            "müzik", "resim", "film", "şiir", "roman", "hikaye", "masal", "şarkı",
            "yemek", "ekmek", "su", "çay", "kahve", "meyve", "sebze", "pilav",
        ]
        
        self.adjectives = [
            "büyük", "küçük", "hızlı", "yavaş", "güzel", "çirkin", "uzun", "kısa",
            "sıcak", "soğuk", "eski", "yeni", "güçlü", "zayıf", "akıllı", "cesur",
            "derin", "sığ", "geniş", "dar", "parlak", "karanlık", "temiz", "kirli",
            "tatlı", "acı", "ekşi", "tuzlu", "yumuşak", "sert", "hafif", "ağır",
            "ilginç", "sıkıcı", "karmaşık", "basit", "önemli", "değerli", "nadir", "yaygın",
        ]
        
        self.verbs = [
            "gider", "gelir", "bakar", "görür", "duyar", "bilir", "öğrenir", "anlar",
            "düşünür", "hatırlar", "unutur", "sever", "ister", "yapar", "bulur", "arar",
            "konuşur", "dinler", "yazar", "okur", "çizer", "çalar", "söyler", "anlatır",
            "koşar", "yüzer", "uçar", "atlar", "döner", "durur", "başlar", "biter",
            "çalışır", "üretir", "keşfeder", "icat eder", "geliştirir", "değişir", "büyür", "küçülür",
        ]
        
        self.adverbs = [
            "çok", "az", "hızla", "yavaşça", "dikkatle", "sessizce", "birlikte",
            "ayrıca", "sonra", "önce", "şimdi", "her zaman", "bazen", "genellikle",
            "oldukça", "gerçekten", "kesinlikle", "muhtemelen", "belki", "tabii ki",
        ]
        
        self.connectors = [
            "ve", "ama", "fakat", "ancak", "çünkü", "bu yüzden", "bu nedenle",
            "ayrıca", "üstelik", "dahası", "buna rağmen", "oysa", "halbuki",
            "eğer", "ise", "yoksa", "hem", "ya da", "veya", "ile birlikte",
        ]
        
        self.categories = [
            "hayvan", "nesne", "kavram", "element", "araç", "bilim",
            "doğa", "teknoloji", "sanat", "spor", "yiyecek", "meslek",
        ]
        
        self.emotions = [
            "mutlu", "üzgün", "heyecanlı", "korkmuş", "şaşırmış", "meraklı",
            "sakin", "gergin", "gururlu", "endişeli", "umutlu", "kararlı",
        ]
        
        self.places = [
            "şehirde", "köyde", "ormanda", "deniz kenarında", "dağda", "evde",
            "okulda", "parkta", "bahçede", "laboratuvarda", "kütüphanede", "müzede",
        ]
        
        self.times = [
            "sabah", "öğleden sonra", "akşam", "gece", "gün doğumunda", "gün batımında",
            "ilkbaharda", "yazın", "sonbaharda", "kışın", "her gün", "hafta sonunda",
        ]
    
    def _noun(self):
        return self.rng.choice(self.nouns)
    
    def _adj(self):
        return self.rng.choice(self.adjectives)
    
    def _verb(self):
        return self.rng.choice(self.verbs)
    
    def _adv(self):
        return self.rng.choice(self.adverbs)
    
    def _conn(self):
        return self.rng.choice(self.connectors)
    
    def _emo(self):
        return self.rng.choice(self.emotions)
    
    def _place(self):
        return self.rng.choice(self.places)
    
    def _time(self):
        return self.rng.choice(self.times)
    
    def _num(self, lo=1, hi=100):
        return self.rng.randint(lo, hi)
    
    def _generate_greeting(self):
        greetings = [
            f"Merhaba, ben bir yapay zeka sistemiyim.",
            f"Selam! Bugün sana nasıl yardımcı olabilirim?",
            f"Merhaba, ben {self._noun()} hakkında bilgi verebilirim.",
            f"Hoş geldin! {self._time()} çalışmaya hazırım.",
            f"Günaydın! Bugün {self._adj()} bir gün olacak.",
            f"Selam, {self._noun()} ile ilgili soruların varsa yardımcı olurum.",
            f"Merhaba! Ben {self._adj()} ve {self._adj()} bir asistan olarak buradayım.",
        ]
        return self.rng.choice(greetings)
    
    def _generate_qa(self):
        templates = [
            f"{self._noun()} nedir? {self._noun()} bir {self.rng.choice(self.categories)} ile ilgili kavramdır.",
            f"{self._noun()} ne işe yarar? {self._noun()} {self._adv()} kullanılan {self._adj()} bir {self.rng.choice(self.categories)} türüdür.",
            f"{self._noun()} nasıl çalışır? Önce {self._noun()} {self._verb()}, sonra {self._noun()} {self._verb()}.",
            f"{self._noun()} ile {self._noun()} arasındaki fark nedir? Birincisi {self._adj()}, ikincisi {self._adj()} olarak bilinir.",
            f"{self._noun()} nerede bulunur? Genellikle {self._place()} bulunur.",
            f"{self._noun()} ne zaman kullanılır? {self._time()} kullanılması tavsiye edilir.",
        ]
        return self.rng.choice(templates)
    
    def _generate_math(self):
        a, b = self._num(), self._num()
        ops = [
            (f"{a} artı {b} eşittir {a+b}.", "toplama"),
            (f"{a} eksi {b} eşittir {a-b}.", "çıkarma"),
            (f"{a} çarpı {b} eşittir {a*b}.", "çarpma"),
            (f"{a} ile {b} sayısını topla. Sonuç {a+b} olur.", "toplama"),
            (f"Eğer {a} ile {b} toplanırsa sonuç {a+b} elde edilir.", "toplama"),
            (f"{a} sayısından {b} çıkarılırsa {a-b} kalır.", "çıkarma"),
            (f"İki sayının toplamı: {a} ve {b} toplandığında {a+b} bulunur.", "toplama"),
        ]
        text, _ = self.rng.choice(ops)
        return text
    
    def _generate_definition(self):
        templates = [
            f"{self._noun()} kavramı {self._adj()} bir {self.rng.choice(self.categories)} alanında kullanılır.",
            f"Bir {self._noun()}, {self._adj()} ve {self._adj()} özelliklere sahip bir yapıdır.",
            f"{self._noun()} terimi, {self._noun()} ile {self._noun()} arasındaki ilişkiyi ifade eder.",
            f"{self._adj()} bir {self._noun()}, genellikle {self._place()} gözlemlenir.",
            f"{self._noun()} bilimi, {self._noun()} ve {self._noun()} konularını inceler.",
        ]
        return self.rng.choice(templates)
    
    def _generate_comparison(self):
        n1, n2 = self._noun(), self._noun()
        a1, a2 = self._adj(), self._adj()
        templates = [
            f"{n1} ile {n2} karşılaştırıldığında, {n1} daha {a1}, {n2} ise daha {a2} olarak değerlendirilir.",
            f"{n1} {a1} iken {n2} {a2} kabul edilir. Her ikisi de {self._adv()} önemlidir.",
            f"Hem {n1} hem de {n2} {self._adj()} yapılardır, {self._conn()} farklı amaçlara hizmet ederler.",
            f"{n1} ve {n2} arasındaki temel fark, birinin {a1} diğerinin {a2} olmasıdır.",
        ]
        return self.rng.choice(templates)
    
    def _generate_cause_effect(self):
        templates = [
            f"Eğer {self._noun()} {self._verb()} ise, {self._noun()} {self._adv()} {self._verb()}.",
            f"{self._noun()} {self._adj()} olduğu için {self._noun()} {self._verb()}.",
            f"{self._noun()} {self._verb()}, {self._conn()} {self._noun()} da {self._verb()}.",
            f"{self._time()} {self._noun()} {self._verb()}, bu durum {self._noun()} üzerinde etkili olur.",
            f"Bir {self._noun()} {self._adv()} {self._verb()} zaman, sonuç olarak {self._noun()} {self._verb()}.",
        ]
        return self.rng.choice(templates)
    
    def _generate_conversation(self):
        templates = [
            f"Kullanıcı: {self._noun()} hakkında bilgi ver. Sistem: {self._noun()} {self._adj()} bir {self.rng.choice(self.categories)} alanında yer alır.",
            f"Soru: {self._noun()} ne demek? Cevap: {self._noun()} {self._adj()} ve {self._adj()} bir kavramdır.",
            f"Kullanıcı: Bana {self._adj()} bir {self._noun()} anlat. Sistem: Elbette, {self._noun()} {self._adv()} ilginç bir konudur.",
            f"Soru: {self._noun()} nasıl {self._verb()}? Cevap: Önce {self._noun()} {self._verb()}, ardından sonuç {self._adv()} ortaya çıkar.",
            f"Kullanıcı: Merhaba! Sistem: Merhaba, size nasıl yardımcı olabilirim?",
            f"Kullanıcı: Teşekkürler. Sistem: Rica ederim, başka sorunuz var mı?",
        ]
        return self.rng.choice(templates)
    
    def _generate_command(self):
        templates = [
            f"{self._noun()} hakkında araştırma yap ve sonuçları raporla.",
            f"Lütfen {self._noun()} ile ilgili verileri analiz et.",
            f"{self._noun()} dosyasını oku ve {self._noun()} verilerini çıkar.",
            f"{self._noun()} sistemini {self._adv()} kontrol et ve durumunu bildir.",
            f"Hafızadan {self._noun()} ile ilgili bilgileri getir.",
            f"Yeni bir {self._noun()} oluştur ve {self._adj()} parametreleri ayarla.",
        ]
        return self.rng.choice(templates)
    
    def _generate_story(self):
        n1, n2 = self._noun(), self._noun()
        templates = [
            f"Bir zamanlar {self._adj()} bir {n1} varmış. Bu {n1} {self._place()} yaşarmış. Her {self._time()} {self._verb()} ve {self._emo()} olurmuş.",
            f"{self._time()} {self._adj()} bir {n1} {self._place()} {self._verb()}. Yanında {self._adj()} bir {n2} de varmış.",
            f"Küçük bir {n1}, {self._adj()} bir {n2} ile tanıştı. Birlikte {self._adv()} {self._verb()} ve {self._emo()} oldular.",
            f"Eski zamanlarda {self._place()} {self._adj()} bir {n1} yaşardı. O {self._adv()} {self._verb()} ve herkes onu severdi.",
        ]
        return self.rng.choice(templates)
    
    def _generate_philosophy(self):
        templates = [
            f"Bilgi {self._adj()} bir güçtür, {self._conn()} onu kullanmak sorumluluk gerektirir.",
            f"Düşünmek {self._verb()} demektir. Her {self._noun()} kendi yolunu {self._adv()} bulur.",
            f"Gerçek zeka, {self._noun()} ile {self._noun()} arasındaki bağlantıyı görebilmektir.",
            f"Öğrenme sonsuz bir yolculuktur. Her {self._noun()} yeni bir {self._noun()} keşfetmeye yol açar.",
            f"Bir {self._noun()} ne kadar {self._adj()} olursa, o kadar {self._adv()} {self._verb()}.",
            f"Yapay zeka, insan düşüncesinin {self._adj()} bir yansımasıdır.",
        ]
        return self.rng.choice(templates)
    
    def generate(self, num_samples: int = 1000) -> List[str]:
        """Zengin ve çeşitli sentetik text örnekleri üret."""
        generators = [
            self._generate_greeting,
            self._generate_qa,
            self._generate_math,
            self._generate_definition,
            self._generate_comparison,
            self._generate_cause_effect,
            self._generate_conversation,
            self._generate_command,
            self._generate_story,
            self._generate_philosophy,
        ]
        
        samples = []
        for _ in range(num_samples):
            gen_fn = self.rng.choice(generators)
            samples.append(gen_fn())
        
        # Ayrıca birleşik/uzun örnekler de üret (paragraf seviyesi)
        for _ in range(num_samples // 5):
            # 2-3 cümleyi birleştir
            parts = [self.rng.choice(generators)() for _ in range(self.rng.randint(2, 3))]
            samples.append(" ".join(parts))
        
        return samples
    
    def generate_instruction_data(self, num_samples: int = 500) -> List[Dict]:
        """Zengin instruction-response çiftleri üret."""
        instructions = []
        
        math_ops = [
            ("topla", "+", lambda a, b: a + b),
            ("çıkar", "-", lambda a, b: a - b),
            ("çarp", "*", lambda a, b: a * b),
        ]
        
        for _ in range(num_samples):
            sample_type = self.rng.choice([
                "math", "qa", "action", "greeting", "explain", "compare", "command"
            ])
            
            if sample_type == "math":
                op_name, op_sym, op_fn = self.rng.choice(math_ops)
                a, b = self._num(), self._num()
                instructions.append({
                    "instruction": f"{a} ile {b} sayısını {op_name}",
                    "response": f"{a} {op_sym} {b} = {op_fn(a, b)}",
                    "action": 0
                })
            
            elif sample_type == "qa":
                n = self._noun()
                cat = self.rng.choice(self.categories)
                instructions.append({
                    "instruction": f"{n} nedir?",
                    "response": f"{n}, {self._adj()} bir {cat} kavramıdır. Genellikle {self._place()} karşılaşılır ve {self._adv()} önemli bir rol oynar.",
                    "action": 4
                })
            
            elif sample_type == "action":
                n = self._noun()
                instructions.append({
                    "instruction": f"{n} hakkında bilgi ara",
                    "response": f"{n} araştırılıyor. Sonuçlara göre {n} {self._adj()} bir {self.rng.choice(self.categories)} olarak değerlendirilir.",
                    "action": 3
                })
            
            elif sample_type == "greeting":
                greet_pairs = [
                    ("Merhaba, nasılsın?", "Merhaba! Ben bir yapay zeka olarak her zaman çalışmaya hazırım."),
                    ("Selam!", f"Selam! {self._time()} sana nasıl yardımcı olabilirim?"),
                    ("İyi günler.", f"İyi günler! {self._noun()} hakkında sormak istediğin bir şey var mı?"),
                    ("Naber?", f"İyiyim, teşekkürler! {self._adj()} bir gün geçiriyorum."),
                ]
                inst, resp = self.rng.choice(greet_pairs)
                instructions.append({"instruction": inst, "response": resp, "action": 0})
            
            elif sample_type == "explain":
                n = self._noun()
                instructions.append({
                    "instruction": f"{n} nasıl çalışır?",
                    "response": f"{n} şu şekilde çalışır: Önce {self._noun()} {self._verb()}, ardından {self._noun()} {self._verb()}. Sonuç olarak {self._adj()} bir çıktı elde edilir.",
                    "action": 4
                })
            
            elif sample_type == "compare":
                n1, n2 = self._noun(), self._noun()
                instructions.append({
                    "instruction": f"{n1} ile {n2} arasındaki fark nedir?",
                    "response": f"{n1} {self._adj()} iken {n2} {self._adj()} olarak bilinir. {n1} {self._adv()} {self._verb()}, {n2} ise {self._adv()} {self._verb()}.",
                    "action": 4
                })
            
            elif sample_type == "command":
                n = self._noun()
                instructions.append({
                    "instruction": f"{n} verilerini analiz et",
                    "response": f"{n} verileri analiz ediliyor. {self._adj()} sonuçlar elde edildi. Toplam {self._num()} kayıt incelendi.",
                    "action": 0
                })
        
        return instructions


# ───────────────────────────── Data Pipeline ──────────────────────

class DataPipeline:
    """
    Ana veri pipeline'ı — load → tokenize → batch → iterator.
    
    Kullanım:
        pipeline = DataPipeline(config)
        pipeline.setup_synthetic()  # veya setup_from_file("corpus.txt")
        
        for batch in pipeline.train_loader():
            input_ids = batch['input_ids']
            target_ids = batch['target_ids']
    """
    
    def __init__(self, config: Optional[DataConfig] = None, tokenizer=None):
        if config is None:
            config = DataConfig()
        self.config = config
        
        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = CharTokenizer()
        
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
    
    def setup_from_file(
        self, 
        file_path: str, 
        chunk_mode: str = "sliding",
        stride: int = 128
    ):
        """Text dosyasından veri yükle."""
        # Tokenizer'ı eğit (BPE ise)
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        if isinstance(self.tokenizer, BPETokenizer) and not self.tokenizer._trained:
            self.tokenizer.train(texts)
        elif isinstance(self.tokenizer, CharTokenizer):
            self.tokenizer.fit(texts)
        
        # Dataset oluştur
        full_dataset = TextFileDataset(
            file_path, self.tokenizer, self.config.max_seq_len,
            chunk_mode, stride
        )
        
        # Train/val split
        self._split_dataset(full_dataset)
    
    def setup_from_instruction_file(self, file_path: str):
        """Instruction JSON'dan veri yükle."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [
            f"{item['instruction']} {item.get('response', '')}" 
            for item in data
        ]
        
        if isinstance(self.tokenizer, BPETokenizer) and not self.tokenizer._trained:
            self.tokenizer.train(texts)
        elif isinstance(self.tokenizer, CharTokenizer):
            self.tokenizer.fit(texts)
        
        full_dataset = InstructionDataset(
            file_path, self.tokenizer, self.config.max_seq_len
        )
        
        self._split_dataset(full_dataset)
    
    def setup_synthetic(self, num_samples: int = 2000):
        """Sentetik veri ile başla (quick start)."""
        generator = SyntheticDataGenerator(self.tokenizer, self.config.seed)
        
        # Text üret
        texts = generator.generate(num_samples)
        
        # Tokenizer'ı eğit
        if isinstance(self.tokenizer, BPETokenizer) and not self.tokenizer._trained:
            self.tokenizer.train(texts)
        elif isinstance(self.tokenizer, CharTokenizer):
            self.tokenizer.fit(texts)
        
        # Geçici dosyaya yaz ve dataset oluştur
        tmp_path = Path("/tmp/nexus_synthetic_data.txt")
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(texts))
        
        full_dataset = TextFileDataset(
            str(tmp_path), self.tokenizer, self.config.max_seq_len,
            chunk_mode="line"
        )
        
        self._split_dataset(full_dataset)
    
    def setup_synthetic_instructions(self, num_samples: int = 1000):
        """Sentetik instruction verisi ile başla."""
        generator = SyntheticDataGenerator(self.tokenizer, self.config.seed)
        data = generator.generate_instruction_data(num_samples)
        
        # Tokenizer'ı eğit
        texts = [f"{d['instruction']} {d['response']}" for d in data]
        if isinstance(self.tokenizer, BPETokenizer) and not self.tokenizer._trained:
            self.tokenizer.train(texts)
        elif isinstance(self.tokenizer, CharTokenizer):
            self.tokenizer.fit(texts)
        
        # Geçici JSON dosyasına yaz
        tmp_path = Path("/tmp/nexus_synthetic_instructions.json")
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        full_dataset = InstructionDataset(
            str(tmp_path), self.tokenizer, self.config.max_seq_len
        )
        
        self._split_dataset(full_dataset)
    
    def _split_dataset(self, full_dataset: Dataset):
        """Dataset'i train/val olarak böl."""
        total = len(full_dataset)
        train_size = int(total * self.config.train_split)
        val_size = total - train_size
        
        generator = torch.Generator().manual_seed(self.config.seed)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
    
    def train_loader(self) -> DataLoader:
        """Eğitim DataLoader'ı döndür."""
        if self.train_dataset is None:
            raise ValueError("Dataset henüz yüklenmedi! setup_* metodlarından birini çağır.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            drop_last=True
        )
    
    def val_loader(self) -> DataLoader:
        """Doğrulama DataLoader'ı döndür."""
        if self.val_dataset is None:
            raise ValueError("Dataset henüz yüklenmedi!")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            drop_last=False
        )
    
    def train_generator(self) -> Iterator:
        """
        Trainer uyumlu generator — (input_ids, target_ids) tuple'ları döner.
        ContinualTrainer.train() ile kullanılır.
        """
        loader = self.train_loader()
        for batch in loader:
            yield batch['input_ids'], batch['target_ids']
    
    def get_info(self) -> Dict[str, Any]:
        """Pipeline bilgileri."""
        return {
            'tokenizer_type': type(self.tokenizer).__name__,
            'vocab_size': self.tokenizer.vocab_size,
            'max_seq_len': self.config.max_seq_len,
            'batch_size': self.config.batch_size,
            'train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'val_samples': len(self.val_dataset) if self.val_dataset else 0,
        }
