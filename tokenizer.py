"""
tokenizer.py — BPE Tokenizer (Proto-AGI)

Gerçek text verilerini token ID'lerine dönüştürür.
İki mod:
  1. CharLevel:  Hızlı prototip için karakter bazlı tokenizer
  2. BPE:        Byte-Pair Encoding — gerçek dil modelleme için

Harici bağımlılık: yok (saf Python). 
Opsiyonel: sentencepiece veya tiktoken varsa onları kullanır.
"""

import json
import re
import os
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field


# ───────────────────────────── Config ─────────────────────────────

@dataclass
class TokenizerConfig:
    """Tokenizer konfigürasyonu."""
    vocab_size: int = 8_000         # Hedef vocabulary boyutu
    min_frequency: int = 2          # Min token frekansı
    special_tokens: List[str] = field(default_factory=lambda: [
        "<PAD>",    # Padding
        "<UNK>",    # Unknown token
        "<BOS>",    # Beginning of sequence
        "<EOS>",    # End of sequence
        "<SEP>",    # Separator
        "<MASK>",   # Masking (MLM tarzı eğitim için)
        "<ACT>",    # Action head tetikleyici
        "<MEM>",    # Memory query tetikleyici
        "<THK>",    # Think/reflect tetikleyici
    ])
    save_dir: str = "./tokenizer_data"
    lowercase: bool = False         # Küçük harfe dönüştür mü?


# ───────────────────────────── Char-Level Tokenizer ───────────────

class CharTokenizer:
    """
    Karakter bazlı tokenizer — hızlı prototipleme için.
    Her karakter bir token. Basit ama vocab boyutu küçük.
    """
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        if config is None:
            config = TokenizerConfig()
        self.config = config
        
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        
        # Special tokens
        for i, token in enumerate(config.special_tokens):
            self.char_to_id[token] = i
            self.id_to_char[i] = token
        
        self._next_id = len(config.special_tokens)
        self.pad_id = self.char_to_id["<PAD>"]
        self.unk_id = self.char_to_id["<UNK>"]
        self.bos_id = self.char_to_id["<BOS>"]
        self.eos_id = self.char_to_id["<EOS>"]
    
    def fit(self, texts: List[str]):
        """Corpus'tan karakter vocabulary'si oluştur."""
        chars = set()
        for text in texts:
            if self.config.lowercase:
                text = text.lower()
            chars.update(text)
        
        for char in sorted(chars):
            if char not in self.char_to_id:
                self.char_to_id[char] = self._next_id
                self.id_to_char[self._next_id] = char
                self._next_id += 1
    
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Text → token ID listesi."""
        if self.config.lowercase:
            text = text.lower()
        
        ids = []
        if add_special:
            ids.append(self.bos_id)
        
        for char in text:
            ids.append(self.char_to_id.get(char, self.unk_id))
        
        if add_special:
            ids.append(self.eos_id)
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Token ID listesi → text."""
        chars = []
        for id_ in ids:
            token = self.id_to_char.get(id_, "<UNK>")
            if token not in self.config.special_tokens:
                chars.append(token)
        return "".join(chars)
    
    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)
    
    def save(self, path: Optional[str] = None):
        """CharTokenizer'ı diske kaydet."""
        save_dir = Path(path or self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        data = {
            'type': 'char',
            'char_to_id': self.char_to_id,
            'id_to_char': {int(k): v for k, v in self.id_to_char.items()},
            'config': {
                'vocab_size': self.config.vocab_size,
                'special_tokens': self.config.special_tokens,
                'lowercase': self.config.lowercase,
            }
        }
        
        with open(save_dir / "tokenizer.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: Optional[str] = None):
        """CharTokenizer'ı diskten yükle."""
        save_dir = Path(path or self.config.save_dir)
        
        with open(save_dir / "tokenizer.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.char_to_id = data['char_to_id']
        self.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
        self._next_id = max(int(k) for k in self.id_to_char.keys()) + 1


# ───────────────────────────── BPE Tokenizer ──────────────────────

class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer — saf Python implementasyonu.
    
    1. Corpus'u karakter seviyesinde tokenize et
    2. En sık geçen karakter çiftini bul
    3. Bu çifti tek bir token olarak birleştir
    4. vocab_size'a ulaşana kadar tekrarla
    
    Sonuç: Sık kelimeler tek token, nadir kelimeler karakter parçaları.
    """
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        if config is None:
            config = TokenizerConfig()
        self.config = config
        
        # Vocabulary
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # BPE merge kuralları (sıralı)
        self.merges: List[Tuple[str, str]] = []
        
        # Special tokens
        for i, token in enumerate(config.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        self._next_id = len(config.special_tokens)
        self.pad_id = self.token_to_id["<PAD>"]
        self.unk_id = self.token_to_id["<UNK>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        
        self._trained = False
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """Text'i kelime ve boşluk parçalarına ayır."""
        if self.config.lowercase:
            text = text.lower()
        # Kelime sınırlarında böl, boşlukları koru
        pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+"""
        return re.findall(pattern, text)
    
    def _get_word_freqs(self, texts: List[str]) -> Dict[Tuple[str, ...], int]:
        """Kelime frekanslarını hesapla (karakter tuple'ları olarak)."""
        word_freqs: Dict[Tuple[str, ...], int] = Counter()
        
        for text in texts:
            words = self._pre_tokenize(text)
            for word in words:
                # Her kelimeyi karakter tuple'ına çevir, sona </w> ekle
                chars = tuple(word) + ("</w>",)
                word_freqs[chars] += 1
        
        return word_freqs
    
    def _get_pair_freqs(
        self, word_freqs: Dict[Tuple[str, ...], int]
    ) -> Counter:
        """Ardışık çift frekanslarını hesapla."""
        pair_freqs: Counter = Counter()
        
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += freq
        
        return pair_freqs
    
    def _merge_pair(
        self, 
        word_freqs: Dict[Tuple[str, ...], int],
        pair: Tuple[str, str]
    ) -> Dict[Tuple[str, ...], int]:
        """En sık çifti birleştir."""
        new_word_freqs: Dict[Tuple[str, ...], int] = {}
        merged = pair[0] + pair[1]
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def train(self, texts: List[str], verbose: bool = True):
        """
        BPE eğitimi — corpus'tan vocabulary oluştur.
        
        texts: Eğitim corpus'u (text listesi)
        """
        if verbose:
            print(f"  BPE Training — target vocab: {self.config.vocab_size}")
        
        # 1. Kelime frekanslarını hesapla
        word_freqs = self._get_word_freqs(texts)
        
        # 2. Tüm karakterleri vocabulary'ye ekle
        all_chars: Set[str] = set()
        for word in word_freqs:
            for char in word:
                all_chars.add(char)
        
        for char in sorted(all_chars):
            if char not in self.token_to_id:
                self.token_to_id[char] = self._next_id
                self.id_to_token[self._next_id] = char
                self._next_id += 1
        
        if verbose:
            print(f"  Initial vocab: {len(self.token_to_id)} (chars + special)")
        
        # 3. BPE merge döngüsü
        num_merges = self.config.vocab_size - len(self.token_to_id)
        
        for step in range(max(0, num_merges)):
            pair_freqs = self._get_pair_freqs(word_freqs)
            
            if not pair_freqs:
                break
            
            # En sık çifti bul
            best_pair = pair_freqs.most_common(1)[0]
            pair, freq = best_pair
            
            if freq < self.config.min_frequency:
                break
            
            # Birleştir
            merged_token = pair[0] + pair[1]
            word_freqs = self._merge_pair(word_freqs, pair)
            
            # Vocabulary'ye ekle
            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = self._next_id
                self.id_to_token[self._next_id] = merged_token
                self._next_id += 1
            
            self.merges.append(pair)
            
            if verbose and (step + 1) % 500 == 0:
                print(f"  Merge {step + 1}/{num_merges} — "
                      f"'{pair[0]}' + '{pair[1]}' → '{merged_token}' (freq={freq})")
        
        self._trained = True
        
        if verbose:
            print(f"  Final vocab: {len(self.token_to_id)} tokens, "
                  f"{len(self.merges)} merges")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tek bir kelimeyi BPE token'larına çevir."""
        tokens = list(word) + ["</w>"]
        
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens = tokens[:i] + [pair[0] + pair[1]] + tokens[i + 2:]
                else:
                    i += 1
        
        return tokens
    
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Text → token ID listesi."""
        words = self._pre_tokenize(text)
        
        ids = []
        if add_special:
            ids.append(self.bos_id)
        
        for word in words:
            tokens = self._tokenize_word(word)
            for token in tokens:
                ids.append(self.token_to_id.get(token, self.unk_id))
        
        if add_special:
            ids.append(self.eos_id)
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Token ID listesi → text."""
        tokens = []
        for id_ in ids:
            token = self.id_to_token.get(id_, "<UNK>")
            if token not in self.config.special_tokens:
                tokens.append(token)
        
        # </w> işaretlerini boşluğa çevir ve birleştir
        text = "".join(tokens)
        text = text.replace("</w>", "")
        return text
    
    def encode_batch(self, texts: List[str], add_special: bool = True) -> List[List[int]]:
        """Birden fazla text'i encode et."""
        return [self.encode(t, add_special) for t in texts]
    
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)
    
    def save(self, path: Optional[str] = None):
        """Tokenizer'ı diske kaydet."""
        save_dir = Path(path or self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        data = {
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
            'merges': self.merges,
            'config': {
                'vocab_size': self.config.vocab_size,
                'min_frequency': self.config.min_frequency,
                'special_tokens': self.config.special_tokens,
                'lowercase': self.config.lowercase,
            }
        }
        
        with open(save_dir / "tokenizer.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: Optional[str] = None):
        """Tokenizer'ı diskten yükle."""
        save_dir = Path(path or self.config.save_dir)
        
        with open(save_dir / "tokenizer.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        self.merges = [tuple(m) for m in data['merges']]
        self._next_id = max(int(k) for k in self.id_to_token.keys()) + 1
        self._trained = True


# ───────────────────────────── Utility: Padding ───────────────────

def pad_sequences(
    sequences: List[List[int]], 
    max_len: int, 
    pad_id: int = 0,
    truncate: bool = True
) -> List[List[int]]:
    """
    Sequence'ları aynı uzunluğa getir.
    Kısa olanları pad'le, uzun olanları kes.
    """
    result = []
    for seq in sequences:
        if truncate and len(seq) > max_len:
            seq = seq[:max_len]
        
        padded = seq + [pad_id] * max(0, max_len - len(seq))
        result.append(padded)
    
    return result
