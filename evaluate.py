"""
evaluate.py — Model Değerlendirme Sistemi (Proto-AGI)

Eğitilmiş modelin kalitesini ölçer:
  1. Perplexity (ne kadar "şaşkın")
  2. Token doğruluğu (next-token prediction accuracy)
  3. Text üretim kalitesi (tekrar oranı, çeşitlilik)
  4. Action head istatistikleri
  5. Hafıza kullanımı

Kullanım:
    python evaluate.py                          # Son checkpoint
    python evaluate.py --checkpoint checkpoints/nexus_epoch_0050.pt
    python evaluate.py --generate_samples 20    # 20 örnek üret
"""

import torch
import torch.nn.functional as F
import argparse
import math
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

from nexus_core import NexusCore, NexusConfig
from tokenizer import BPETokenizer, CharTokenizer, TokenizerConfig
from data_pipeline import DataPipeline, DataConfig, SyntheticDataGenerator
from action_space import ActionType


def parse_args():
    parser = argparse.ArgumentParser(description="Proto-AGI Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer_dir", type=str, default="./tokenizer_data")
    parser.add_argument("--synthetic_samples", type=int, default=500)
    parser.add_argument("--generate_samples", type=int, default=10)
    parser.add_argument("--max_gen_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=30)
    return parser.parse_args()


def load_tokenizer(tok_dir):
    """Kaydedilmiş tokenizer'ı yükle."""
    tok_file = Path(tok_dir) / "tokenizer.json"
    if not tok_file.exists():
        print("  ⚠️ Tokenizer bulunamadı, CharTokenizer fallback")
        ct = CharTokenizer()
        ct.fit(["abcçdefgğhıijklmnoöprsştuüvyz 0123456789.,!?"])
        return ct
    
    with open(tok_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if data.get('type') == 'char':
        tok = CharTokenizer()
        tok.load(tok_dir)
    else:
        tok = BPETokenizer()
        tok.load(tok_dir)
    
    return tok


def find_latest_checkpoint(cp_dir="./checkpoints"):
    """En son checkpoint."""
    p = Path(cp_dir)
    if not p.exists():
        return None
    cps = sorted(p.glob("nexus_*.pt"))
    return str(cps[-1]) if cps else None


def load_model(checkpoint_path, tokenizer):
    """Model'i checkpoint'tan yükle."""
    cp = torch.load(checkpoint_path, weights_only=False)
    
    config = cp.get('config')
    if config and isinstance(config, NexusConfig):
        nexus_config = config
    elif config and hasattr(config, 'nexus_config'):
        nexus_config = config.nexus_config
    else:
        nexus_config = NexusConfig(vocab_size=tokenizer.vocab_size)
    
    nexus_config.vocab_size = tokenizer.vocab_size
    model = NexusCore(nexus_config)
    model.load_state_dict(cp['model_state_dict'], strict=False)
    model.eval()
    
    epoch = cp.get('epoch', '?')
    step = cp.get('global_step', '?')
    return model, epoch, step


# ────────────── Metrikler ──────────────

def compute_perplexity(model, tokenizer, texts, seq_len=128, batch_size=16):
    """Validation verisi üzerinde perplexity hesapla."""
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Encode
        batch_ids = []
        for text in batch_texts:
            ids = tokenizer.encode(text)
            if len(ids) < 3:
                continue
            ids = ids[:seq_len]
            # Pad
            if len(ids) < seq_len:
                pad_id = getattr(tokenizer, 'pad_id', 0)
                ids = ids + [pad_id] * (seq_len - len(ids))
            batch_ids.append(ids)
        
        if not batch_ids:
            continue
        
        input_tensor = torch.tensor(batch_ids, dtype=torch.long)
        input_ids = input_tensor[:, :-1]
        target_ids = input_tensor[:, 1:]
        
        with torch.no_grad():
            output = model(input_ids, use_memory=False, deterministic=True)
            logits = output['language_logits']
            
            # Cross entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=getattr(tokenizer, 'pad_id', 0),
                reduction='sum'
            )
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            mask = target_ids != getattr(tokenizer, 'pad_id', 0)
            correct = (preds == target_ids) & mask
            correct_tokens += correct.sum().item()
            total_tokens += mask.sum().item()
            total_loss += loss.item()
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))  # Cap at exp(20) to avoid overflow
    accuracy = correct_tokens / max(total_tokens, 1)
    
    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'total_tokens': total_tokens,
    }


def generate_sample(model, tokenizer, prompt, max_tokens=50, temperature=0.8, top_k=30,
                     repetition_penalty=1.5, ngram_block_size=3):
    """Tek bir text örneği üret (n-gram blocking + repetition/frequency penalty)."""
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long)
    
    generated_ids = list(ids)
    new_ids = []  # sadece yeni üretilen token'lar
    eos_id = getattr(tokenizer, 'eos_id', -1)
    
    for _ in range(max_tokens):
        with torch.no_grad():
            output = model(input_ids[:, -128:], use_memory=True, deterministic=False)
            logits = output['language_logits'][0, -1, :]
            
            # --- 1. Repetition Penalty ---
            if new_ids and repetition_penalty != 1.0:
                from collections import Counter
                freq = Counter(new_ids)
                for token_id, count in freq.items():
                    # Frequency-scaled penalty: ne kadar çok tekrar → o kadar sert ceza
                    penalty = repetition_penalty * (1.0 + 0.2 * min(count, 5))
                    if logits[token_id] > 0:
                        logits[token_id] /= penalty
                    else:
                        logits[token_id] *= penalty
            
            # --- 2. N-gram Blocking ---
            if len(new_ids) >= ngram_block_size:
                # Son (n-1) token'ı al, bu prefix ile daha önce görülen devamları engelle
                prefix = tuple(new_ids[-(ngram_block_size - 1):])
                for i in range(len(new_ids) - ngram_block_size + 1):
                    existing = tuple(new_ids[i:i + ngram_block_size - 1])
                    if existing == prefix:
                        blocked_id = new_ids[i + ngram_block_size - 1]
                        logits[blocked_id] = float('-inf')
            
            # --- 3. Temperature ---
            logits = logits / temperature
            
            # --- 4. Top-k filtering ---
            if top_k > 0:
                values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(0, indices, values)
                logits = mask
            
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
        
        if next_id == eos_id:
            break
        
        generated_ids.append(next_id)
        new_ids.append(next_id)
        input_ids = torch.tensor([generated_ids], dtype=torch.long)
    
    generated_text = tokenizer.decode(generated_ids[len(ids):])
    action_idx = output['action_idx'][0].item()
    confidence = output['confidence'][0].item()
    
    return {
        'prompt': prompt,
        'generated': generated_text,
        'total_tokens': len(generated_ids) - len(ids),
        'action': action_idx,
        'confidence': confidence,
    }


def analyze_generation_quality(samples: List[Dict]) -> Dict:
    """Üretim kalitesini analiz et."""
    all_text = " ".join(s['generated'] for s in samples)
    
    # Karakter çeşitliliği
    char_counts = Counter(all_text)
    unique_chars = len(char_counts)
    total_chars = len(all_text)
    
    # Tekrar oranı (bigram)
    bigrams = [all_text[i:i+2] for i in range(len(all_text)-1)]
    bigram_counts = Counter(bigrams)
    unique_bigrams = len(bigram_counts)
    total_bigrams = len(bigrams)
    bigram_diversity = unique_bigrams / max(total_bigrams, 1)
    
    # Kelime çeşitliliği
    words = all_text.split()
    unique_words = len(set(words))
    total_words = len(words)
    word_diversity = unique_words / max(total_words, 1)
    
    # Ortalama cümle uzunluğu
    avg_length = sum(s['total_tokens'] for s in samples) / max(len(samples), 1)
    
    # Türkçe karakter oranı
    turkish_chars = set("abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ ")
    turkish_ratio = sum(1 for c in all_text if c in turkish_chars) / max(total_chars, 1)
    
    # Boşluk oranı (kelime oluşturma becerisi)
    space_ratio = all_text.count(' ') / max(total_chars, 1)
    
    # Action dağılımı
    action_counts = Counter(s['action'] for s in samples)
    avg_confidence = sum(s['confidence'] for s in samples) / max(len(samples), 1)
    
    return {
        'unique_chars': unique_chars,
        'char_diversity': unique_chars / max(total_chars, 1),
        'bigram_diversity': bigram_diversity,
        'word_diversity': word_diversity,
        'unique_words': unique_words,
        'total_words': total_words,
        'avg_token_length': avg_length,
        'turkish_char_ratio': turkish_ratio,
        'space_ratio': space_ratio,
        'action_distribution': dict(action_counts),
        'avg_confidence': avg_confidence,
    }


def grade_model(perplexity_metrics, quality_metrics) -> Tuple[str, str]:
    """Modele not ver ve yorumla."""
    ppl = perplexity_metrics['perplexity']
    acc = perplexity_metrics['accuracy']
    diversity = quality_metrics['bigram_diversity']
    turkish = quality_metrics['turkish_char_ratio']
    space_ratio = quality_metrics['space_ratio']
    
    score = 0
    notes = []
    
    # 1. Perplexity puanlama (max 25)
    if ppl < 5:
        score += 25
        notes.append("🟢 Perplexity mükemmel (<5)")
    elif ppl < 15:
        score += 18
        notes.append("🟡 Perplexity iyi (5-15)")
    elif ppl < 50:
        score += 10
        notes.append("🟠 Perplexity orta (15-50)")
    elif ppl < 200:
        score += 5
        notes.append("🔴 Perplexity yüksek (50-200)")
    else:
        score += 0
        notes.append("⚫ Perplexity çok yüksek (>200) — neredeyse rastgele")
    
    # 2. Accuracy puanlama (max 20)
    if acc > 0.6:
        score += 20
        notes.append("🟢 Token accuracy yüksek (>60%)")
    elif acc > 0.3:
        score += 12
        notes.append("🟡 Token accuracy orta (30-60%)")
    elif acc > 0.1:
        score += 5
        notes.append("🟠 Token accuracy düşük (10-30%)")
    else:
        score += 0
        notes.append("🔴 Token accuracy çok düşük (<10%)")
    
    # 3. Çeşitlilik puanlama (max 20)
    if 0.4 < diversity < 0.85:
        score += 20
        notes.append("🟢 Bigram çeşitliliği sağlıklı")
    elif diversity > 0.85:
        score += 10
        notes.append("🟡 Bigram çeşitliliği fazla yüksek (rastgele olabilir)")
    elif diversity > 0.2:
        score += 8
        notes.append("🟠 Bigram çeşitliliği düşük (tekrarlayan)")
    else:
        score += 0
        notes.append("🔴 Bigram çeşitliliği çok düşük (aşırı tekrarlayan)")
    
    # 4. Türkçe karakter oranı (max 15)
    if turkish > 0.7:
        score += 15
        notes.append("🟢 Türkçe karakter oranı yüksek")
    elif turkish > 0.4:
        score += 8
        notes.append("🟡 Türkçe karakter oranı orta")
    else:
        score += 0
        notes.append("🔴 Türkçe karakter oranı düşük")
    
    # 5. Kelime oluşturma becerisi (max 20) — boşluk oranı + word diversity
    word_div = quality_metrics.get('word_diversity', 0)
    if 0.08 < space_ratio < 0.25 and word_div > 0.5:
        score += 20
        notes.append("🟢 Kelime oluşturma becerisi iyi (anlamlı boşluk ve kelime çeşitliliği)")
    elif 0.05 < space_ratio < 0.3 and word_div > 0.3:
        score += 12
        notes.append("🟡 Kelime oluşturma becerisi orta")
    elif space_ratio < 0.05:
        score += 0
        notes.append("🔴 Kelime oluşturamıyor (boşluk oranı çok düşük → karakter çorbası)")
    else:
        score += 5
        notes.append("🟠 Kelime oluşturma becerisi zayıf")
    
    # Genel not
    if score >= 85:
        grade = "A — Mükemmel 🏆"
    elif score >= 70:
        grade = "B — İyi 👍"
    elif score >= 50:
        grade = "C — Orta 📊"
    elif score >= 30:
        grade = "D — Zayıf, daha fazla eğitim gerekli ⚠️"
    else:
        grade = "F — Yetersiz, çok daha fazla eğitim gerekli ❌"
    
    return grade, score, notes


# ────────────── Ana Akış ──────────────

def run_evaluation(args):
    print(r"""
    ╔═══════════════════════════════════════════════════╗
    ║      NexusCore — Model Değerlendirmesi           ║
    ║      "Zekânın ne kadar geliştiğini ölç"          ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    # ─── Yükleme ───
    print("  ═══ Yükleme ═══")
    tokenizer = load_tokenizer(args.tokenizer_dir)
    print(f"  📝 Tokenizer: {type(tokenizer).__name__}, vocab={tokenizer.vocab_size}")
    
    cp_path = args.checkpoint or find_latest_checkpoint()
    if not cp_path:
        print("  ❌ Checkpoint bulunamadı! Önce eğitim yap.")
        return
    
    model, epoch, step = load_model(cp_path, tokenizer)
    info = model.get_model_info()
    print(f"  🧠 Model: {info['total_params_M']} (epoch={epoch}, step={step})")
    
    # ─── Validation verisi ───
    print("\n  ═══ 1. Perplexity & Accuracy ═══")
    gen = SyntheticDataGenerator(None, seed=99)
    val_texts = gen.generate(args.synthetic_samples)
    
    ppl_metrics = compute_perplexity(model, tokenizer, val_texts)
    print(f"  📊 Perplexity:  {ppl_metrics['perplexity']:.2f}")
    print(f"  📊 Avg Loss:    {ppl_metrics['avg_loss']:.4f}")
    print(f"  📊 Accuracy:    {ppl_metrics['accuracy']*100:.1f}%")
    print(f"  📊 Eval tokens: {ppl_metrics['total_tokens']}")
    
    # ─── Text Üretim ───
    print(f"\n  ═══ 2. Text Üretim Örnekleri ({args.generate_samples} adet) ═══")
    prompts = [
        "Merhaba",
        "yapay zeka",
        "hesapla",
        "bellek",
        "strateji",
        "eğitim",
        "analiz",
        "sistem",
        "çözüm",
        "karar",
        "öğrenme",
        "hafıza sistemi",
        "otonom",
        "meta",
        "model eğitimi",
    ]
    
    samples = []
    for i in range(min(args.generate_samples, len(prompts))):
        sample = generate_sample(
            model, tokenizer, prompts[i],
            max_tokens=args.max_gen_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        samples.append(sample)
        action_name = ActionType(sample['action']).name if sample['action'] < len(ActionType) else f"#{sample['action']}"
        print(f"  [{i+1}] \"{sample['prompt']}\" → \"{sample['generated'][:50]}\"")
        print(f"      Action: {action_name} | Conf: {sample['confidence']:.3f} | Tokens: {sample['total_tokens']}")
    
    # ─── Kalite Analizi ───
    print(f"\n  ═══ 3. Üretim Kalitesi Analizi ═══")
    quality = analyze_generation_quality(samples)
    print(f"  📊 Unique chars:     {quality['unique_chars']}")
    print(f"  📊 Bigram diversity: {quality['bigram_diversity']:.3f}")
    print(f"  📊 Word diversity:   {quality['word_diversity']:.3f} ({quality['unique_words']}/{quality['total_words']} kelime)")
    print(f"  📊 Avg token length: {quality['avg_token_length']:.1f}")
    print(f"  📊 Turkish ratio:    {quality['turkish_char_ratio']*100:.1f}%")
    print(f"  📊 Space ratio:      {quality['space_ratio']*100:.1f}%")
    print(f"  📊 Action dağılımı:  {quality['action_distribution']}")
    print(f"  📊 Avg confidence:   {quality['avg_confidence']:.3f}")
    
    # ─── Hafıza ───
    print(f"\n  ═══ 4. Hafıza Durumu ═══")
    try:
        model.memory.load_all()
    except:
        pass
    mem_stats = model.memory.get_stats()
    for k, v in mem_stats.items():
        print(f"  📊 {k}: {v}")
    
    # ─── Genel Not ───
    print(f"\n  ═══ 5. Genel Değerlendirme ═══")
    grade, score, notes = grade_model(ppl_metrics, quality)
    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │  NOT: {grade}")
    print(f"  │  SKOR: {score}/100")
    print(f"  ├─────────────────────────────────────┤")
    for note in notes:
        print(f"  │  {note}")
    print(f"  └─────────────────────────────────────┘")
    
    # Tavsiyeler
    print(f"\n  ═══ 6. Tavsiyeler ═══")
    if ppl_metrics['perplexity'] > 50:
        print("  💡 Perplexity çok yüksek — daha fazla epoch ile eğit:")
        print("     python train_real.py --epochs 50 --synthetic_samples 10000")
    if quality['bigram_diversity'] > 0.85:
        print("  💡 Üretim çok rastgele — temperature'ı düşür (0.5-0.7)")
    if quality['turkish_char_ratio'] < 0.5:
        print("  💡 Türkçe karakter oranı düşük — daha fazla Türkçe corpus ekle")
    if ppl_metrics['accuracy'] < 0.3:
        print("  💡 Token accuracy düşük — d_model veya n_layers artır")
    if score >= 60:
        print("  🎉 Model iyi durumda! Gerçek corpus ile eğitmeyi dene.")
    
    print(f"\n  ═══ Değerlendirme tamamlandı ═══")
    return ppl_metrics, quality, grade, score


if __name__ == "__main__":
    args = parse_args()
    try:
        run_evaluation(args)
    except Exception as e:
        print(f"\n  ❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
