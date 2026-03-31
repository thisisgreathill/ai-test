"""
train_real.py — Gerçek Eğitim Script'i (Proto-AGI)

Tokenizer + Data Pipeline + ContinualTrainer ile tam eğitim döngüsü.
İlk çalıştırmada sentetik veri kullanır, gerçek corpus varsa onunla eğitir.

Kullanım:
    python train_real.py                           # Sentetik veri ile
    python train_real.py --corpus corpus.txt       # Gerçek corpus ile
    python train_real.py --epochs 50 --d_model 256 # Özel parametreler
"""

import torch
import argparse
import time
import sys
from pathlib import Path

from nexus_core import NexusCore, NexusConfig
from memory import MemoryConfig
from tokenizer import BPETokenizer, CharTokenizer, TokenizerConfig
from data_pipeline import DataPipeline, DataConfig, SyntheticDataGenerator
from trainer import ContinualTrainer, TrainingConfig
from reward_engine import RewardConfig
from constitution import ConstitutionConfig
from meta_cognition import MetaCognitionConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Proto-AGI Training")
    
    # Data
    parser.add_argument("--corpus", type=str, default=None,
                       help="Text corpus dosyası (.txt)")
    parser.add_argument("--instructions", type=str, default=None,
                       help="Instruction-response JSON dosyası")
    parser.add_argument("--synthetic_samples", type=int, default=5000,
                       help="Sentetik veri sayısı (corpus yoksa)")
    
    # Model
    parser.add_argument("--d_model", type=int, default=128,
                       help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=3,
                       help="Transformer layer sayısı")
    parser.add_argument("--n_heads", type=int, default=4,
                       help="Attention head sayısı")
    parser.add_argument("--d_ff", type=int, default=512,
                       help="Feed-forward hidden dim")
    parser.add_argument("--vocab_size", type=int, default=4000,
                       help="BPE vocab boyutu")
    
    # Training
    parser.add_argument("--epochs", type=int, default=20,
                       help="Epoch sayısı")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch boyutu")
    parser.add_argument("--seq_len", type=int, default=128,
                       help="Max sequence uzunluğu")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping")
    
    # Tokenizer
    parser.add_argument("--tokenizer", type=str, default="char",
                       choices=["char", "bpe"],
                       help="Tokenizer tipi")
    
    # Checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Checkpoint dizini")
    parser.add_argument("--resume", type=str, default=None,
                       help="Checkpoint'tan devam et")
    
    return parser.parse_args()


def build_tokenizer(args, texts):
    """Tokenizer oluştur ve eğit."""
    if args.tokenizer == "bpe":
        print("  📝 BPE Tokenizer eğitiliyor...")
        tok_config = TokenizerConfig(
            vocab_size=args.vocab_size,
            min_frequency=2
        )
        tokenizer = BPETokenizer(tok_config)
        tokenizer.train(texts, verbose=True)
        tokenizer.save()
        print(f"  ✅ BPE Tokenizer: {tokenizer.vocab_size} tokens, "
              f"{len(tokenizer.merges)} merges")
    else:
        print("  📝 Char Tokenizer oluşturuluyor...")
        tokenizer = CharTokenizer()
        tokenizer.fit(texts)
        print(f"  ✅ Char Tokenizer: {tokenizer.vocab_size} tokens")
    
    return tokenizer


def build_pipeline(args, tokenizer):
    """Data pipeline oluştur."""
    data_config = DataConfig(
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
        shuffle=True,
    )
    pipeline = DataPipeline(data_config, tokenizer=tokenizer)
    
    if args.corpus:
        print(f"  📂 Corpus yükleniyor: {args.corpus}")
        pipeline.setup_from_file(args.corpus, chunk_mode="sliding", stride=64)
    elif args.instructions:
        print(f"  📂 Instructions yükleniyor: {args.instructions}")
        pipeline.setup_from_instruction_file(args.instructions)
    else:
        print(f"  🧪 Sentetik veri üretiliyor: {args.synthetic_samples} örnek")
        pipeline.setup_synthetic(num_samples=args.synthetic_samples)
    
    info = pipeline.get_info()
    print(f"  ✅ Pipeline: train={info['train_samples']}, "
          f"val={info['val_samples']}, vocab={info['vocab_size']}")
    
    return pipeline


def build_trainer(args, vocab_size):
    """Model ve trainer oluştur."""
    nexus_config = NexusConfig(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
    )
    
    training_config = TrainingConfig(
        nexus_config=nexus_config,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_epochs=args.epochs,
        grad_clip=args.grad_clip,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=10,
        replay_min_size=args.batch_size * 3,
        fast_mode=True,             # Hızlı eğitim modu
        heavy_step_interval=10,     # Her 10 step'te bir tam pipeline
    )
    
    trainer = ContinualTrainer(training_config)
    
    # Checkpoint'tan devam
    if args.resume:
        print(f"  📥 Checkpoint yükleniyor: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"  ✅ Epoch {trainer.epoch}, step {trainer.global_step}'den devam")
    
    return trainer


def run_training(args):
    """Ana eğitim akışı."""
    print(r"""
    ╔═══════════════════════════════════════════════════╗
    ║      NexusCore — Real Training Pipeline          ║
    ║      "Zeka çekirdeğini besle, evrilmesini izle"  ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    # ─── 1. Veri Toplama ───
    print("  ═══ Step 1: Veri Hazırlığı ═══")
    
    if args.corpus:
        with open(args.corpus, 'r', encoding='utf-8') as f:
            texts = f.readlines()
    elif args.instructions:
        with open(args.instructions, 'r', encoding='utf-8') as f:
            data = json.load(f)
            texts = [f"{d['instruction']} {d.get('response', '')}" for d in data]
    else:
        gen = SyntheticDataGenerator(None, seed=42)
        texts = gen.generate(args.synthetic_samples)
        # Ek instruction data da üret
        inst_data = gen.generate_instruction_data(args.synthetic_samples // 2)
        texts.extend([f"{d['instruction']} {d['response']}" for d in inst_data])
    
    print(f"  📊 Toplam {len(texts)} text örneği")
    
    # ─── 2. Tokenizer ───
    print("\n  ═══ Step 2: Tokenizer ═══")
    tokenizer = build_tokenizer(args, texts)
    
    # ─── 3. Data Pipeline ───
    print("\n  ═══ Step 3: Data Pipeline ═══")
    pipeline = build_pipeline(args, tokenizer)
    
    # ─── 4. Model & Trainer ───
    print("\n  ═══ Step 4: Model & Trainer ═══")
    trainer = build_trainer(args, tokenizer.vocab_size)
    
    info = trainer.model.get_model_info()
    print(f"  🧠 Model: {info['total_params_M']} parameters")
    print(f"  Config: d_model={args.d_model}, layers={args.n_layers}, "
          f"heads={args.n_heads}")
    
    # ─── 5. Eğitim ───
    print("\n  ═══ Step 5: Training ═══")
    
    def data_gen_fn():
        return pipeline.train_generator()
    
    trainer.train(data_gen_fn)
    
    # ─── 6. Sonuçlar ───
    total_time = time.time() - start_time
    
    print(f"\n  ═══ Eğitim Tamamlandı ═══")
    print(f"  ⏱️  Toplam süre: {total_time:.1f}s")
    print(f"  📈 Steps: {trainer.global_step}")
    print(f"  🧠 Memory: {trainer.model.memory.get_stats()}")
    print(f"  🛡️  Constitution: {trainer.constitution.get_stats()}")
    print(f"  🔬 Sandbox: {trainer.sandbox.get_stats()}")
    print(f"  🔄 Replay buffer: {len(trainer.replay_buffer)} deneyim")
    
    # Tokenizer'ı kaydet (her zaman)
    tokenizer.save()
    print(f"  💾 Tokenizer kaydedildi: ./tokenizer_data/")
    
    # Final checkpoint
    final_path = Path(args.checkpoint_dir) / "nexus_final.pt"
    trainer.save_checkpoint(trainer.epoch)
    print(f"  💾 Checkpoint kaydedildi: {final_path}")
    
    # Hafızayı kaydet
    trainer.model.memory.save_all()
    print(f"  💾 Hafıza kaydedildi: ./memory_store/")
    
    # Quick generation test
    print(f"\n  ═══ Quick Generation Test ═══")
    trainer.model.eval()
    
    test_texts = ["Merhaba", "yapay zeka", "hesapla"]
    for test in test_texts:
        ids = tokenizer.encode(test)
        input_tensor = torch.tensor([ids[:args.seq_len]], dtype=torch.long)
        
        with torch.no_grad():
            output = trainer.model(input_tensor, use_memory=True, deterministic=True)
        
        # Greedy decode
        next_logits = output['language_logits'][0, -1, :]
        next_tokens = torch.topk(next_logits, 5).indices.tolist()
        id_map = getattr(tokenizer, 'id_to_token', None) or getattr(tokenizer, 'id_to_char', {})
        predicted = [id_map.get(t, '?') for t in next_tokens]
        
        action_idx = output['action_idx'][0].item()
        action_conf = output['confidence'][0].item()
        
        print(f"  '{test}' → next_top5={predicted} | "
              f"action={action_idx} conf={action_conf:.3f}")
    
    print(f"\n  🧠 Proto-AGI eğitimi tamamlandı. chat.py ile konuşabilirsin.")
    
    return trainer, tokenizer, pipeline


if __name__ == "__main__":
    args = parse_args()
    
    try:
        trainer, tokenizer, pipeline = run_training(args)
    except KeyboardInterrupt:
        print("\n  ⚠️ Eğitim kullanıcı tarafından durduruldu.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  ❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
