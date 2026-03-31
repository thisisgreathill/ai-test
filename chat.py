"""
chat.py — Interactive Terminal Chat (Proto-AGI)

Eğitilmiş NexusCore modeli ile terminal üzerinden sohbet.

Kullanım:
    python chat.py                                  # Son checkpoint
    python chat.py --checkpoint checkpoints/nexus_epoch_0020.pt
    python chat.py --temperature 0.7 --top_k 30

Komutlar:
    /quit     → Çıkış
    /stats    → Model istatistikleri
    /memory   → Hafıza durumu
    /history  → Konuşma geçmişi
    /clear    → Geçmişi temizle
    /action   → Son action detayları
    /save     → Hafızayı kaydet
    /tools    → Tool executor istatistikleri
"""

import torch
import argparse
import sys
import os
from pathlib import Path

from nexus_core import NexusCore, NexusConfig
from tokenizer import BPETokenizer, CharTokenizer, TokenizerConfig
from inference import InteractiveSession, InferenceConfig, TextGenerator, ActionInterpreter
from constitution import ConstitutionGuard, ConstitutionConfig
from sandbox import SandboxEnvironment
from tool_executor import ToolRouter, ToolConfig
from action_space import ActionType


def parse_args():
    parser = argparse.ArgumentParser(description="Proto-AGI Chat")
    
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Model checkpoint dosyası (.pt)")
    parser.add_argument("--tokenizer_dir", type=str, default="./tokenizer_data",
                       help="Tokenizer dizini")
    parser.add_argument("--tokenizer_type", type=str, default="auto",
                       choices=["auto", "char", "bpe"],
                       help="Tokenizer tipi (auto = otomatik algıla)")
    
    # Generation
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    
    # Features
    parser.add_argument("--auto_execute", action="store_true",
                       help="Action'ları otomatik çalıştır")
    parser.add_argument("--use_tools", action="store_true",
                       help="Gerçek tool executor kullan")
    
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir="./checkpoints"):
    """En son checkpoint'u bul."""
    cp_dir = Path(checkpoint_dir)
    if not cp_dir.exists():
        return None
    
    checkpoints = sorted(cp_dir.glob("nexus_*.pt"))
    if not checkpoints:
        return None
    
    return str(checkpoints[-1])


def load_tokenizer(args):
    """Tokenizer'ı yükle veya oluştur."""
    import json
    tok_dir = Path(args.tokenizer_dir)
    tok_file = tok_dir / "tokenizer.json"
    
    if tok_file.exists() and args.tokenizer_type in ("auto", "char", "bpe"):
        # Dosyadan tipi algıla
        with open(tok_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tok_type = data.get('type', 'bpe')  # BPE'de type field yok, CharTokenizer'da var
        
        if tok_type == 'char':
            print("  📝 CharTokenizer yükleniyor (diskten)...")
            tokenizer = CharTokenizer()
            tokenizer.load(args.tokenizer_dir)
        else:
            print("  📝 BPE Tokenizer yükleniyor (diskten)...")
            tokenizer = BPETokenizer()
            tokenizer.load(args.tokenizer_dir)
        
        print(f"  ✅ Vocab: {tokenizer.vocab_size} tokens")
        return tokenizer
    else:
        print("  ⚠️ Kaydedilmiş tokenizer bulunamadı!")
        print("  📝 Fallback: Char Tokenizer oluşturuluyor...")
        tokenizer = CharTokenizer()
        tokenizer.fit([
            "Merhaba dünya! Ben Proto-AGI, bir zeka çekirdeğiyim.",
            "abcçdefgğhıijklmnoöprsştuüvyz ABCÇDEFGĞHIIJKLMNOÖPRSŞTUÜVYZ",
            "0123456789 .,!?;:'-\"()[]{}+*/@#$%&=<>_~`^\\|",
        ])
        print(f"  ✅ Vocab: {tokenizer.vocab_size} tokens")
        print("  💡 İpucu: Önce 'python train_real.py' çalıştır!")
        return tokenizer


def load_model(args, tokenizer):
    """Model'i yükle."""
    checkpoint_path = args.checkpoint
    
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
    
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"  🧠 Checkpoint yükleniyor: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        config = checkpoint.get('config')
        if config and hasattr(config, 'nexus_config'):
            nexus_config = config.nexus_config
        else:
            nexus_config = NexusConfig(
                vocab_size=tokenizer.vocab_size,
                max_seq_len=128,
                d_model=128, n_heads=4, n_layers=3, d_ff=512,
            )
        
        # vocab_size uyumsuzluğu kontrolü
        nexus_config.vocab_size = tokenizer.vocab_size
        
        model = NexusCore(nexus_config)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"  ✅ Model yüklendi (epoch {checkpoint.get('epoch', '?')}, "
                  f"step {checkpoint.get('global_step', '?')})")
        except Exception as e:
            print(f"  ⚠️ Checkpoint uyumsuz, yeni model oluşturuluyor: {e}")
            model = NexusCore(nexus_config)
    else:
        print("  ⚠️ Checkpoint bulunamadı, eğitimsiz model oluşturuluyor...")
        nexus_config = NexusConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=128,
            d_model=128, n_heads=4, n_layers=3, d_ff=512,
        )
        model = NexusCore(nexus_config)
        print("  💡 İpucu: Önce 'python train_real.py' ile modeli eğit!")
    
    info = model.get_model_info()
    print(f"  📊 {info['total_params_M']} parameters")
    
    # Hafızayı yükle
    try:
        model.memory.load_all()
        stats = model.memory.get_stats()
        if stats['episodic_memory_size'] > 0:
            print(f"  🧠 Hafıza yüklendi: {stats['episodic_memory_size']} episodic, "
                  f"{stats['semantic_memory_size']} semantic")
    except:
        pass
    
    return model


def run_chat(args):
    """Chat döngüsünü başlat."""
    
    print(r"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗          ║
    ║     ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝          ║
    ║     ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗          ║
    ║     ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║          ║
    ║     ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║          ║
    ║     ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚══════╝          ║
    ║                                                           ║
    ║            Interactive Chat Session                       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # ─── Yükleme ───
    print("  ═══ Sistem Başlatılıyor ═══\n")
    
    tokenizer = load_tokenizer(args)
    model = load_model(args, tokenizer)
    
    # Constitution
    constitution = ConstitutionGuard(ConstitutionConfig(
        max_actions_per_minute=30,
        max_api_calls_per_minute=5,
    ))
    
    # Tool Router
    tool_router = None
    if args.use_tools:
        tool_router = ToolRouter(ToolConfig())
        print("  🔧 Tool Executor: AKTİF")
    
    # Inference config
    inf_config = InferenceConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        auto_execute=args.auto_execute,
    )
    
    # Session
    session = InteractiveSession(
        model=model,
        tokenizer=tokenizer,
        config=inf_config,
        constitution=constitution,
    )
    
    print(f"\n  ═══ Hazır ═══")
    print(f"  🌡️ Temperature: {args.temperature} | Top-k: {args.top_k} | "
          f"Top-p: {args.top_p}")
    print(f"  📝 Max tokens: {args.max_tokens} | "
          f"Auto-execute: {'✅' if args.auto_execute else '❌'}")
    print(f"\n  Komutlar: /quit /stats /memory /history /clear /action /save /tools")
    print("═" * 60 + "\n")
    
    # ─── Chat Döngüsü ───
    last_action = None
    
    while True:
        try:
            user_input = input("  🧑 Sen > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Oturum sonlandırıldı. Hoşça kal! 👋")
            break
        
        if not user_input:
            continue
        
        # ─── Özel Komutlar ───
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            
            if cmd == "/quit":
                print("  Oturum sonlandırıldı. Hoşça kal! 👋")
                break
            
            elif cmd == "/stats":
                info = model.get_model_info()
                print(f"\n  ╔═══ Model İstatistikleri ═══")
                print(f"  ║ Parameters:  {info['total_params_M']}")
                print(f"  ║ Vocab:       {info['config']['vocab_size']}")
                print(f"  ║ d_model:     {info['config']['d_model']}")
                print(f"  ║ Layers:      {info['config']['n_layers']}")
                print(f"  ║ Heads:       {info['config']['n_heads']}")
                print(f"  ║ Actions:     {info['config']['num_actions']}")
                print(f"  ║ Strategies:  {info['config']['num_strategies']}")
                print(f"  ╚{'═' * 30}\n")
                continue
            
            elif cmd == "/memory":
                stats = model.memory.get_stats()
                print(f"\n  ╔═══ Hafıza Durumu ═══")
                for k, v in stats.items():
                    print(f"  ║ {k}: {v}")
                print(f"  ╚{'═' * 30}\n")
                continue
            
            elif cmd == "/history":
                msgs = session.conversation_history[-10:]
                print(f"\n  ╔═══ Son {len(msgs)} Mesaj ═══")
                for msg in msgs:
                    role = "🧑 Sen" if msg['role'] == 'user' else "🧠 AGI"
                    content = msg['content'][:60]
                    print(f"  ║ {role}: {content}...")
                print(f"  ╚{'═' * 30}\n")
                continue
            
            elif cmd == "/clear":
                session.conversation_history.clear()
                model.memory.working.clear()
                print("  🧹 Geçmiş ve working memory temizlendi.\n")
                continue
            
            elif cmd == "/action":
                if last_action:
                    print(f"\n  ╔═══ Son Action ═══")
                    for k, v in last_action.items():
                        print(f"  ║ {k}: {v}")
                    print(f"  ╚{'═' * 30}\n")
                else:
                    print("  Henüz action yok.\n")
                continue
            
            elif cmd == "/save":
                model.memory.save_all()
                print("  💾 Hafıza kaydedildi.\n")
                continue
            
            elif cmd == "/tools":
                if tool_router:
                    stats = tool_router.get_stats()
                    print(f"\n  ╔═══ Tool İstatistikleri ═══")
                    for k, v in stats.items():
                        print(f"  ║ {k}: {v}")
                    print(f"  ╚{'═' * 30}\n")
                else:
                    print("  Tool executor aktif değil. --use_tools ile başlat.\n")
                continue
            
            else:
                print(f"  Bilinmeyen komut: {cmd}")
                print(f"  Komutlar: /quit /stats /memory /history /clear /action /save /tools\n")
                continue
        
        # ─── Normal Yanıt ───
        result = session.respond(user_input)
        
        # Response
        response = result['response']
        if len(response) > 200:
            response = response[:200] + "..."
        
        print(f"\n  🧠 AGI > {response}")
        
        # Action bilgisi
        if result['action']:
            action = result['action']
            last_action = action
            status = "✅" if action['allowed'] else "❌"
            line = f"  [{action['action_name']} {status}"
            
            if action.get('requires_sandbox'):
                line += " 🔬sandbox"
            if action.get('requires_escalation'):
                line += " 🚨escalation"
            
            line += "]"
            print(line, end="")
            
            # Tool execution
            if action['executed'] and action['result']:
                print(f" → {action['result']['status']}: {action['result']['details']}")
            elif args.use_tools and action['allowed'] and tool_router:
                # ToolRouter ile gerçek execution
                action_type = ActionType[action['action_name']]
                tool_result = tool_router.execute(action_type)
                if tool_result.success:
                    print(f" → 🔧 {tool_result.output[:80]}")
                else:
                    print(f" → ⚠️ {tool_result.error[:80]}")
            else:
                print()
        
        # Meta bilgi
        print(f"  [Confidence: {result['confidence']:.3f} | "
              f"Tokens: {result['num_tokens']}]\n")


if __name__ == "__main__":
    args = parse_args()
    
    try:
        run_chat(args)
    except Exception as e:
        print(f"\n  ❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
