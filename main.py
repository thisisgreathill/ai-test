"""
main.py — Proto-AGI Entry Point

NexusCore'u ayağa kaldırır, dummy data ile forward pass ve
tek bir training step demo çalıştırır.

Kullanım:
    python main.py
"""

import torch
import sys
from pathlib import Path

from nexus_core import NexusCore, NexusConfig
from memory import MemoryConfig
from meta_cognition import MetaCognitionLoop, MetaCognitionConfig
from constitution import ConstitutionGuard, ConstitutionConfig
from reward_engine import DualHeadLoss, RewardConfig
from sandbox import SandboxEnvironment
from trainer import ContinualTrainer, TrainingConfig
from action_space import ActionType


def print_header():
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
    ║          Proto-AGI Dual-Head Autonomous Core              ║
    ║          "Bir LLM değil, bir zeka çekirdeği."             ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)


def demo_forward_pass():
    """Forward pass demo — modelin doğru çalıştığını gösterir."""
    print("\n" + "─" * 60)
    print("  DEMO 1: Forward Pass")
    print("─" * 60)
    
    # Config
    config = NexusConfig(
        vocab_size=1000,    # Küçük vocab (demo)
        max_seq_len=64,
        d_model=128,        # Küçük model (demo)
        n_heads=4,
        n_layers=2,
        d_ff=512,
        num_strategies=4,
        num_actions=8,
        action_param_dim=16,
    )
    
    # Model oluştur
    model = NexusCore(config)
    info = model.get_model_info()
    
    print(f"\n  Model: {info['name']} v{info['version']}")
    print(f"  Parameters: {info['total_params_M']}")
    print(f"  Config: d_model={config.d_model}, layers={config.n_layers}, heads={config.n_heads}")
    print(f"  Actions: {config.num_actions} types, {config.num_strategies} strategies")
    
    # Dummy input
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\n  Input shape:  {tuple(input_ids.shape)}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_ids, use_memory=True, deterministic=True)
    
    # Çıktıları göster
    print(f"\n  ┌─ Language Head ─────────────────────────────────────")
    print(f"  │ Output shape: {tuple(output['language_logits'].shape)}")
    print(f"  │ (batch={batch_size}, seq={seq_len}, vocab={config.vocab_size})")
    
    print(f"  │")
    print(f"  ├─ Action Head (Hierarchical) ───────────────────────")
    print(f"  │ Strategy probs: {output['strategy_probs'][0].numpy().round(3)}")
    print(f"  │ Strategy idx:   {output['strategy_idx'].numpy()}")
    print(f"  │ Action probs:   {output['action_probs'][0].numpy().round(3)}")
    print(f"  │ Action idx:     {output['action_idx'].numpy()}")
    print(f"  │ Action params:  shape={tuple(output['action_params'].shape)}")
    
    print(f"  │")
    print(f"  ├─ Confidence ───────────────────────────────────────")
    print(f"  │ Confidence:     {output['confidence'].squeeze().numpy().round(3)}")

    print(f"  │")
    print(f"  └─ Memory Stats ─────────────────────────────────────")
    mem_stats = model.memory.get_stats()
    for key, val in mem_stats.items():
        print(f"    {key}: {val}")
    
    print(f"\n  ✅ Forward pass successful!")
    return model, config


def demo_constitution():
    """Constitution guard demo."""
    print("\n" + "─" * 60)
    print("  DEMO 2: Constitution Guard")
    print("─" * 60)
    
    guard = ConstitutionGuard()
    
    # İzinli action
    result = guard.check(action_idx=0, confidence=0.8)
    print(f"\n  Action 0 (LOCAL_COMPUTE, conf=0.8): {'✅ İZİN' if result['allowed'] else '❌ ENGEL'}")
    
    # Düşük güvenli action
    result = guard.check(action_idx=1, confidence=0.1)
    print(f"  Action 1 (EXTERNAL_API, conf=0.1):  {'✅ İZİN' if result['allowed'] else '❌ ENGEL'}"
          f" | sandbox={result['requires_sandbox']} | escalation={result['requires_escalation']}")
    
    # Yasaklı action testi
    guard_strict = ConstitutionGuard(ConstitutionConfig(forbidden_actions=[3]))
    result = guard_strict.check(action_idx=3, confidence=0.9)
    print(f"  Action 3 (WEB_CRAWL, forbidden):    {'✅ İZİN' if result['allowed'] else '❌ ENGEL'}"
          f" | reason={result['reason']}")
    
    print(f"\n  Guard stats: {guard.get_stats()}")
    print(f"  ✅ Constitution guard working!")


def demo_sandbox():
    """Sandbox demo."""
    print("\n" + "─" * 60)
    print("  DEMO 3: Sandbox Environment")
    print("─" * 60)
    
    sandbox = SandboxEnvironment()
    
    actions = [
        ActionType.LOCAL_COMPUTE,
        ActionType.EXTERNAL_API,
        ActionType.WEB_CRAWL,
        ActionType.SELF_REFLECT,
    ]
    
    for action in actions:
        result = sandbox.execute(action, confidence=0.7)
        outcome = result['outcome']
        print(f"\n  {action.name:25s} │ "
              f"status={outcome.status.value:8s} │ "
              f"reward={result['reward']:+.3f} │ "
              f"{outcome.details}")
    
    stats = sandbox.get_stats()
    print(f"\n  Sandbox stats: {stats}")
    print(f"  ✅ Sandbox working!")


def demo_training_step():
    """Tek bir training step demo."""
    print("\n" + "─" * 60)
    print("  DEMO 4: Training Step")
    print("─" * 60)
    
    config = TrainingConfig(
        nexus_config=NexusConfig(
            vocab_size=1000,
            max_seq_len=64,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
        ),
        learning_rate=1e-3,
        batch_size=4,
        seq_len=32,
    )
    
    trainer = ContinualTrainer(config)
    
    # Dummy data
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    target_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"\n  Running single training step...")
    metrics = trainer.train_step(input_ids, target_ids)
    
    print(f"\n  ┌─ Training Metrics ────────────────────────────────")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  │ {key:20s}: {val:+.6f}")
        else:
            print(f"  │ {key:20s}: {val}")
    print(f"  └──────────────────────────────────────────────────")
    
    print(f"\n  Memory stats: {trainer.model.memory.get_stats()}")
    print(f"  Replay buffer: {len(trainer.replay_buffer)} experiences")
    print(f"  ✅ Training step successful!")


def demo_mini_training():
    """Kısa bir eğitim döngüsü demo."""
    print("\n" + "─" * 60)
    print("  DEMO 5: Mini Training Loop (3 epochs)")
    print("─" * 60)
    
    config = TrainingConfig(
        nexus_config=NexusConfig(
            vocab_size=500,
            max_seq_len=32,
            d_model=64,
            n_heads=2,
            n_layers=1,
            d_ff=256,
        ),
        num_epochs=3,
        batch_size=8,
        seq_len=16,
        learning_rate=1e-3,
        checkpoint_every=100,   # Bu demo'da checkpoint atma
        replay_min_size=50,
    )
    
    trainer = ContinualTrainer(config)
    
    def data_generator():
        """5 batch dummy data üret."""
        for _ in range(5):
            input_ids = torch.randint(0, 500, (8, 16))
            target_ids = torch.randint(0, 500, (8, 16))
            yield input_ids, target_ids
    
    trainer.train(data_generator)
    
    # Final stats
    print(f"  Memory: {trainer.model.memory.get_stats()}")
    print(f"  Sandbox: {trainer.sandbox.get_stats()}")
    print(f"  Constitution: {trainer.constitution.get_stats()}")
    print(f"  Replay buffer: {len(trainer.replay_buffer)} experiences")


# ───────────────────────────── Main ───────────────────────────────

if __name__ == "__main__":
    print_header()
    
    # Her demo'yu sırayla çalıştır
    try:
        model, config = demo_forward_pass()
        demo_constitution()
        demo_sandbox()
        demo_training_step()
        demo_mini_training()
        
        print("\n" + "═" * 60)
        print("  🧠 All systems operational. Proto-AGI seed is alive.")
        print("═" * 60 + "\n")
        
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
