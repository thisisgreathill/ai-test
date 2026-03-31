"""
trainer.py — Continual Learning Eğitim Döngüsü (Proto-AGI)

Ana eğitim döngüsü:
  forward → dual loss → constitution check → sandbox → reward → 
  meta-cognition → memory update → backprop

Bileşenler:
  - TrainingConfig:        Tüm hyperparameters
  - ExperienceReplayBuffer: Catastrophic forgetting'e karşı replay
  - ContinualTrainer:       Ana eğitim yöneticisi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time
import json
import random

from nexus_core import NexusCore, NexusConfig
from memory import MemoryConfig, Episode
from meta_cognition import MetaCognitionLoop, MetaCognitionConfig
from constitution import ConstitutionGuard, ConstitutionConfig
from reward_engine import DualHeadLoss, RewardConfig, RewardSignal
from sandbox import SandboxEnvironment, OutcomeStatus
from action_space import ActionType


# ───────────────────────────── Config ─────────────────────────────

@dataclass
class TrainingConfig:
    """Eğitim konfigürasyonu."""
    # Model
    nexus_config: NexusConfig = None
    
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    
    # Training loop
    num_epochs: int = 100
    batch_size: int = 16
    seq_len: int = 128
    
    # Fast mode — sandbox/constitution/memory'yi her step yerine
    # heavy_step_interval step'te bir çalıştırır (5-10x hızlanma)
    fast_mode: bool = False
    heavy_step_interval: int = 10   # Her N step'te bir tam pipeline
    
    # Replay buffer
    replay_capacity: int = 10_000
    replay_batch_size: int = 32
    replay_min_size: int = 100    # Bu kadar dolmadan replay yapma
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = 10      # Her N epoch'ta checkpoint
    
    # Logging
    log_dir: str = "./logs/training"
    log_every: int = 5              # Her N step'te log
    
    # Reward
    reward_config: RewardConfig = None
    
    # Constitution
    constitution_config: ConstitutionConfig = None
    
    # Meta-cognition
    meta_config: MetaCognitionConfig = None
    
    def __post_init__(self):
        if self.nexus_config is None:
            self.nexus_config = NexusConfig()
        if self.reward_config is None:
            self.reward_config = RewardConfig()
        if self.constitution_config is None:
            self.constitution_config = ConstitutionConfig()
        if self.meta_config is None:
            self.meta_config = MetaCognitionConfig(
                feature_dim=self.nexus_config.d_model
            )


# ───────────────────────────── Experience Replay Buffer ───────────

@dataclass
class Experience:
    """Tek bir eğitim deneyimi."""
    input_ids: torch.Tensor         # (seq_len,)
    target_ids: torch.Tensor        # (seq_len,)
    action_idx: int
    reward: float
    confidence: float
    success: bool


class ExperienceReplayBuffer:
    """
    Catastrophic forgetting'e karşı deneyim tekrar tamponu.
    
    Model yeni şeyler öğrenirken eski bilgileri unutmasını önler.
    Rastgele geçmiş deneyimleri yeniden oynatarak stabilite sağlar.
    """
    
    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position: int = 0
    
    def push(self, experience: Experience):
        """Yeni deneyim ekle."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Rastgele batch örnekle."""
        n = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, n)
    
    def __len__(self) -> int:
        return len(self.buffer)


# ───────────────────────────── Continual Trainer ──────────────────

class ContinualTrainer:
    """
    Ana Eğitim Yöneticisi — Continual Learning.
    
    Her step:
      1. Forward pass (NexusCore)
      2. Constitution check (action'ı denetle)  [fast_mode'da aralıklı]
      3. Sandbox test (izole çalıştırma)        [fast_mode'da aralıklı]
      4. Reward hesaplama
      5. Meta-cognition (öz-yansıtma)
      6. Memory güncelleme                      [fast_mode'da aralıklı]
      7. Loss hesaplama + Backpropagation
      8. Experience replay (eski deneyimleri tekrarla) [fast_mode'da aralıklı]
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        if config is None:
            config = TrainingConfig()
        self.config = config
        
        # ─── Modüller ───
        self.model = NexusCore(config.nexus_config)
        self.meta_cognition = MetaCognitionLoop(config.meta_config)
        self.constitution = ConstitutionGuard(config.constitution_config)
        self.sandbox = SandboxEnvironment()
        self.loss_fn = DualHeadLoss(config.reward_config)
        self.replay_buffer = ExperienceReplayBuffer(config.replay_capacity)
        
        # ─── Optimizer ───
        # Model + meta-cognition parametrelerini birleştir
        all_params = list(self.model.parameters()) + list(self.meta_cognition.parameters())
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas
        )
        
        # ─── Scheduler ───
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # ─── Tracking ───
        self.global_step: int = 0
        self.epoch: int = 0
        self.training_log: List[Dict[str, Any]] = []
        
        # Dizinleri oluştur
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def _is_heavy_step(self) -> bool:
        """Bu step'te tam pipeline (constitution/sandbox/memory) çalışsın mı?"""
        if not self.config.fast_mode:
            return True  # fast_mode kapalıysa her step tam
        return self.global_step % self.config.heavy_step_interval == 0
    
    def train_step(
        self, 
        input_ids: torch.Tensor, 
        target_ids: torch.Tensor
    ) -> Dict[str, float]:
        """
        Tek bir eğitim adımı — tüm pipeline.
        
        input_ids:  (batch, seq_len) — input token'lar
        target_ids: (batch, seq_len) — target token'lar (language head için)
        
        return: loss ve metrik değerleri
        """
        self.model.train()
        self.meta_cognition.train()
        
        batch_size = input_ids.shape[0]
        do_heavy = self._is_heavy_step()
        
        # ═══════ 1. FORWARD PASS ═══════
        output = self.model(input_ids, use_memory=do_heavy, deterministic=False)
        
        language_logits = output['language_logits']     # (B, S, V)
        action_probs = output['action_probs']           # (B, num_actions)
        action_idx = output['action_idx']               # (B,)
        action_params = output['action_params']         # (B, param_dim)
        confidence = output['confidence']               # (B, 1)
        pooled = output['pooled_features']              # (B, d_model)
        
        # Action log-probabilities (REINFORCE için)
        action_log_probs = torch.log(
            action_probs.gather(1, action_idx.unsqueeze(1)) + 1e-8
        ).squeeze(1)  # (B,)
        
        # ═══════ 2. CONSTITUTION + SANDBOX (heavy step only) ═══════
        rewards = torch.zeros(batch_size)
        successes = torch.zeros(batch_size)
        
        if do_heavy:
            for i in range(batch_size):
                a_idx = action_idx[i].item()
                conf = confidence[i].item()
                
                guard_result = self.constitution.check(a_idx, conf)
                
                if not guard_result['allowed']:
                    rewards[i] = -1.0
                    successes[i] = 0.0
                    continue
                
                action_type = ActionType(min(a_idx, len(ActionType) - 1))
                
                if guard_result['requires_sandbox'] or guard_result['requires_escalation']:
                    sandbox_result = self.sandbox.execute(
                        action_type=action_type,
                        action_params=action_params[i],
                        confidence=conf
                    )
                    rewards[i] = sandbox_result['reward']
                    successes[i] = 1.0 if sandbox_result['outcome'].status == OutcomeStatus.SUCCESS else 0.0
                else:
                    rewards[i] = 0.5
                    successes[i] = 1.0
        else:
            # Light step: default rewards
            rewards.fill_(0.0)
            successes.fill_(0.5)
        
        # ═══════ 4. META-COGNITION ═══════
        meta_confidence = self.meta_cognition.evaluate_confidence(
            action_probs, pooled
        )
        
        # ═══════ 5. LOSS HESAPLAMA ═══════
        losses = self.loss_fn.compute_total_loss(
            language_logits=language_logits,
            language_targets=target_ids,
            action_log_probs=action_log_probs,
            action_probs=action_probs,
            rewards=rewards,
            confidence=confidence,
            actual_success=successes,
        )
        
        total_loss = losses['total_loss']
        
        # ═══════ 6. BACKPROPAGATION ═══════
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        
        self.optimizer.step()
        
        # ═══════ 7. MEMORY UPDATE (heavy step only) ═══════
        if do_heavy:
            for i in range(batch_size):
                episode = Episode(
                    context=pooled[i].detach(),
                    action=action_idx[i].item(),
                    reward=rewards[i].item(),
                    outcome="success" if successes[i] > 0.5 else "failure",
                    confidence=confidence[i].item()
                )
                self.model.memory.record_episode(episode)
        
        # ═══════ 8. EXPERIENCE REPLAY (heavy step only) ═══════
        replay_loss = 0.0
        if do_heavy:
            # Mevcut deneyimi buffer'a ekle
            for i in range(batch_size):
                exp = Experience(
                    input_ids=input_ids[i].detach(),
                    target_ids=target_ids[i].detach(),
                    action_idx=action_idx[i].item(),
                    reward=rewards[i].item(),
                    confidence=confidence[i].item(),
                    success=successes[i].item() > 0.5
                )
                self.replay_buffer.push(exp)
            
            # Replay (yeterli deneyim varsa)
            if len(self.replay_buffer) >= self.config.replay_min_size:
                replay_loss = self._replay_step()
        
        # ═══════ TRACKING ═══════
        self.global_step += 1
        
        metrics = {
            'step': self.global_step,
            'total_loss': total_loss.item(),
            'language_loss': losses['language_loss'].item(),
            'action_loss': losses['action_loss'].item(),
            'entropy_bonus': losses['entropy_bonus'].item(),
            'confidence_loss': losses['confidence_loss'].item(),
            'avg_reward': rewards.mean().item(),
            'avg_confidence': confidence.mean().item(),
            'success_rate': successes.mean().item(),
            'replay_loss': replay_loss,
        }
        
        # Logging
        if self.global_step % self.config.log_every == 0:
            self._log_metrics(metrics)
        
        return metrics
    
    def _replay_step(self) -> float:
        """Experience replay — geçmiş deneyimleri tekrarla."""
        experiences = self.replay_buffer.sample(self.config.replay_batch_size)
        
        if not experiences:
            return 0.0
        
        # Batch oluştur
        input_batch = torch.stack([e.input_ids for e in experiences])
        target_batch = torch.stack([e.target_ids for e in experiences])
        
        # Forward pass (replay)
        with torch.no_grad():
            self.model.eval()
        
        self.model.train()
        output = self.model(input_batch, use_memory=False)
        
        # Sadece language loss ile replay (action'lar farklı context'te)
        replay_loss = self.loss_fn.language_loss(
            output['language_logits'], target_batch
        )
        
        # Replay loss'u ana loss'a ekle (ağırlıklı)
        scaled_loss = 0.1 * replay_loss  # Replay weight
        scaled_loss.backward()
        
        return replay_loss.item()
    
    def train_epoch(
        self, 
        data_generator,
        epoch_num: int
    ) -> Dict[str, float]:
        """
        Bir epoch'luk eğitim.
        
        data_generator: (input_ids, target_ids) döndüren iterable
        """
        self.epoch = epoch_num
        epoch_metrics: List[Dict[str, float]] = []
        
        for batch_input, batch_target in data_generator:
            metrics = self.train_step(batch_input, batch_target)
            epoch_metrics.append(metrics)
        
        # Scheduler step
        self.scheduler.step()
        
        # Epoch ortalamaları
        avg_metrics = {}
        if epoch_metrics:
            for key in epoch_metrics[0]:
                if key != 'step':
                    values = [m[key] for m in epoch_metrics]
                    avg_metrics[key] = sum(values) / len(values)
        
        avg_metrics['epoch'] = epoch_num
        avg_metrics['learning_rate'] = self.scheduler.get_last_lr()[0]
        
        # Checkpointing
        if epoch_num % self.config.checkpoint_every == 0:
            self.save_checkpoint(epoch_num)
        
        return avg_metrics
    
    def train(self, data_generator_fn):
        """
        Tam eğitim döngüsü.
        
        data_generator_fn: Her çağrıda yeni epoch batch'i döndüren fonksiyon
        """
        print(f"\n{'='*60}")
        print(f"  NexusCore Proto-AGI — Continual Learning Training")
        print(f"{'='*60}")
        print(f"  Model:     {self.model.get_model_info()['total_params_M']} parameters")
        print(f"  Epochs:    {self.config.num_epochs}")
        print(f"  Batch:     {self.config.batch_size}")
        print(f"  LR:        {self.config.learning_rate}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            data_gen = data_generator_fn()
            metrics = self.train_epoch(data_gen, epoch)
            
            epoch_time = time.time() - epoch_start
            
            print(
                f"  Epoch {epoch:4d} │ "
                f"Loss: {metrics.get('total_loss', 0):.4f} │ "
                f"Lang: {metrics.get('language_loss', 0):.4f} │ "
                f"Act: {metrics.get('action_loss', 0):.4f} │ "
                f"Reward: {metrics.get('avg_reward', 0):+.3f} │ "
                f"Conf: {metrics.get('avg_confidence', 0):.3f} │ "
                f"Succ: {metrics.get('success_rate', 0):.1%} │ "
                f"LR: {metrics.get('learning_rate', 0):.2e} │ "
                f"{epoch_time:.1f}s"
            )
        
        print(f"\n{'='*60}")
        print(f"  Training complete! {self.global_step} steps.")
        print(f"{'='*60}\n")
        
        # Final checkpoint
        self.save_checkpoint(self.config.num_epochs)
        self.model.memory.save_all()
    
    def save_checkpoint(self, epoch: int):
        """Checkpoint kaydet."""
        path = Path(self.config.checkpoint_dir) / f"nexus_epoch_{epoch:04d}.pt"
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'meta_cognition_state_dict': self.meta_cognition.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Checkpoint yükle."""
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_cognition.load_state_dict(checkpoint['meta_cognition_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Metrikleri log dosyasına yaz."""
        self.training_log.append(metrics)
        
        log_path = Path(self.config.log_dir) / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
