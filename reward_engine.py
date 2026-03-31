"""
reward_engine.py — RL Reward & Custom Loss (Proto-AGI)

Ödül/Ceza mekanizması ve özel kayıp fonksiyonları:
  - DualHeadLoss:       Language (CE) + Action (REINFORCE) birleşik loss
  - RewardSignal:       Sandbox sonucuna göre +reward / -penalty
  - RewardAccumulator:  Episodik return hesabı (discount factor)
  - IntrinsicCuriosity: Bilinmeyen durumlar için exploration bonusu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ───────────────────────────── Config ─────────────────────────────

@dataclass
class RewardConfig:
    """Reward sistemi konfigürasyonu."""
    # Loss ağırlıkları
    alpha_language: float = 0.5     # Language head loss ağırlığı
    alpha_action: float = 0.3       # Action head loss ağırlığı
    alpha_curiosity: float = 0.1    # Curiosity bonus ağırlığı
    alpha_confidence: float = 0.1   # Confidence calibration loss ağırlığı
    
    # RL parametreleri
    gamma: float = 0.99             # Discount factor
    entropy_coeff: float = 0.01     # Entropy regularization (exploration teşviği)
    
    # Intrinsic Curiosity
    curiosity_hidden_dim: int = 128
    curiosity_feature_dim: int = 256
    
    # Reward clipping
    reward_clip: float = 10.0       # Max |reward| değeri


# ───────────────────────────── Reward Signal ──────────────────────

@dataclass
class RewardSignal:
    """Bir action'ın sandbox/gerçek ortam sonucu."""
    reward: float               # + ödül, - ceza
    success: bool               # Başarılı mı?
    details: str = ""           # Neden başarılı/başarısız
    intrinsic_bonus: float = 0.0  # Curiosity bonusu
    
    @property
    def total(self) -> float:
        return self.reward + self.intrinsic_bonus


# ───────────────────────────── Reward Accumulator ─────────────────

class RewardAccumulator:
    """
    Episodik reward tracking ve discounted return hesabı.
    
    R_t = r_t + γ * r_{t+1} + γ² * r_{t+2} + ...
    """
    
    def __init__(self, gamma: float = 0.99, clip: float = 10.0):
        self.gamma = gamma
        self.clip = clip
        self.episode_rewards: List[float] = []
        self.episode_log_probs: List[torch.Tensor] = []
    
    def add(self, reward: float, log_prob: torch.Tensor):
        """Bir step'in reward ve log-prob'unu kaydet."""
        clipped = max(-self.clip, min(self.clip, reward))
        self.episode_rewards.append(clipped)
        self.episode_log_probs.append(log_prob)
    
    def compute_returns(self) -> torch.Tensor:
        """
        Discounted returns hesapla.
        return: (episode_length,) tensor
        """
        returns = []
        R = 0.0
        
        for r in reversed(self.episode_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        
        # Normalize (variance reduction)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def compute_policy_loss(self) -> torch.Tensor:
        """
        REINFORCE policy gradient loss.
        L = -Σ log π(a|s) * R_t
        """
        returns = self.compute_returns()
        
        policy_loss = torch.tensor(0.0)
        for log_prob, R in zip(self.episode_log_probs, returns):
            policy_loss += -log_prob * R
        
        return policy_loss
    
    def reset(self):
        """Episode sonunda sıfırla."""
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
    
    @property
    def total_reward(self) -> float:
        return sum(self.episode_rewards)
    
    @property
    def episode_length(self) -> int:
        return len(self.episode_rewards)


# ───────────────────────────── Intrinsic Curiosity Module ─────────

class IntrinsicCuriosity(nn.Module):
    """
    Curiosity-Driven Exploration (Pathak et al. 2017 stilinde).
    
    Model bilinmeyen/beklenmedik durumlarla karşılaştığında bonus reward üretir.
    Bu, modeli keşfe teşvik eder ve sadece bildiği action'lara takılmasını önler.
    
    Mekanizma:
      - Forward Model:  (state, action) → predicted_next_state
      - Prediction Error = |predicted - actual|² → curiosity bonus
      - Yüksek hata = bilinmeyen durum = yüksek bonus
    """
    
    def __init__(self, config: RewardConfig):
        super().__init__()
        
        feature_dim = config.curiosity_feature_dim
        hidden_dim = config.curiosity_hidden_dim
        
        # Feature encoder (raw state → compact feature)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Forward dynamics model: (feature, action_onehot) → predicted next feature
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim),  # +8 for action one-hot
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Inverse model: (feature_t, feature_t+1) → predicted action
        # Bu, feature encoder'ın task-relevant features öğrenmesini sağlar
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 8)  # num_actions
        )
    
    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        num_actions: int = 8
    ) -> Dict[str, torch.Tensor]:
        """
        state:      (batch, feature_dim)
        next_state: (batch, feature_dim)
        action:     (batch,) — action indices
        
        return: {
            'curiosity_bonus': (batch,),     — exploration reward
            'forward_loss':    scalar,        — forward model loss
            'inverse_loss':    scalar,        — inverse model loss
        }
        """
        # Encode states
        feat = self.feature_encoder(state)           # (batch, hidden)
        feat_next = self.feature_encoder(next_state)  # (batch, hidden)
        
        # Action one-hot
        action_onehot = F.one_hot(
            action.long(), num_classes=num_actions
        ).float()  # (batch, num_actions)
        
        # Forward model: predict next state feature
        forward_input = torch.cat([feat, action_onehot], dim=-1)
        predicted_next = self.forward_model(forward_input)  # (batch, hidden)
        
        # Curiosity bonus = prediction error
        prediction_error = (predicted_next - feat_next.detach()).pow(2).mean(dim=-1)
        curiosity_bonus = prediction_error.detach()  # (batch,)
        
        # Forward loss (minimize prediction error over time)
        forward_loss = prediction_error.mean()
        
        # Inverse model: predict action from consecutive states
        inverse_input = torch.cat([feat, feat_next], dim=-1)
        predicted_action_logits = self.inverse_model(inverse_input)
        inverse_loss = F.cross_entropy(predicted_action_logits, action.long())
        
        return {
            'curiosity_bonus': curiosity_bonus,
            'forward_loss': forward_loss,
            'inverse_loss': inverse_loss
        }


# ───────────────────────────── Dual Head Loss ─────────────────────

class DualHeadLoss(nn.Module):
    """
    İki Başlıklı Birleşik Kayıp Fonksiyonu.
    
    Total Loss = α_lang * L_language 
               + α_action * L_action 
               + α_curiosity * L_curiosity
               + α_confidence * L_confidence
               + entropy_coeff * H(π)
    
    Bileşenler:
      - L_language:   CrossEntropy (next-token prediction)
      - L_action:     REINFORCE policy gradient
      - L_curiosity:  Forward + Inverse model loss
      - L_confidence: Confidence calibration (binary CE vs actual success)
      - H(π):         Entropy bonus (exploration teşviği)
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        super().__init__()
        
        if config is None:
            config = RewardConfig()
        self.config = config
        
        self.language_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.curiosity = IntrinsicCuriosity(config)
        self.accumulator = RewardAccumulator(config.gamma, config.reward_clip)
    
    def language_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Language head loss (CrossEntropy).
        logits:  (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        """
        # Reshape for CE: (batch*seq, vocab) vs (batch*seq,)
        B, S, V = logits.shape
        loss = self.language_loss_fn(
            logits.reshape(B * S, V),
            targets.reshape(B * S)
        )
        return loss
    
    def action_loss(
        self, 
        action_log_probs: torch.Tensor, 
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Action head loss (REINFORCE).
        action_log_probs: (batch,) — log π(a|s)
        rewards:          (batch,) — R_t (discounted returns)
        """
        # Normalize rewards
        if rewards.numel() > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Policy gradient: -log π(a|s) * R
        loss = -(action_log_probs * rewards).mean()
        return loss
    
    def entropy_bonus(self, action_probs: torch.Tensor) -> torch.Tensor:
        """
        Entropy regularization — distribution'ın uniform'a yakın olmasını teşvik eder.
        Bu, exploration'ı artırır.
        action_probs: (batch, num_actions)
        """
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
        return entropy.mean()
    
    def confidence_loss(
        self, 
        predicted_confidence: torch.Tensor, 
        actual_success: torch.Tensor
    ) -> torch.Tensor:
        """
        Confidence calibration loss.
        Model, kendi güvenini doğru tahmin etmeyi öğrenir.
        predicted_confidence: (batch, 1)
        actual_success:       (batch,) — 0 veya 1
        """
        return F.binary_cross_entropy(
            predicted_confidence.squeeze(-1),
            actual_success.float()
        )
    
    def compute_total_loss(
        self,
        language_logits: torch.Tensor,
        language_targets: torch.Tensor,
        action_log_probs: torch.Tensor,
        action_probs: torch.Tensor,
        rewards: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        actual_success: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        next_state: Optional[torch.Tensor] = None,
        action_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tüm loss bileşenlerini hesapla ve birleştir.
        
        return: {
            'total_loss':       scalar,
            'language_loss':    scalar,
            'action_loss':      scalar,
            'entropy_bonus':    scalar,
            'confidence_loss':  scalar,
            'curiosity_loss':   scalar,
        }
        """
        losses = {}
        
        # Language loss
        l_lang = self.language_loss(language_logits, language_targets)
        losses['language_loss'] = l_lang
        
        # Action loss (REINFORCE)
        l_action = self.action_loss(action_log_probs, rewards)
        losses['action_loss'] = l_action
        
        # Entropy bonus
        h_entropy = self.entropy_bonus(action_probs)
        losses['entropy_bonus'] = h_entropy
        
        # Confidence calibration loss
        l_confidence = torch.tensor(0.0)
        if confidence is not None and actual_success is not None:
            l_confidence = self.confidence_loss(confidence, actual_success)
        losses['confidence_loss'] = l_confidence
        
        # Curiosity loss
        l_curiosity = torch.tensor(0.0)
        curiosity_forward = torch.tensor(0.0)
        if state is not None and next_state is not None and action_indices is not None:
            curiosity_out = self.curiosity(state, next_state, action_indices)
            curiosity_forward = curiosity_out['forward_loss']
            l_curiosity = curiosity_out['forward_loss'] + curiosity_out['inverse_loss']
        losses['curiosity_loss'] = l_curiosity
        
        # Total loss
        total = (
            self.config.alpha_language * l_lang
            + self.config.alpha_action * l_action
            - self.config.entropy_coeff * h_entropy  # Negatif: maximize entropy
            + self.config.alpha_confidence * l_confidence
            + self.config.alpha_curiosity * l_curiosity
        )
        losses['total_loss'] = total
        
        return losses
