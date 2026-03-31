"""
meta_cognition.py — Self-Reflection & Confidence Estimation (Proto-AGI)

Öz-yansıtma katmanı:
  - SelfReflector:       'Bu kararım neden başarılı/başarısız oldu?' analizi
  - ConfidenceEstimator: Entropy-based karar güvenilirliği
  - MetaCognitionLoop:   Her karar sonrası analiz → memory → strateji güncelleme
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ───────────────────────────── Config ─────────────────────────────

@dataclass
class MetaCognitionConfig:
    """Meta-biliş konfigürasyonu."""
    feature_dim: int = 256          # Input feature boyutu
    hidden_dim: int = 128           # Hidden layer boyutu
    critique_dim: int = 64          # Critique embedding boyutu
    confidence_threshold: float = 0.3   # Bu altında → sandbox zorunlu
    escalation_threshold: float = 0.15  # Bu altında → human-in-the-loop


# ───────────────────────────── Self-Reflector ─────────────────────

class SelfReflector(nn.Module):
    """
    Bir action'ın sonucunu analiz eden mini ağ.
    
    Input:  action context + expected outcome + actual outcome
    Output: critique score (0-1) + critique embedding
    
    'Bu kararımı neden verdim ve sonuç beklentimle ne kadar uyuştu?'
    """
    
    def __init__(self, config: MetaCognitionConfig):
        super().__init__()
        self.config = config
        
        # Input: action_embedding + expected + actual = 3 * feature_dim
        input_dim = config.feature_dim * 3
        
        self.analysis_net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        
        # Critique score (0 = tamamen yanlış, 1 = mükemmel karar)
        self.score_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Critique embedding (memory'ye kaydedilecek özet)
        self.embedding_head = nn.Linear(config.hidden_dim, config.critique_dim)
    
    def forward(
        self,
        action_context: torch.Tensor,
        expected_outcome: torch.Tensor,
        actual_outcome: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        action_context:   (batch, feature_dim)
        expected_outcome: (batch, feature_dim)
        actual_outcome:   (batch, feature_dim)
        
        return: {
            'critique_score':     (batch, 1),
            'critique_embedding': (batch, critique_dim)
        }
        """
        combined = torch.cat([
            action_context, expected_outcome, actual_outcome
        ], dim=-1)
        
        hidden = self.analysis_net(combined)
        score = self.score_head(hidden)
        embedding = self.embedding_head(hidden)
        
        return {
            'critique_score': score,
            'critique_embedding': embedding
        }


# ───────────────────────────── Confidence Estimator ───────────────

class ConfidenceEstimator(nn.Module):
    """
    Action distribution'ın entropy'sine bakarak güven skoru üretir.
    
    Düşük entropy = yüksek güven (model emin)
    Yüksek entropy = düşük güven (model kararsız)
    
    Düşük güven durumunda:
      - confidence < confidence_threshold → sandbox zorunlu
      - confidence < escalation_threshold → human escalation
    """
    
    def __init__(self, config: MetaCognitionConfig):
        super().__init__()
        self.config = config
        
        # Entropy + features → calibrated confidence
        self.calibrator = nn.Sequential(
            nn.Linear(config.feature_dim + 1, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Action probability distribution'ın entropy'si."""
        # Numerical stability
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        return entropy
    
    def forward(
        self, 
        action_probs: torch.Tensor, 
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        action_probs: (batch, num_actions) — action olasılık dağılımı
        features:     (batch, feature_dim) — backbone features
        
        return: {
            'confidence':   (batch, 1),
            'entropy':      (batch, 1),
            'needs_sandbox': (batch,) bool,
            'needs_human':   (batch,) bool,
        }
        """
        entropy = self.compute_entropy(action_probs)  # (batch, 1)
        
        # Calibrated confidence
        calib_input = torch.cat([features, entropy], dim=-1)
        confidence = self.calibrator(calib_input)  # (batch, 1)
        
        # Threshold kontrolleri
        needs_sandbox = (confidence.squeeze(-1) < self.config.confidence_threshold)
        needs_human = (confidence.squeeze(-1) < self.config.escalation_threshold)
        
        return {
            'confidence': confidence,
            'entropy': entropy,
            'needs_sandbox': needs_sandbox,
            'needs_human': needs_human
        }


# ───────────────────────────── Outcome Predictor ──────────────────

class OutcomePredictor(nn.Module):
    """
    Bir action'ın beklenen sonucunu tahmin eder.
    Bu, SelfReflector'a 'expected_outcome' sağlar.
    Model zamanla hangi action'ların neye yol açtığını öğrenir.
    """
    
    def __init__(self, config: MetaCognitionConfig):
        super().__init__()
        
        # Input: pooled features + action embedding
        self.predictor = nn.Sequential(
            nn.Linear(config.feature_dim + config.feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.feature_dim)
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        action_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        features:         (batch, feature_dim) — mevcut state
        action_embedding: (batch, feature_dim) — seçilen action'ın embedding'i
        return:           (batch, feature_dim) — tahmin edilen outcome
        """
        combined = torch.cat([features, action_embedding], dim=-1)
        predicted_outcome = self.predictor(combined)
        return predicted_outcome


# ───────────────────────────── Meta-Cognition Loop ────────────────

class MetaCognitionLoop(nn.Module):
    """
    Her karar sonrasında çalışan öz-değerlendirme döngüsü.
    
    Akış:
      1. OutcomePredictor: 'Ne olmasını bekliyordum?'
      2. (Sandbox/gerçek ortam sonucu gelir)
      3. SelfReflector: beklenti vs gerçek → critique
      4. Critique → episodic memory'ye kaydet
      5. Düşük skor → strateji değiştir
    """
    
    def __init__(self, config: Optional[MetaCognitionConfig] = None):
        super().__init__()
        
        if config is None:
            config = MetaCognitionConfig()
        self.config = config
        
        self.reflector = SelfReflector(config)
        self.confidence_estimator = ConfidenceEstimator(config)
        self.outcome_predictor = OutcomePredictor(config)
        
        # Action embedding projection (action params → feature space)
        self.action_proj = nn.Linear(16, config.feature_dim)  # param_dim=16
        
        # Strategy adjustment signal
        self.strategy_adjustment = nn.Sequential(
            nn.Linear(config.critique_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 4),  # num_strategies
            nn.Softmax(dim=-1)
        )
    
    def predict_outcome(
        self, 
        features: torch.Tensor, 
        action_params: torch.Tensor
    ) -> torch.Tensor:
        """Action sonucunun beklenen outcome'unu tahmin et."""
        action_emb = self.action_proj(action_params)
        return self.outcome_predictor(features, action_emb)
    
    def reflect(
        self,
        features: torch.Tensor,
        action_params: torch.Tensor,
        actual_outcome: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Karar sonrası öz-değerlendirme.
        
        features:       (batch, feature_dim) — karar anındaki state
        action_params:  (batch, param_dim)   — alınan action parametreleri
        actual_outcome: (batch, feature_dim) — gerçek sonuç embedding'i
        
        return: critique score, embedding, strateji ayarlama sinyali
        """
        # Beklenen sonuç
        action_emb = self.action_proj(action_params)
        expected = self.outcome_predictor(features, action_emb)
        
        # Öz-yansıtma
        reflection = self.reflector(features, expected, actual_outcome)
        
        # Strateji ayarlama sinyali
        strategy_adj = self.strategy_adjustment(
            reflection['critique_embedding']
        )
        
        return {
            'critique_score': reflection['critique_score'],
            'critique_embedding': reflection['critique_embedding'],
            'expected_outcome': expected,
            'strategy_adjustment': strategy_adj
        }
    
    def evaluate_confidence(
        self, 
        action_probs: torch.Tensor, 
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Karar güvenilirliğini değerlendir."""
        return self.confidence_estimator(action_probs, features)
