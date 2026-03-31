"""
action_space.py — Parametric & Hierarchical Action Space (Proto-AGI)

Discrete action ID değil, parametrik ve hiyerarşik eylem uzayı:
  - ActionDefinition: Her action'ın meta bilgisi (risk, reversibility, params)
  - ParametricAction: Action type + parameter vector
  - HierarchicalPolicy: Strategy → Tactic iki seviyeli karar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import IntEnum


# ───────────────────────────── Action Definitions ─────────────────

class ActionType(IntEnum):
    """Temel action tipleri."""
    LOCAL_COMPUTE = 0       # Yerel hesaplama / reasoning
    EXTERNAL_API = 1        # Dış API çağrısı
    ISOLATED_SIMULATION = 2 # İzole sandbox simülasyonu
    WEB_CRAWL = 3           # Otonom web taraması
    MEMORY_QUERY = 4        # Hafıza sorgulama
    SELF_REFLECT = 5        # Meta-cognition tetikleme
    DELEGATE = 6            # Alt-göreve böl ve delege et
    WAIT = 7                # Bekle / daha fazla bilgi topla


@dataclass
class ActionDefinition:
    """Bir action tipinin meta bilgisi."""
    action_type: ActionType
    name: str
    risk_level: float           # 0.0 (güvenli) → 1.0 (yüksek risk)
    reversible: bool            # Geri alınabilir mi?
    requires_sandbox: bool      # Sandbox'ta test edilmeli mi?
    param_dim: int = 16         # Parameter vector boyutu
    description: str = ""
    
    
# Varsayılan action kataloğu
DEFAULT_ACTION_CATALOG: List[ActionDefinition] = [
    ActionDefinition(
        action_type=ActionType.LOCAL_COMPUTE,
        name="local_compute",
        risk_level=0.0,
        reversible=True,
        requires_sandbox=False,
        description="Yerel reasoning ve hesaplama"
    ),
    ActionDefinition(
        action_type=ActionType.EXTERNAL_API,
        name="external_api",
        risk_level=0.6,
        reversible=False,
        requires_sandbox=True,
        description="Dış API çağrısı (parametre gerektirir)"
    ),
    ActionDefinition(
        action_type=ActionType.ISOLATED_SIMULATION,
        name="isolated_sim",
        risk_level=0.1,
        reversible=True,
        requires_sandbox=True,
        description="İzole ortamda simülasyon"
    ),
    ActionDefinition(
        action_type=ActionType.WEB_CRAWL,
        name="web_crawl",
        risk_level=0.4,
        reversible=False,
        requires_sandbox=True,
        description="Otonom web taraması"
    ),
    ActionDefinition(
        action_type=ActionType.MEMORY_QUERY,
        name="memory_query",
        risk_level=0.0,
        reversible=True,
        requires_sandbox=False,
        description="Hafıza sistemini sorgula"
    ),
    ActionDefinition(
        action_type=ActionType.SELF_REFLECT,
        name="self_reflect",
        risk_level=0.0,
        reversible=True,
        requires_sandbox=False,
        description="Meta-cognition döngüsü tetikle"
    ),
    ActionDefinition(
        action_type=ActionType.DELEGATE,
        name="delegate",
        risk_level=0.3,
        reversible=True,
        requires_sandbox=False,
        description="Alt-göreve böl ve delege et"
    ),
    ActionDefinition(
        action_type=ActionType.WAIT,
        name="wait",
        risk_level=0.0,
        reversible=True,
        requires_sandbox=False,
        description="Bekle, daha fazla bilgi topla"
    ),
]


# ───────────────────────────── Parametric Action ──────────────────

@dataclass
class ParametricAction:
    """
    Sadece 'API çağır' değil — 'hangi API, hangi parametrelerle' bilgisini taşır.
    
    action_type: Hangi tip action
    params: Action'a özgü parametre vektörü
    confidence: Model'in bu karara güveni
    """
    action_type: ActionType
    params: torch.Tensor        # (param_dim,) — action'a özgü parametreler
    confidence: float = 0.0
    strategy_id: int = -1       # Hangi üst strateji tarafından üretildi


# ───────────────────────────── Strategy Head ──────────────────────

class StrategyHead(nn.Module):
    """
    Üst düzey strateji kararı.
    'Ne yapılmalı?' sorusunu cevaplar.
    
    Çıktı: Strateji distribution (K adet üst düzey strateji)
    """
    
    def __init__(self, input_dim: int, num_strategies: int = 4):
        super().__init__()
        self.num_strategies = num_strategies
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_strategies)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        features: (batch, input_dim)
        return: (strategy_logits: (batch, num_strategies), 
                 strategy_probs: (batch, num_strategies))
        """
        logits = self.net(features)  # (batch, num_strategies)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


# ───────────────────────────── Tactic Head ────────────────────────

class TacticHead(nn.Module):
    """
    Alt düzey taktik kararı.
    'Nasıl yapılmalı?' sorusunu cevaplar.
    
    Strateji embedding + backbone features → action type + params
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_strategies: int = 4,
        num_actions: int = 8,
        param_dim: int = 16
    ):
        super().__init__()
        self.num_actions = num_actions
        self.param_dim = param_dim
        
        # Strategy embedding
        self.strategy_embed = nn.Embedding(num_strategies, input_dim // 4)
        
        combined_dim = input_dim + input_dim // 4
        
        # Action type selector
        self.action_net = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, num_actions)
        )
        
        # Parameter generator
        self.param_net = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.GELU(),
            nn.Linear(combined_dim // 2, param_dim)
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        strategy_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        features: (batch, input_dim)
        strategy_idx: (batch,) — seçilen strateji indeksi
        return: (action_logits, action_probs, params)
        """
        strat_emb = self.strategy_embed(strategy_idx)  # (batch, input_dim//4)
        combined = torch.cat([features, strat_emb], dim=-1)  # (batch, combined_dim)
        
        action_logits = self.action_net(combined)       # (batch, num_actions)
        action_probs = F.softmax(action_logits, dim=-1)
        params = self.param_net(combined)               # (batch, param_dim)
        
        return action_logits, action_probs, params


# ───────────────────────────── Hierarchical Policy ────────────────

class HierarchicalPolicy(nn.Module):
    """
    İki seviyeli karar mekanizması:
      1. StrategyHead: 'Ne yapılmalı?' (üst düzey)
      2. TacticHead:  'Nasıl yapılmalı?' (alt düzey)
    
    Backbone features → Strategy → Tactic → ParametricAction
    """
    
    def __init__(
        self,
        input_dim: int,
        num_strategies: int = 4,
        num_actions: int = 8,
        param_dim: int = 16
    ):
        super().__init__()
        self.num_actions = num_actions
        self.param_dim = param_dim
        
        self.strategy_head = StrategyHead(input_dim, num_strategies)
        self.tactic_head = TacticHead(
            input_dim, num_strategies, num_actions, param_dim
        )
        
        self.action_catalog = DEFAULT_ACTION_CATALOG
    
    def forward(
        self, 
        features: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        features: (batch, input_dim)
        return: dict with strategy/action logits, probs, and params
        """
        # 1. Strateji seç
        strategy_logits, strategy_probs = self.strategy_head(features)
        
        if deterministic:
            strategy_idx = torch.argmax(strategy_probs, dim=-1)
        else:
            strategy_idx = torch.multinomial(strategy_probs, 1).squeeze(-1)
        
        # 2. Taktik seç (strateji bağlamında)
        action_logits, action_probs, params = self.tactic_head(
            features, strategy_idx
        )
        
        if deterministic:
            action_idx = torch.argmax(action_probs, dim=-1)
        else:
            action_idx = torch.multinomial(action_probs, 1).squeeze(-1)
        
        return {
            'strategy_logits': strategy_logits,
            'strategy_probs': strategy_probs,
            'strategy_idx': strategy_idx,
            'action_logits': action_logits,
            'action_probs': action_probs,
            'action_idx': action_idx,
            'action_params': params,
        }
    
    def create_parametric_action(
        self, 
        action_idx: int, 
        params: torch.Tensor, 
        confidence: float,
        strategy_idx: int
    ) -> ParametricAction:
        """Forward çıktısından ParametricAction oluştur."""
        action_type = ActionType(min(action_idx, len(ActionType) - 1))
        return ParametricAction(
            action_type=action_type,
            params=params.detach(),
            confidence=confidence,
            strategy_id=strategy_idx
        )
    
    def get_action_risk(self, action_idx: int) -> float:
        """Bir action'ın risk seviyesini döndür."""
        if action_idx < len(self.action_catalog):
            return self.action_catalog[action_idx].risk_level
        return 1.0  # Bilinmeyen action = max risk
    
    def get_action_definition(self, action_idx: int) -> Optional[ActionDefinition]:
        """Bir action'ın tanımını döndür."""
        if action_idx < len(self.action_catalog):
            return self.action_catalog[action_idx]
        return None
