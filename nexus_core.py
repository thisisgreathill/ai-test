"""
nexus_core.py — Dual-Head Otonom Melez Model (Proto-AGI)

Projenin kalbi. Saf PyTorch nn.Module sınıfları:
  - TransformerBackbone: Shared feature extractor (mini Transformer encoder)
  - LanguageHead:        Vocabulary projection — diyalog üretimi
  - NexusCore:           Üst modül — Backbone + LanguageHead + HierarchicalPolicy + Memory
  
Bu bir LLM değil. Bu, kendi kurallarıyla evrilen bir zeka çekirdeği.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from memory import MemoryConfig, MemoryController
from action_space import HierarchicalPolicy


# ───────────────────────────── Config ─────────────────────────────

@dataclass
class NexusConfig:
    """NexusCore model konfigürasyonu."""
    # Vocabulary & Embedding
    vocab_size: int = 32_000        # Token vocabulary boyutu
    max_seq_len: int = 512          # Maximum sequence uzunluğu
    
    # Transformer Backbone
    d_model: int = 256              # Model dimension
    n_heads: int = 8                # Attention head sayısı
    n_layers: int = 4               # Transformer layer sayısı
    d_ff: int = 1024                # Feed-forward hidden dim
    dropout: float = 0.1
    
    # Action Space
    num_strategies: int = 4         # Üst düzey strateji sayısı
    num_actions: int = 8            # Action tipi sayısı
    action_param_dim: int = 16      # Action parametre vektör boyutu
    
    # Memory
    memory: MemoryConfig = None
    
    def __post_init__(self):
        if self.memory is None:
            self.memory = MemoryConfig(
                wm_slot_dim=self.d_model,
                sm_key_dim=self.d_model,
                sm_value_dim=self.d_model
            )


# ───────────────────────────── Positional Encoding ────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Sinüzoidal pozisyonel kodlama (Vaswani et al. 2017)."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ───────────────────────────── Transformer Backbone ───────────────

class TransformerBackbone(nn.Module):
    """
    Shared feature extractor — Mini Transformer Encoder.
    
    Tüm head'ler bu gövdeden çıkan representation'ları kullanır.
    Memory injection: Working memory, attention mekanizmasına gated olarak eklenir.
    """
    
    def __init__(self, config: NexusConfig):
        super().__init__()
        self.config = config
        
        # Token embedding + positional encoding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm (daha stabil eğitim)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.n_layers,
            norm=nn.LayerNorm(config.d_model)
        )
        
        # Memory injection gate — backbone output'a memory bilgisi karıştırır
        self.memory_gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid()
        )
        
        # Pooling projection (sequence → single vector for action head)
        self.pool_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU()
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        input_ids:    (batch, seq_len) — token ID'leri
        memory_state: (batch, d_model) — working memory'den okunan bilgi
        
        return: {
            'sequence_features': (batch, seq_len, d_model),  — tüm token features
            'pooled_features':   (batch, d_model),            — aggregate feature
        }
        """
        # Embedding
        x = self.token_embedding(input_ids)  # (batch, seq, d_model)
        x = self.pos_encoding(x)
        
        # Transformer encoding
        if attention_mask is not None:
            # Causal mask veya padding mask
            x = self.encoder(x, src_key_padding_mask=attention_mask)
        else:
            x = self.encoder(x)
        
        # Memory injection (varsa)
        if memory_state is not None:
            # memory_state: (batch, d_model) → (batch, 1, d_model) broadcast
            mem_expanded = memory_state.unsqueeze(1).expand_as(x)
            gate_input = torch.cat([x, mem_expanded], dim=-1)
            gate = self.memory_gate(gate_input)
            x = gate * mem_expanded + (1 - gate) * x
        
        # Pooled representation (mean pooling + projection)
        pooled = x.mean(dim=1)  # (batch, d_model)
        pooled = self.pool_proj(pooled)
        
        return {
            'sequence_features': x,
            'pooled_features': pooled
        }


# ───────────────────────────── Language Head ──────────────────────

class LanguageHead(nn.Module):
    """
    Dil Katmanı — Sequence features → vocabulary distribution.
    
    Kullanıcıyla veya sistemlerle derinlemesine diyalog kurar.
    Token bazında next-token prediction (autoregressive generation desteği).
    """
    
    def __init__(self, config: NexusConfig):
        super().__init__()
        
        self.pre_norm = nn.LayerNorm(config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.d_model, config.vocab_size)
        
        # Weight tying: output weights = embedding weights (opsiyonel)
        # Bu, parametre sayısını azaltır ve genelleştirmeyi artırır
    
    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """
        sequence_features: (batch, seq_len, d_model)
        return: (batch, seq_len, vocab_size) — log-probability distribution
        """
        x = self.pre_norm(sequence_features)
        x = self.proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.output(x)  # (batch, seq_len, vocab_size)
        return logits
    
    def tie_weights(self, embedding_layer: nn.Embedding):
        """Embedding weights ile output weights'i bağla."""
        self.output.weight = embedding_layer.weight


# ───────────────────────────── NexusCore (Ana Modül) ──────────────

class NexusCore(nn.Module):
    """
    Ana Zeka Çekirdeği — Dual-Head Otonom Melez Model.
    
    Backbone (shared Transformer) + LanguageHead + HierarchicalPolicy + Memory.
    
    Forward pass:
      1. Working memory'den ilgili bilgiyi oku
      2. Backbone'da input'u + memory'yi işle
      3. Language Head: dil çıktısı üret
      4. Action Head (HierarchicalPolicy): strateji → taktik → parametrik eylem
      5. Working memory'ye yeni bilgiyi yaz
    """
    
    def __init__(self, config: Optional[NexusConfig] = None):
        super().__init__()
        
        if config is None:
            config = NexusConfig()
        self.config = config
        
        # ─── Shared Backbone ───
        self.backbone = TransformerBackbone(config)
        
        # ─── Language Head (Dil Katmanı) ───
        self.language_head = LanguageHead(config)
        
        # ─── Action Head (Eylem/Yönlendirme Katmanı) ───
        self.action_head = HierarchicalPolicy(
            input_dim=config.d_model,
            num_strategies=config.num_strategies,
            num_actions=config.num_actions,
            param_dim=config.action_param_dim
        )
        
        # ─── Memory System ───
        self.memory = MemoryController(config.memory)
        
        # ─── Confidence Estimation (entropy-based) ───
        self.confidence_proj = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Parametre sayısını raporla
        self._report_params()
    
    def _report_params(self):
        """Toplam parametre sayısını hesapla."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._total_params = total
        self._trainable_params = trainable
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_memory: bool = True,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Ana forward pass — tüm head'lerden çıktı üretir.
        
        Args:
            input_ids:      (batch, seq_len) — token ID'leri
            attention_mask: (batch, seq_len) — padding mask
            use_memory:     Working memory kullanılsın mı?
            deterministic:  Stochastic / deterministic action seçimi
        
        Returns: {
            'language_logits':   (batch, seq_len, vocab_size),
            'strategy_logits':   (batch, num_strategies),
            'strategy_probs':    (batch, num_strategies),
            'strategy_idx':      (batch,),
            'action_logits':     (batch, num_actions),
            'action_probs':      (batch, num_actions),
            'action_idx':        (batch,),
            'action_params':     (batch, param_dim),
            'confidence':        (batch, 1),
            'pooled_features':   (batch, d_model),
        }
        """
        batch_size = input_ids.shape[0]
        
        # ─── 1. Memory Read ───
        memory_state = None
        if use_memory:
            # Basit bir query oluştur (ilk token'ın embedding'i)
            with torch.no_grad():
                query_embed = self.backbone.token_embedding(
                    input_ids[:, 0]
                )  # (batch, d_model)
            memory_state = self.memory.read_working(query_embed)
        
        # ─── 2. Backbone Forward ───
        backbone_out = self.backbone(
            input_ids, 
            memory_state=memory_state,
            attention_mask=attention_mask
        )
        seq_features = backbone_out['sequence_features']   # (batch, seq, d_model)
        pooled = backbone_out['pooled_features']            # (batch, d_model)
        
        # ─── 3. Language Head ───
        language_logits = self.language_head(seq_features)
        # (batch, seq_len, vocab_size)
        
        # ─── 4. Action Head (Hierarchical) ───
        action_out = self.action_head(pooled, deterministic=deterministic)
        # Dict with strategy/action logits, probs, indices, params
        
        # ─── 5. Confidence Estimation ───
        confidence = self.confidence_proj(pooled)  # (batch, 1)
        
        # ─── 6. Memory Write ───
        if use_memory:
            # Pooled feature'ı working memory'ye yaz
            for i in range(batch_size):
                self.memory.write_working(
                    pooled[i].detach(), 
                    relevance=confidence[i].item()
                )
            # Working memory decay
            self.memory.working.decay(factor=0.95)
        
        # ─── Çıktı ───
        output = {
            'language_logits': language_logits,
            'confidence': confidence,
            'pooled_features': pooled,
        }
        output.update(action_out)
        
        return output
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür."""
        return {
            'name': 'NexusCore Proto-AGI',
            'version': '0.1.0',
            'total_params': self._total_params,
            'trainable_params': self._trainable_params,
            'total_params_M': f"{self._total_params / 1e6:.2f}M",
            'config': {
                'vocab_size': self.config.vocab_size,
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'n_layers': self.config.n_layers,
                'd_ff': self.config.d_ff,
                'num_strategies': self.config.num_strategies,
                'num_actions': self.config.num_actions,
            },
            'memory_stats': self.memory.get_stats()
        }
    
    def save_checkpoint(self, path: str):
        """Model ve memory state'i kaydet."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, path)
        self.memory.save_all()
    
    def load_checkpoint(self, path: str):
        """Model ve memory state'i yükle."""
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.memory.load_all()
