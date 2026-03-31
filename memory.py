"""
memory.py — 3-Tier Memory Architecture (Proto-AGI)

Bu bir LLM context window'u DEĞİL.
3 katmanlı, kalıcı, sorgulanabilir hafıza sistemi:
  - WorkingMemory:  Prefrontal korteks — anlık attention buffer
  - EpisodicMemory: Hipokampüs — deneyim kaydı (disk-backed)
  - SemanticMemory: Neokorteks — konsept store (embedding retrieval)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path


# ───────────────────────────── Config ─────────────────────────────

@dataclass
class MemoryConfig:
    """Hafıza sistemi konfigürasyonu."""
    # Working Memory
    wm_num_slots: int = 64          # Anlık bellekteki slot sayısı
    wm_slot_dim: int = 256          # Her slot'un embedding boyutu
    
    # Episodic Memory
    em_capacity: int = 10_000       # Max deneyim sayısı
    em_priority_alpha: float = 0.6  # Priority sampling exponent
    em_save_dir: str = "./memory_store/episodic"
    
    # Semantic Memory
    sm_capacity: int = 50_000       # Max konsept sayısı
    sm_key_dim: int = 256           # Key embedding boyutu
    sm_value_dim: int = 256         # Value embedding boyutu
    sm_save_dir: str = "./memory_store/semantic"
    
    # Consolidation
    consolidation_threshold: int = 100  # Her N episode'da bir consolidation
    consolidation_top_k: int = 20       # En önemli N deneyimi semantic'e taşı


# ───────────────────────────── Working Memory ─────────────────────

class WorkingMemory(nn.Module):
    """
    Prefrontal Korteks — Sabit kapasiteli, attention-gated anlık bellek.
    
    LLM'deki context window'dan farklı olarak:
      - Sabit sayıda slot (overwrite ile güncelleme)
      - Relevance scoring ile en eski/en az ilgili slot'u override eder
      - Backbone'a gated attention ile enjekte edilir
    """
    
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.num_slots = config.wm_num_slots
        self.slot_dim = config.wm_slot_dim
        
        # Bellek slotları (learnable değil, dynamic buffer)
        self.register_buffer(
            'slots', torch.zeros(self.num_slots, self.slot_dim)
        )
        # Her slot'un relevance skoru
        self.register_buffer(
            'relevance', torch.zeros(self.num_slots)
        )
        # Slot doluluk sayacı
        self.register_buffer(
            'write_cursor', torch.tensor(0, dtype=torch.long)
        )
        
        # Gating mekanizması: backbone feature + memory → gated output
        self.gate_proj = nn.Linear(self.slot_dim * 2, self.slot_dim)
        self.gate_sigmoid = nn.Sigmoid()
        
        # Query projection (backbone hidden → memory query)
        self.query_proj = nn.Linear(self.slot_dim, self.slot_dim)
        
    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Attention-based okuma. 
        query: (batch, dim) → backbone'dan gelen hidden state
        return: (batch, dim) → memory'den okunan bilgi
        """
        # Query projection
        q = self.query_proj(query)  # (batch, dim)
        
        # Clone slots to avoid in-place autograd issues
        current_slots = self.slots.clone()
        
        # Attention scores: query vs tüm slotlar
        # slots: (num_slots, dim), q: (batch, dim)
        scores = torch.matmul(q, current_slots.T)  # (batch, num_slots)
        scores = scores / (self.slot_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_slots)
        
        # Weighted sum of slots
        retrieved = torch.matmul(attn_weights, current_slots)  # (batch, dim)
        
        # Gated fusion: query + retrieved → gated output
        combined = torch.cat([query, retrieved], dim=-1)  # (batch, 2*dim)
        gate = self.gate_sigmoid(self.gate_proj(combined))  # (batch, dim)
        output = gate * retrieved + (1 - gate) * query     # (batch, dim)
        
        return output
    
    def write(self, content: torch.Tensor, relevance_score: float = 1.0):
        """
        Belleğe yazma. En düşük relevance'lı slot'u overwrite eder.
        content: (dim,) → yazılacak embedding
        """
        content = content.detach()
        
        if self.write_cursor < self.num_slots:
            # Henüz dolmamış, sırayla yaz
            idx = self.write_cursor.item()
            new_slots = self.slots.clone()
            new_relevance = self.relevance.clone()
            new_slots[idx] = content
            new_relevance[idx] = relevance_score
            self.slots.copy_(new_slots)
            self.relevance.copy_(new_relevance)
            self.write_cursor.fill_(self.write_cursor.item() + 1)
        else:
            # Dolu, en düşük relevance'lı slot'u bul ve overwrite et
            min_idx = torch.argmin(self.relevance).item()
            new_slots = self.slots.clone()
            new_relevance = self.relevance.clone()
            new_slots[min_idx] = content
            new_relevance[min_idx] = relevance_score
            self.slots.copy_(new_slots)
            self.relevance.copy_(new_relevance)
    
    def decay(self, factor: float = 0.95):
        """Relevance decay — zamanla eski bilgiler önemini kaybeder."""
        self.relevance.copy_(self.relevance * factor)
    
    def clear(self):
        """Bellekteki tüm slotları sıfırla."""
        self.slots.zero_()
        self.relevance.zero_()
        self.write_cursor.zero_()


# ───────────────────────────── Episodic Memory ────────────────────

@dataclass
class Episode:
    """Tek bir deneyim kaydı."""
    context: torch.Tensor       # Karar anındaki state embedding
    action: int                 # Alınan action
    reward: float               # Kazanılan ödül
    outcome: str                # "success" | "failure" | "partial"
    confidence: float = 0.0     # Model'in karar anındaki güven skoru
    critique: str = ""          # Meta-cognition critique
    priority: float = 0.0       # Priority score (|reward| based)
    timestamp: int = 0          # Sıra numarası


class EpisodicMemory:
    """
    Hipokampüs — Deneyim arşivi. Disk-backed, priority-sampled.
    
    Her karar döngüsünün (context, action, reward, outcome) kaydını tutar.
    Yüksek reward veya yüksek penalty deneyimler öncelikli saklanır.
    """
    
    def __init__(self, config: MemoryConfig):
        self.capacity = config.em_capacity
        self.priority_alpha = config.em_priority_alpha
        self.save_dir = Path(config.em_save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.episodes: List[Episode] = []
        self.counter: int = 0
        
        # Disk'ten yükle (varsa)
        self._load_from_disk()
    
    def store(self, episode: Episode):
        """Yeni deneyim kaydet."""
        episode.timestamp = self.counter
        episode.priority = abs(episode.reward) + 0.01  # Minimum priority
        self.counter += 1
        
        if len(self.episodes) < self.capacity:
            self.episodes.append(episode)
        else:
            # Kapasiteyi aştık → en düşük priority'li deneyimi sil
            min_idx = min(range(len(self.episodes)), 
                        key=lambda i: self.episodes[i].priority)
            if episode.priority > self.episodes[min_idx].priority:
                self.episodes[min_idx] = episode
    
    def sample(self, batch_size: int) -> List[Episode]:
        """Priority-weighted sampling."""
        if len(self.episodes) == 0:
            return []
        
        priorities = torch.tensor(
            [e.priority ** self.priority_alpha for e in self.episodes]
        )
        probs = priorities / priorities.sum()
        
        n = min(batch_size, len(self.episodes))
        indices = torch.multinomial(probs, n, replacement=False).tolist()
        return [self.episodes[i] for i in indices]
    
    def get_recent(self, n: int = 10) -> List[Episode]:
        """Son N deneyimi getir."""
        return sorted(self.episodes, key=lambda e: e.timestamp, reverse=True)[:n]
    
    def get_high_reward(self, n: int = 10) -> List[Episode]:
        """En yüksek reward'lı N deneyimi getir."""
        return sorted(self.episodes, key=lambda e: e.reward, reverse=True)[:n]
    
    def save_to_disk(self):
        """Deneyimleri diske kaydet."""
        data = []
        for ep in self.episodes:
            data.append({
                'context': ep.context.cpu().tolist(),
                'action': ep.action,
                'reward': ep.reward,
                'outcome': ep.outcome,
                'confidence': ep.confidence,
                'critique': ep.critique,
                'priority': ep.priority,
                'timestamp': ep.timestamp
            })
        
        save_path = self.save_dir / "episodes.json"
        with open(save_path, 'w') as f:
            json.dump(data, f)
    
    def _load_from_disk(self):
        """Disk'ten deneyimleri yükle."""
        save_path = self.save_dir / "episodes.json"
        if not save_path.exists():
            return
        
        with open(save_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            ctx_tensor = torch.tensor(item['context'])
            # Boyut uyumsuzluğu kontrolü — eski/farklı model'den kalan verileri atla
            ep = Episode(
                context=ctx_tensor,
                action=item['action'],
                reward=item['reward'],
                outcome=item['outcome'],
                confidence=item.get('confidence', 0.0),
                critique=item.get('critique', ''),
                priority=item.get('priority', 0.0),
                timestamp=item.get('timestamp', 0)
            )
            self.episodes.append(ep)
        
        self.counter = max((e.timestamp for e in self.episodes), default=0) + 1

    def __len__(self) -> int:
        return len(self.episodes)


# ───────────────────────────── Semantic Memory ────────────────────

class SemanticMemory(nn.Module):
    """
    Neokorteks — Genel bilgi ve öğrenilmiş konseptler.
    
    Key-value embedding store. Cosine similarity ile retrieval.
    Episodic memory'den consolidation ile beslenir.
    """
    
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.capacity = config.sm_capacity
        self.key_dim = config.sm_key_dim
        self.value_dim = config.sm_value_dim
        self.save_dir = Path(config.sm_save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Key-value store (dynamic buffers)
        self.register_buffer(
            'keys', torch.zeros(self.capacity, self.key_dim)
        )
        self.register_buffer(
            'values', torch.zeros(self.capacity, self.value_dim)
        )
        self.register_buffer(
            'usage', torch.zeros(self.capacity)  # Erişim sıklığı
        )
        self.register_buffer(
            'write_idx', torch.tensor(0, dtype=torch.long)
        )
        
        # Query transformation
        self.query_transform = nn.Linear(self.key_dim, self.key_dim)
    
    def store(self, key: torch.Tensor, value: torch.Tensor):
        """
        Yeni konsept kaydet.
        key:   (key_dim,)
        value: (value_dim,)
        """
        key = key.detach()
        value = value.detach()
        
        # Dimension mismatch koruması — farklı d_model ile eğitilmiş episodları atla
        if key.shape[-1] != self.key_dim:
            return  # Boyut uyumsuz, sessizce atla
        if value.shape[-1] != self.value_dim:
            return
        
        idx = self.write_idx.item()
        
        if idx < self.capacity:
            self.keys[idx] = key
            self.values[idx] = value
            self.usage[idx] = 1.0
            self.write_idx += 1
        else:
            # Kapasiteyi aştık → en az kullanılanı overwrite et
            min_idx = torch.argmin(self.usage).item()
            self.keys[min_idx] = key
            self.values[min_idx] = value
            self.usage[min_idx] = 1.0
    
    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cosine similarity ile en yakın konseptleri getir.
        query: (batch, key_dim)
        return: (values: (batch, top_k, value_dim), scores: (batch, top_k))
        """
        n_stored = min(self.write_idx.item(), self.capacity)
        if n_stored == 0:
            batch = query.shape[0]
            return (torch.zeros(batch, top_k, self.value_dim, device=query.device),
                    torch.zeros(batch, top_k, device=query.device))
        
        q = self.query_transform(query)  # (batch, key_dim)
        
        # Cosine similarity
        stored_keys = self.keys[:n_stored]  # (n_stored, key_dim)
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(stored_keys, dim=-1)
        
        sim = torch.matmul(q_norm, k_norm.T)  # (batch, n_stored)
        
        actual_k = min(top_k, n_stored)
        scores, indices = torch.topk(sim, actual_k, dim=-1)  # (batch, actual_k)
        
        # Gather values
        stored_values = self.values[:n_stored]
        retrieved_values = stored_values[indices]  # (batch, actual_k, value_dim)
        
        # Usage counter güncelle
        for idx_batch in indices:
            for idx in idx_batch:
                self.usage[idx] += 1.0
        
        # Pad if needed
        if actual_k < top_k:
            batch = query.shape[0]
            pad_vals = torch.zeros(batch, top_k - actual_k, self.value_dim, device=query.device)
            pad_scores = torch.zeros(batch, top_k - actual_k, device=query.device)
            retrieved_values = torch.cat([retrieved_values, pad_vals], dim=1)
            scores = torch.cat([scores, pad_scores], dim=1)
        
        return retrieved_values, scores
    
    def save_to_disk(self):
        """Semantic memory'yi diske kaydet."""
        save_path = self.save_dir / "semantic_store.pt"
        torch.save({
            'keys': self.keys.cpu(),
            'values': self.values.cpu(),
            'usage': self.usage.cpu(),
            'write_idx': self.write_idx.cpu()
        }, save_path)
    
    def load_from_disk(self):
        """Disk'ten yükle."""
        save_path = self.save_dir / "semantic_store.pt"
        if not save_path.exists():
            return
        data = torch.load(save_path, weights_only=True)
        
        # Dimension mismatch koruması
        stored_keys = data['keys']
        if stored_keys.shape[-1] != self.key_dim or stored_keys.shape[0] != self.capacity:
            # Eski boyutlarla uyumsuz → temiz başla
            return
        
        self.keys.copy_(stored_keys)
        self.values.copy_(data['values'])
        self.usage.copy_(data['usage'])
        self.write_idx.copy_(data['write_idx'])


# ───────────────────────────── Memory Controller ──────────────────

class MemoryController(nn.Module):
    """
    3 katman arasındaki okuma/yazma/consolidation akışını yönetir.
    
    Her forward pass:
      1. Working memory'den oku (attention-gated)
      2. Working memory'ye yaz (yeni bilgi)
    
    Her episode sonu:
      3. Episodic memory'ye yaz (deneyim kaydı)
    
    Periyodik:
      4. Episodic → Semantic consolidation (uyku benzeri transfer)
    """
    
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        
        self.working = WorkingMemory(config)
        self.episodic = EpisodicMemory(config)
        self.semantic = SemanticMemory(config)
        
        # Consolidation counter
        self._episode_count = 0
        
        # Semantic retrieval → working memory injection projection
        self.semantic_proj = nn.Linear(config.sm_value_dim, config.wm_slot_dim)
    
    def read_working(self, query: torch.Tensor) -> torch.Tensor:
        """Working memory'den attention-based okuma."""
        return self.working.read(query)
    
    def write_working(self, content: torch.Tensor, relevance: float = 1.0):
        """Working memory'ye yeni bilgi yaz."""
        self.working.write(content, relevance)
    
    def record_episode(self, episode: Episode):
        """
        Bir karar döngüsünü episodic memory'ye kaydet.
        Gerekirse consolidation tetikle.
        """
        self.episodic.store(episode)
        self._episode_count += 1
        
        # Periyodik consolidation
        if self._episode_count % self.config.consolidation_threshold == 0:
            self.consolidate()
    
    def query_semantic(self, query: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """
        Semantic memory'den ilgili konseptleri getir.
        return: (batch, top_k, value_dim)
        """
        values, scores = self.semantic.retrieve(query, top_k)
        return values
    
    def consolidate(self):
        """
        Uyku benzeri süreç: Episodic → Semantic transfer.
        En önemli deneyimlerin context embedding'lerini semantic memory'ye taşır.
        """
        top_episodes = self.episodic.get_high_reward(
            self.config.consolidation_top_k
        )
        
        for ep in top_episodes:
            key = ep.context
            # Value = context + reward bilgisi encoded
            value = ep.context.clone()
            self.semantic.store(key, value)
    
    def inject_semantic_to_working(self, query: torch.Tensor):
        """
        Semantic'ten ilgili bilgiyi çekip working memory'ye enjekte et.
        Uzun süreli bilgiyi kısa süreli belleğe taşır.
        """
        values, scores = self.semantic.retrieve(query.unsqueeze(0), top_k=3)
        # En yüksek score'lu value'yu working memory'ye yaz
        if scores.sum() > 0:
            best_value = values[0, 0]  # (value_dim,)
            projected = self.semantic_proj(best_value)
            self.working.write(projected, relevance=scores[0, 0].item())
    
    def save_all(self):
        """Tüm kalıcı hafızaları diske kaydet."""
        self.episodic.save_to_disk()
        self.semantic.save_to_disk()
    
    def load_all(self):
        """Tüm kalıcı hafızaları disk'ten yükle."""
        self.semantic.load_from_disk()
    
    def get_stats(self) -> Dict[str, Any]:
        """Hafıza istatistikleri."""
        return {
            'working_memory_cursor': self.working.write_cursor.item(),
            'working_memory_capacity': self.working.num_slots,
            'episodic_memory_size': len(self.episodic),
            'episodic_memory_capacity': self.episodic.capacity,
            'semantic_memory_size': self.semantic.write_idx.item(),
            'semantic_memory_capacity': self.semantic.capacity,
            'total_episodes_recorded': self._episode_count,
        }
