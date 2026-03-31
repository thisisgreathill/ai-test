"""
inference.py — Interactive Inference & Text Generation (Proto-AGI)

Modelle gerçek zamanlı etkileşim:
  - TextGenerator:       Autoregressive text üretimi (top-k, top-p, temperature)
  - ActionInterpreter:   Action head çıktısını yorumla ve çalıştır
  - InteractiveSession:  Tam etkileşim döngüsü (input → think → act → respond)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from nexus_core import NexusCore, NexusConfig
from memory import MemoryConfig
from tokenizer import BPETokenizer, CharTokenizer, TokenizerConfig
from constitution import ConstitutionGuard, ConstitutionConfig
from sandbox import SandboxEnvironment
from meta_cognition import MetaCognitionLoop, MetaCognitionConfig
from action_space import ActionType


# ───────────────────────────── Config ─────────────────────────────

@dataclass
class InferenceConfig:
    """Inference konfigürasyonu."""
    # Generation
    max_new_tokens: int = 128       # Üretilecek max token sayısı
    temperature: float = 0.8        # Sampling temperature (düşük=conservative)
    top_k: int = 50                 # Top-k sampling
    top_p: float = 0.9             # Nucleus (top-p) sampling
    repetition_penalty: float = 1.5 # Tekrar cezası
    ngram_block_size: int = 3        # N-gram blocking (0=off)
    
    # Action
    action_threshold: float = 0.3   # Bu üzerinde action execute et
    auto_execute: bool = False       # Action'ları otomatik çalıştır mı?
    
    # Device
    device: str = "cpu"


# ───────────────────────────── Text Generator ─────────────────────

class TextGenerator:
    """
    Autoregressive text üretimi.
    
    Sampling stratejileri:
      - Greedy:      En yüksek olasılıklı token
      - Top-k:       En yüksek k token arasından sample
      - Top-p:       Kümülatif olasılık p'ye ulaşana kadar token ekle
      - Temperature: Dağılımı keskinleştir veya yumuşat
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        if config is None:
            config = InferenceConfig()
        self.config = config
    
    def _apply_repetition_penalty(
        self, 
        logits: torch.Tensor, 
        generated_ids: List[int]
    ) -> torch.Tensor:
        """Frequency-scaled repetition penalty + n-gram blocking."""
        if not generated_ids:
            return logits
        
        # --- Frequency-scaled repetition penalty ---
        if self.config.repetition_penalty != 1.0:
            from collections import Counter
            freq = Counter(generated_ids)
            for token_id, count in freq.items():
                penalty = self.config.repetition_penalty * (1.0 + 0.2 * min(count, 5))
                if logits[token_id] > 0:
                    logits[token_id] /= penalty
                else:
                    logits[token_id] *= penalty
        
        # --- N-gram blocking ---
        n = self.config.ngram_block_size
        if n > 0 and len(generated_ids) >= n:
            prefix = tuple(generated_ids[-(n - 1):])
            for i in range(len(generated_ids) - n + 1):
                existing = tuple(generated_ids[i:i + n - 1])
                if existing == prefix:
                    blocked_id = generated_ids[i + n - 1]
                    logits[blocked_id] = float('-inf')
        
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """En yüksek k token dışındakileri -inf yap."""
        if k <= 0 or k >= logits.shape[-1]:
            return logits
        
        top_k_vals, _ = torch.topk(logits, k)
        threshold = top_k_vals[..., -1]
        logits[logits < threshold] = float('-inf')
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Kümülatif olasılık p'yi aşan token'ları filtrele."""
        if p >= 1.0:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # p'yi aşan indeksleri bul
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
        sorted_logits[sorted_mask] = float('-inf')
        
        # Orijinal sıraya geri dön
        logits.scatter_(-1, sorted_indices, sorted_logits)
        return logits
    
    def sample_token(
        self, 
        logits: torch.Tensor, 
        generated_ids: List[int]
    ) -> int:
        """
        Tek bir token sample et.
        logits: (vocab_size,)
        """
        # Repetition penalty
        logits = self._apply_repetition_penalty(logits.clone(), generated_ids)
        
        # Temperature
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature
        
        # Top-k filtering
        logits = self._top_k_filtering(logits, self.config.top_k)
        
        # Top-p filtering
        logits = self._top_p_filtering(logits, self.config.top_p)
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, 1).item()
        
        return token_id
    
    @torch.no_grad()
    def generate(
        self, 
        model: NexusCore,
        input_ids: torch.Tensor,
        tokenizer,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Autoregressive text üretimi.
        
        input_ids: (1, seq_len) — başlangıç token'ları
        
        return: {
            'generated_ids': List[int],
            'generated_text': str,
            'action_outputs': List[Dict],
            'confidence_scores': List[float],
        }
        """
        model.eval()
        max_new = max_new_tokens or self.config.max_new_tokens
        
        generated_ids = input_ids[0].tolist()
        action_outputs = []
        confidence_scores = []
        
        current_input = input_ids
        
        for step in range(max_new):
            # Sequence'ı truncate et (model max_seq_len'i aşmasın)
            if current_input.shape[1] > model.config.max_seq_len:
                current_input = current_input[:, -model.config.max_seq_len:]
            
            # Forward pass
            output = model(current_input, use_memory=True, deterministic=False)
            
            # Son token'ın logits'i
            next_logits = output['language_logits'][0, -1, :]  # (vocab_size,)
            
            # Token sample et
            next_token = self.sample_token(next_logits, generated_ids)
            generated_ids.append(next_token)
            
            # Action bilgisi kaydet
            action_outputs.append({
                'step': step,
                'action_idx': output['action_idx'][0].item(),
                'action_probs': output['action_probs'][0].tolist(),
                'strategy_idx': output['strategy_idx'][0].item(),
            })
            confidence_scores.append(output['confidence'][0].item())
            
            # EOS check
            if hasattr(tokenizer, 'eos_id') and next_token == tokenizer.eos_id:
                break
            
            # Yeni token'ı input'a ekle
            next_tensor = torch.tensor([[next_token]], dtype=torch.long)
            current_input = torch.cat([current_input, next_tensor], dim=1)
        
        # Decode
        generated_text = tokenizer.decode(generated_ids)
        
        return {
            'generated_ids': generated_ids,
            'generated_text': generated_text,
            'action_outputs': action_outputs,
            'confidence_scores': confidence_scores,
            'num_tokens': len(generated_ids) - input_ids.shape[1],
        }


# ───────────────────────────── Action Interpreter ─────────────────

class ActionInterpreter:
    """
    Action head çıktısını yorumlayıp tool_executor'a yönlendirir.
    
    Her step'te:
      1. Action probs'a bak
      2. En yüksek action'ı constitution'a sor
      3. Onay varsa → sandbox veya doğrudan çalıştır
      4. Sonucu rapora ekle
    """
    
    def __init__(
        self,
        constitution: Optional[ConstitutionGuard] = None,
        sandbox: Optional[SandboxEnvironment] = None,
    ):
        self.constitution = constitution or ConstitutionGuard()
        self.sandbox = sandbox or SandboxEnvironment()
    
    def interpret(
        self, 
        action_output: Dict,
        confidence: float,
        auto_execute: bool = False
    ) -> Dict[str, Any]:
        """
        Bir action çıktısını yorumla.
        
        return: {
            'action_type': ActionType,
            'allowed': bool,
            'executed': bool,
            'result': Optional[Dict],
            'reason': str,
        }
        """
        action_idx = action_output['action_idx']
        action_type = ActionType(min(action_idx, len(ActionType) - 1))
        
        # Constitution check
        guard_result = self.constitution.check(action_idx, confidence)
        
        result = {
            'action_type': action_type,
            'action_name': action_type.name,
            'allowed': guard_result['allowed'],
            'executed': False,
            'result': None,
            'reason': guard_result['reason'],
            'requires_sandbox': guard_result['requires_sandbox'],
            'requires_escalation': guard_result['requires_escalation'],
        }
        
        # Auto-execute (izin verilmişse)
        if auto_execute and guard_result['allowed']:
            sandbox_result = self.sandbox.execute(
                action_type=action_type,
                confidence=confidence
            )
            result['executed'] = True
            result['result'] = {
                'status': sandbox_result['outcome'].status.value,
                'reward': sandbox_result['reward'],
                'details': sandbox_result['outcome'].details,
            }
        
        return result


# ───────────────────────────── Interactive Session ────────────────

class InteractiveSession:
    """
    Tam etkileşimli oturum yöneticisi.
    
    Akış:
      User Input → Tokenize → Forward Pass → 
        Language Head → Text Response
        Action Head  → Action Interpretation
      → Display → User Input ...
    """
    
    def __init__(
        self,
        model: NexusCore,
        tokenizer,
        config: Optional[InferenceConfig] = None,
        constitution: Optional[ConstitutionGuard] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()
        
        self.generator = TextGenerator(self.config)
        self.interpreter = ActionInterpreter(constitution)
        self.meta = MetaCognitionLoop(
            MetaCognitionConfig(feature_dim=model.config.d_model)
        )
        
        self.conversation_history: List[Dict[str, str]] = []
    
    def respond(self, user_input: str) -> Dict[str, Any]:
        """
        Kullanıcı girdisine yanıt üret.
        
        return: {
            'response': str,
            'action': Dict,
            'confidence': float,
            'memory_stats': Dict,
        }
        """
        # Tokenize
        input_ids = self.tokenizer.encode(user_input)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        # Generate
        gen_result = self.generator.generate(
            self.model, input_tensor, self.tokenizer
        )
        
        # Action interpretation (son step'in action'ı)
        action_result = None
        if gen_result['action_outputs']:
            last_action = gen_result['action_outputs'][-1]
            avg_confidence = (
                sum(gen_result['confidence_scores']) 
                / len(gen_result['confidence_scores'])
            )
            action_result = self.interpreter.interpret(
                last_action, avg_confidence, self.config.auto_execute
            )
        
        # Conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_input,
        })
        self.conversation_history.append({
            'role': 'assistant',
            'content': gen_result['generated_text'],
        })
        
        return {
            'response': gen_result['generated_text'],
            'num_tokens': gen_result['num_tokens'],
            'action': action_result,
            'confidence': (
                sum(gen_result['confidence_scores']) 
                / max(1, len(gen_result['confidence_scores']))
            ),
            'memory_stats': self.model.memory.get_stats(),
        }
    
    def run_interactive(self):
        """
        Terminal'de interaktif chat döngüsü başlat.
        """
        print("\n" + "═" * 60)
        print("  🧠 NexusCore Interactive Session")
        print("  Komutlar: /quit, /stats, /memory, /history, /clear")
        print("═" * 60 + "\n")
        
        while True:
            try:
                user_input = input("  Sen > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Oturum sonlandırıldı.")
                break
            
            if not user_input:
                continue
            
            # Özel komutlar
            if user_input == "/quit":
                print("  Oturum sonlandırıldı.")
                break
            
            elif user_input == "/stats":
                info = self.model.get_model_info()
                print(f"\n  Model: {info['total_params_M']}")
                print(f"  Memory: {info['memory_stats']}")
                print()
                continue
            
            elif user_input == "/memory":
                stats = self.model.memory.get_stats()
                print(f"\n  Memory Stats:")
                for k, v in stats.items():
                    print(f"    {k}: {v}")
                print()
                continue
            
            elif user_input == "/history":
                print(f"\n  Conversation ({len(self.conversation_history)} messages):")
                for msg in self.conversation_history[-10:]:
                    role = "Sen" if msg['role'] == 'user' else "AGI"
                    print(f"    {role}: {msg['content'][:80]}...")
                print()
                continue
            
            elif user_input == "/clear":
                self.conversation_history.clear()
                self.model.memory.working.clear()
                print("  Geçmiş ve working memory temizlendi.\n")
                continue
            
            # Normal yanıt
            result = self.respond(user_input)
            
            print(f"\n  AGI > {result['response']}")
            
            if result['action']:
                action = result['action']
                status = "✅" if action['allowed'] else "❌"
                print(f"  [Action: {action['action_name']} {status}]", end="")
                if action['executed'] and action['result']:
                    print(f" → {action['result']['status']}: {action['result']['details']}")
                else:
                    print()
            
            print(f"  [Confidence: {result['confidence']:.3f} | "
                  f"Tokens: {result['num_tokens']}]\n")
