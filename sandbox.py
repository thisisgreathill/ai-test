"""
sandbox.py — İzole Simülasyon Ortamı (Proto-AGI)

Her action kararını izole ortamda test eder:
  - ActionSimulator:   Her action tipi için mock executor
  - OutcomeEvaluator:  Sonucu puanlayan değerlendirici
  - SandboxEnvironment: Ana izolasyon ortamı
"""

import torch
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from action_space import ActionType, ActionDefinition, DEFAULT_ACTION_CATALOG


# ───────────────────────────── Outcome ────────────────────────────

class OutcomeStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class SandboxOutcome:
    """Sandbox'ta çalıştırılan bir action'ın sonucu."""
    status: OutcomeStatus
    reward: float                   # -1.0 → +1.0
    details: str = ""
    execution_time: float = 0.0     # saniye
    side_effects: List[str] = field(default_factory=list)
    output_data: Any = None


# ───────────────────────────── Action Simulator ───────────────────

class ActionSimulator:
    """
    Her action tipi için mock executor.
    Gerçek sisteme dokunmadan, action'ın beklenen davranışını simüle eder.
    
    Gerçek dünyaya bağlanmadan:
      - LOCAL_COMPUTE:        Basit logic testi
      - EXTERNAL_API:         Mock API response
      - ISOLATED_SIMULATION:  İç simülasyon
      - WEB_CRAWL:            Mock web response
      - MEMORY_QUERY:         Memory check
      - SELF_REFLECT:         Always success (iç operasyon)
      - DELEGATE:             Alt-görev simülasyonu
      - WAIT:                 Always success (pasif)
    """
    
    def __init__(self):
        self.simulators: Dict[ActionType, Callable] = {
            ActionType.LOCAL_COMPUTE: self._sim_local_compute,
            ActionType.EXTERNAL_API: self._sim_external_api,
            ActionType.ISOLATED_SIMULATION: self._sim_isolated,
            ActionType.WEB_CRAWL: self._sim_web_crawl,
            ActionType.MEMORY_QUERY: self._sim_memory_query,
            ActionType.SELF_REFLECT: self._sim_self_reflect,
            ActionType.DELEGATE: self._sim_delegate,
            ActionType.WAIT: self._sim_wait,
        }
        
        # Configurable failure rates (test amaçlı)
        self.failure_rates: Dict[ActionType, float] = {
            ActionType.LOCAL_COMPUTE: 0.05,
            ActionType.EXTERNAL_API: 0.20,
            ActionType.ISOLATED_SIMULATION: 0.10,
            ActionType.WEB_CRAWL: 0.25,
            ActionType.MEMORY_QUERY: 0.02,
            ActionType.SELF_REFLECT: 0.01,
            ActionType.DELEGATE: 0.15,
            ActionType.WAIT: 0.0,
        }
    
    def simulate(
        self, 
        action_type: ActionType, 
        params: Optional[torch.Tensor] = None,
        context: Optional[Dict] = None
    ) -> SandboxOutcome:
        """Bir action'ı simüle et."""
        start_time = time.time()
        
        simulator = self.simulators.get(action_type, self._sim_unknown)
        outcome = simulator(params, context or {})
        
        outcome.execution_time = time.time() - start_time
        return outcome
    
    def _sim_local_compute(self, params, context) -> SandboxOutcome:
        if random.random() < self.failure_rates[ActionType.LOCAL_COMPUTE]:
            return SandboxOutcome(
                status=OutcomeStatus.FAILURE,
                reward=-0.2,
                details="Yerel hesaplama hatası"
            )
        return SandboxOutcome(
            status=OutcomeStatus.SUCCESS,
            reward=0.5,
            details="Yerel hesaplama başarılı"
        )
    
    def _sim_external_api(self, params, context) -> SandboxOutcome:
        if random.random() < self.failure_rates[ActionType.EXTERNAL_API]:
            return SandboxOutcome(
                status=OutcomeStatus.ERROR,
                reward=-0.5,
                details="API timeout / hata",
                side_effects=["network_request_failed"]
            )
        return SandboxOutcome(
            status=OutcomeStatus.SUCCESS,
            reward=0.8,
            details="API çağrısı başarılı",
            side_effects=["network_request_sent"]
        )
    
    def _sim_isolated(self, params, context) -> SandboxOutcome:
        if random.random() < self.failure_rates[ActionType.ISOLATED_SIMULATION]:
            return SandboxOutcome(
                status=OutcomeStatus.PARTIAL,
                reward=0.1,
                details="Simülasyon kısmen başarılı"
            )
        return SandboxOutcome(
            status=OutcomeStatus.SUCCESS,
            reward=0.6,
            details="İzole simülasyon tamamlandı"
        )
    
    def _sim_web_crawl(self, params, context) -> SandboxOutcome:
        if random.random() < self.failure_rates[ActionType.WEB_CRAWL]:
            return SandboxOutcome(
                status=OutcomeStatus.TIMEOUT,
                reward=-0.3,
                details="Web tarama zaman aşımı"
            )
        return SandboxOutcome(
            status=OutcomeStatus.SUCCESS,
            reward=0.7,
            details="Web tarama başarılı",
            side_effects=["web_page_fetched"]
        )
    
    def _sim_memory_query(self, params, context) -> SandboxOutcome:
        if random.random() < self.failure_rates[ActionType.MEMORY_QUERY]:
            return SandboxOutcome(
                status=OutcomeStatus.FAILURE,
                reward=-0.1,
                details="Hafızada ilgili bilgi bulunamadı"
            )
        return SandboxOutcome(
            status=OutcomeStatus.SUCCESS,
            reward=0.4,
            details="Hafıza sorgusu başarılı"
        )
    
    def _sim_self_reflect(self, params, context) -> SandboxOutcome:
        return SandboxOutcome(
            status=OutcomeStatus.SUCCESS,
            reward=0.3,
            details="Öz-yansıtma tamamlandı"
        )
    
    def _sim_delegate(self, params, context) -> SandboxOutcome:
        if random.random() < self.failure_rates[ActionType.DELEGATE]:
            return SandboxOutcome(
                status=OutcomeStatus.PARTIAL,
                reward=0.0,
                details="Alt-görev kısmen başarılı"
            )
        return SandboxOutcome(
            status=OutcomeStatus.SUCCESS,
            reward=0.6,
            details="Alt-görev delegasyonu başarılı"
        )
    
    def _sim_wait(self, params, context) -> SandboxOutcome:
        return SandboxOutcome(
            status=OutcomeStatus.SUCCESS,
            reward=0.1,
            details="Bekleme tamamlandı"
        )
    
    def _sim_unknown(self, params, context) -> SandboxOutcome:
        return SandboxOutcome(
            status=OutcomeStatus.ERROR,
            reward=-1.0,
            details="Bilinmeyen action tipi!"
        )


# ───────────────────────────── Outcome Evaluator ──────────────────

class OutcomeEvaluator:
    """
    Sandbox sonucunu değerlendiren puanlama sistemi.
    
    Ham sandbox outcome'unu alır, modelin öğrenebileceği
    bir reward signal'e dönüştürür.
    """
    
    def __init__(self):
        # Action-tipine göre başarı ağırlıkları
        self.success_weights = {
            ActionType.LOCAL_COMPUTE: 1.0,
            ActionType.EXTERNAL_API: 1.5,       # Dış API başarısı daha değerli
            ActionType.ISOLATED_SIMULATION: 1.2,
            ActionType.WEB_CRAWL: 1.3,
            ActionType.MEMORY_QUERY: 0.8,
            ActionType.SELF_REFLECT: 0.5,
            ActionType.DELEGATE: 1.1,
            ActionType.WAIT: 0.3,
        }
    
    def evaluate(
        self, 
        outcome: SandboxOutcome, 
        action_type: ActionType,
        confidence: float = 0.5
    ) -> float:
        """
        Outcome'u değerlendir ve final reward üret.
        
        Faktörler:
          1. Ham reward (sandbox'tan)
          2. Action tipi ağırlığı
          3. Confidence calibration bonus
          4. Execution time penalty
        """
        base_reward = outcome.reward
        weight = self.success_weights.get(action_type, 1.0)
        
        # Confidence calibration: doğru güvenle doğru karar → bonus
        if outcome.status == OutcomeStatus.SUCCESS and confidence > 0.5:
            calibration_bonus = 0.1 * confidence
        elif outcome.status == OutcomeStatus.FAILURE and confidence < 0.3:
            calibration_bonus = 0.05  # Düşük güvenle hatalı karar = az ceza
        else:
            calibration_bonus = 0.0
        
        # Execution time penalty (çok yavaş action'lar cezalandırılır)
        time_penalty = min(0.1, outcome.execution_time * 0.01)
        
        final_reward = base_reward * weight + calibration_bonus - time_penalty
        
        return max(-10.0, min(10.0, final_reward))  # Clip


# ───────────────────────────── Sandbox Environment ────────────────

class SandboxEnvironment:
    """
    Ana izolasyon ortamı.
    
    Her action kararını güvenli bir sandbox'ta test eder,
    sonucu değerlendirir ve reward signal üretir.
    
    Akış:
      1. ConstitutionGuard action'ı onaylar
      2. Action sandbox'a gönderilir
      3. ActionSimulator mock execution yapar
      4. OutcomeEvaluator sonucu puanlar
      5. Reward signal modele döner
    """
    
    def __init__(self):
        self.simulator = ActionSimulator()
        self.evaluator = OutcomeEvaluator()
        self.execution_history: List[Dict[str, Any]] = []
    
    def execute(
        self, 
        action_type: ActionType,
        action_params: Optional[torch.Tensor] = None,
        confidence: float = 0.5,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Action'ı sandbox'ta çalıştır ve sonuç döndür.
        
        return: {
            'outcome': SandboxOutcome,
            'reward': float,
            'action_type': ActionType,
        }
        """
        # Simulate
        outcome = self.simulator.simulate(action_type, action_params, context)
        
        # Evaluate
        reward = self.evaluator.evaluate(outcome, action_type, confidence)
        
        # Record
        record = {
            'action_type': action_type,
            'outcome': outcome,
            'reward': reward,
            'confidence': confidence,
            'timestamp': time.time()
        }
        self.execution_history.append(record)
        
        return record
    
    def get_stats(self) -> Dict[str, Any]:
        """Sandbox istatistikleri."""
        if not self.execution_history:
            return {'total_executions': 0}
        
        total = len(self.execution_history)
        successes = sum(
            1 for r in self.execution_history 
            if r['outcome'].status == OutcomeStatus.SUCCESS
        )
        avg_reward = sum(r['reward'] for r in self.execution_history) / total
        
        return {
            'total_executions': total,
            'success_rate': successes / total,
            'average_reward': avg_reward,
            'last_reward': self.execution_history[-1]['reward']
        }
    
    def reset(self):
        """Execution history'yi temizle."""
        self.execution_history.clear()
