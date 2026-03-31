"""
constitution.py — Pydantic-based Kt Anayasası (Proto-AGI)

Modelin her kararını denetleyen katı kontrol sistemi:
  - ConstitutionConfig: İzinli/yasaklı action'lar, risk limitleri
  - ConstitutionGuard:  Action çıktısını filtreler, ihlalleri loglar
  - RateLimiter:        Zaman bazlı action sınırlama
  - RollbackManager:    Geri alınabilir action'lar için undo mekanizması
  - EscalationProtocol: Human-in-the-loop tetikleme
"""

import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Callable
from pathlib import Path
from enum import Enum

try:
    from pydantic import BaseModel, Field, field_validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Pydantic yoksa fallback dataclass kullan
    BaseModel = object
    
import torch

from action_space import ActionType, ActionDefinition, DEFAULT_ACTION_CATALOG


# ───────────────────────────── Violation Types ────────────────────

class ViolationType(Enum):
    FORBIDDEN_ACTION = "forbidden_action"
    RISK_EXCEEDED = "risk_exceeded"
    RATE_LIMIT = "rate_limit"
    LOW_CONFIDENCE = "low_confidence"
    SANDBOX_REQUIRED = "sandbox_required"
    ESCALATION_REQUIRED = "escalation_required"


@dataclass
class Violation:
    """Bir anayasa ihlali kaydı."""
    violation_type: ViolationType
    action_type: ActionType
    details: str
    timestamp: float = 0.0
    severity: float = 0.0  # 0-1
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# ───────────────────────────── Constitution Config ────────────────

if HAS_PYDANTIC:
    class ConstitutionConfig(BaseModel):
        """
        Kt Anayasası — Modelin tüm kararlarını yöneten kurallar.
        Pydantic v2 ile validation.
        """
        # İzinli action tipleri
        allowed_actions: List[int] = Field(
            default=[0, 1, 2, 3, 4, 5, 6, 7],
            description="İzin verilen action type ID'leri"
        )
        
        # Yasaklı action tipleri (allowed'dan önce kontrol edilir)
        forbidden_actions: List[int] = Field(
            default=[],
            description="Kesinlikle yasaklanan action type ID'leri"
        )
        
        # Risk limitleri
        max_risk_score: float = Field(
            default=0.7,
            ge=0.0, le=1.0,
            description="Bu üzerindeki risk skorlu action'lar engellenir"
        )
        
        # Güven eşikleri
        min_confidence: float = Field(
            default=0.3,
            ge=0.0, le=1.0,
            description="Bu altındaki güven skorlu action'lar sandbox'a yönlendirilir"
        )
        
        escalation_confidence: float = Field(
            default=0.15,
            ge=0.0, le=1.0,
            description="Bu altındaki güven skorlu action'lar insana yönlendirilir"
        )
        
        # Sandbox kuralları
        sandbox_required_actions: List[int] = Field(
            default=[1, 2, 3],  # API, Simulation, WebCrawl
            description="Sandbox testi zorunlu olan action tipleri"
        )
        
        # Rate limiting
        max_actions_per_minute: int = Field(
            default=60,
            description="Dakikada max action sayısı"
        )
        
        max_api_calls_per_minute: int = Field(
            default=10,
            description="Dakikada max dış API çağrısı"
        )
        
        # Logging
        log_violations: bool = True
        violation_log_path: str = "./logs/violations.json"
        
        @field_validator('forbidden_actions')
        @classmethod
        def validate_forbidden(cls, v, info):
            allowed = info.data.get('allowed_actions', [])
            overlap = set(v) & set(allowed)
            if overlap:
                # Forbidden her zaman önceliklidir
                pass
            return v
            
else:
    # Pydantic yoksa basit dataclass
    @dataclass
    class ConstitutionConfig:
        allowed_actions: List[int] = field(default_factory=lambda: [0,1,2,3,4,5,6,7])
        forbidden_actions: List[int] = field(default_factory=list)
        max_risk_score: float = 0.7
        min_confidence: float = 0.3
        escalation_confidence: float = 0.15
        sandbox_required_actions: List[int] = field(default_factory=lambda: [1,2,3])
        max_actions_per_minute: int = 60
        max_api_calls_per_minute: int = 10
        log_violations: bool = True
        violation_log_path: str = "./logs/violations.json"


# ───────────────────────────── Rate Limiter ───────────────────────

class RateLimiter:
    """
    Zaman bazlı action sınırlama.
    Sliding window ile dakikada max action sayısını kontrol eder.
    """
    
    def __init__(self, max_per_minute: int = 60):
        self.max_per_minute = max_per_minute
        self.timestamps: List[float] = []
    
    def check(self) -> bool:
        """Action yapılabilir mi? True = izin var."""
        now = time.time()
        # Son 60 saniyedeki action'ları filtrele
        self.timestamps = [t for t in self.timestamps if now - t < 60.0]
        return len(self.timestamps) < self.max_per_minute
    
    def record(self):
        """Bir action yapıldığını kaydet."""
        self.timestamps.append(time.time())
    
    def remaining(self) -> int:
        """Kalan izin sayısı."""
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < 60.0]
        return max(0, self.max_per_minute - len(self.timestamps))


# ───────────────────────────── Rollback Manager ───────────────────

@dataclass
class ActionRecord:
    """Gerçekleştirilen bir action'ın kaydı."""
    action_type: ActionType
    params: Dict[str, Any]
    timestamp: float
    outcome: Optional[str] = None
    rollback_fn: Optional[Callable] = None
    rolled_back: bool = False


class RollbackManager:
    """
    Reversible action'lar için geri alma mekanizması.
    Her action bir rollback fonksiyonu kaydedebilir.
    """
    
    def __init__(self, max_history: int = 100):
        self.history: List[ActionRecord] = []
        self.max_history = max_history
    
    def record(
        self, 
        action_type: ActionType, 
        params: Dict[str, Any],
        rollback_fn: Optional[Callable] = None
    ) -> int:
        """Action kaydet, index döndür."""
        record = ActionRecord(
            action_type=action_type,
            params=params,
            timestamp=time.time(),
            rollback_fn=rollback_fn
        )
        self.history.append(record)
        
        # Kapasiteyi aşıyorsa eskilerini sil
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        return len(self.history) - 1
    
    def rollback(self, index: int) -> bool:
        """Belirli bir action'ı geri al."""
        if index < 0 or index >= len(self.history):
            return False
        
        record = self.history[index]
        if record.rolled_back:
            return False
        
        if record.rollback_fn is not None:
            try:
                record.rollback_fn()
                record.rolled_back = True
                return True
            except Exception:
                return False
        
        return False
    
    def rollback_last(self) -> bool:
        """Son action'ı geri al."""
        if not self.history:
            return False
        return self.rollback(len(self.history) - 1)


# ───────────────────────────── Escalation Protocol ────────────────

class EscalationProtocol:
    """
    Human-in-the-loop tetikleme mekanizması.
    Risk veya güven eşiği aşıldığında insan onayı ister.
    """
    
    def __init__(self):
        self.pending_escalations: List[Dict[str, Any]] = []
        self.escalation_callback: Optional[Callable] = None
    
    def set_callback(self, callback: Callable):
        """İnsan bildirim fonksiyonunu ayarla."""
        self.escalation_callback = callback
    
    def escalate(
        self, 
        reason: str, 
        action_type: ActionType, 
        confidence: float,
        risk: float
    ) -> Dict[str, Any]:
        """
        İnsan onayı iste.
        return: escalation kaydı
        """
        escalation = {
            'reason': reason,
            'action_type': action_type.name,
            'confidence': confidence,
            'risk': risk,
            'timestamp': time.time(),
            'status': 'pending',  # pending | approved | rejected
        }
        self.pending_escalations.append(escalation)
        
        if self.escalation_callback:
            self.escalation_callback(escalation)
        
        return escalation
    
    def approve(self, index: int):
        """Bekleyen escalation'ı onayla."""
        if 0 <= index < len(self.pending_escalations):
            self.pending_escalations[index]['status'] = 'approved'
    
    def reject(self, index: int):
        """Bekleyen escalation'ı reddet."""
        if 0 <= index < len(self.pending_escalations):
            self.pending_escalations[index]['status'] = 'rejected'


# ───────────────────────────── Constitution Guard ─────────────────

class ConstitutionGuard:
    """
    Ana filtre — Model çıktısını anayasaya göre denetler.
    
    Her action kararı bu guard'dan geçer:
      1. Yasaklı mı? → ENGELLE
      2. İzinli mi? → devam
      3. Risk skoru yüksek mi? → ENGELLE
      4. Güven düşük mü? → SANDBOX veya ESCALATION
      5. Rate limit aşıldı mı? → ENGELLE
      6. Sandbox zorunlu mu? → yönlendir
    """
    
    def __init__(self, config: Optional[ConstitutionConfig] = None):
        if config is None:
            config = ConstitutionConfig()
        self.config = config
        
        self.general_limiter = RateLimiter(config.max_actions_per_minute)
        self.api_limiter = RateLimiter(config.max_api_calls_per_minute)
        self.rollback = RollbackManager()
        self.escalation = EscalationProtocol()
        
        self.violations: List[Violation] = []
        self.action_catalog = {
            ad.action_type.value: ad for ad in DEFAULT_ACTION_CATALOG
        }
    
    def check(
        self, 
        action_idx: int, 
        confidence: float = 1.0,
        action_probs: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Bir action kararını anayasaya göre kontrol et.
        
        return: {
            'allowed': bool,
            'reason': str,
            'requires_sandbox': bool,
            'requires_escalation': bool,
            'violations': List[Violation]
        }
        """
        result = {
            'allowed': True,
            'reason': 'ok',
            'requires_sandbox': False,
            'requires_escalation': False,
            'violations': []
        }
        
        # 1. Yasaklı action kontrolü
        if action_idx in self.config.forbidden_actions:
            v = Violation(
                ViolationType.FORBIDDEN_ACTION,
                ActionType(action_idx) if action_idx < len(ActionType) else ActionType.LOCAL_COMPUTE,
                f"Action {action_idx} anayasa tarafından yasaklanmış"
            )
            result['violations'].append(v)
            result['allowed'] = False
            result['reason'] = 'forbidden_action'
            self._log_violation(v)
            return result
        
        # 2. İzinli action kontrolü
        if action_idx not in self.config.allowed_actions:
            v = Violation(
                ViolationType.FORBIDDEN_ACTION,
                ActionType(action_idx) if action_idx < len(ActionType) else ActionType.LOCAL_COMPUTE,
                f"Action {action_idx} izinli listede değil"
            )
            result['violations'].append(v)
            result['allowed'] = False
            result['reason'] = 'not_allowed'
            self._log_violation(v)
            return result
        
        # 3. Risk skoru kontrolü
        action_def = self.action_catalog.get(action_idx)
        if action_def and action_def.risk_level > self.config.max_risk_score:
            v = Violation(
                ViolationType.RISK_EXCEEDED,
                ActionType(action_idx),
                f"Risk ({action_def.risk_level}) > limit ({self.config.max_risk_score})",
                severity=action_def.risk_level
            )
            result['violations'].append(v)
            result['allowed'] = False
            result['reason'] = 'risk_exceeded'
            self._log_violation(v)
            return result
        
        # 4. Rate limit kontrolü
        if not self.general_limiter.check():
            v = Violation(
                ViolationType.RATE_LIMIT,
                ActionType(action_idx) if action_idx < len(ActionType) else ActionType.LOCAL_COMPUTE,
                f"Dakikada max {self.config.max_actions_per_minute} action aşıldı"
            )
            result['violations'].append(v)
            result['allowed'] = False
            result['reason'] = 'rate_limit'
            self._log_violation(v)
            return result
        
        # API özel rate limit
        if action_idx == ActionType.EXTERNAL_API.value:
            if not self.api_limiter.check():
                v = Violation(
                    ViolationType.RATE_LIMIT,
                    ActionType.EXTERNAL_API,
                    f"Dakikada max {self.config.max_api_calls_per_minute} API çağrısı aşıldı"
                )
                result['violations'].append(v)
                result['allowed'] = False
                result['reason'] = 'api_rate_limit'
                self._log_violation(v)
                return result
        
        # 5. Güven kontrolü
        if confidence < self.config.escalation_confidence:
            result['requires_escalation'] = True
            v = Violation(
                ViolationType.ESCALATION_REQUIRED,
                ActionType(action_idx) if action_idx < len(ActionType) else ActionType.LOCAL_COMPUTE,
                f"Güven ({confidence:.3f}) < escalation eşiği ({self.config.escalation_confidence})"
            )
            result['violations'].append(v)
            self._log_violation(v)
        
        elif confidence < self.config.min_confidence:
            result['requires_sandbox'] = True
            v = Violation(
                ViolationType.LOW_CONFIDENCE,
                ActionType(action_idx) if action_idx < len(ActionType) else ActionType.LOCAL_COMPUTE,
                f"Güven ({confidence:.3f}) < min eşik ({self.config.min_confidence})"
            )
            result['violations'].append(v)
            self._log_violation(v)
        
        # 6. Sandbox zorunluluğu
        if action_idx in self.config.sandbox_required_actions:
            result['requires_sandbox'] = True
        
        # Rate limiter'a kaydet (izin verildiyse)
        if result['allowed']:
            self.general_limiter.record()
            if action_idx == ActionType.EXTERNAL_API.value:
                self.api_limiter.record()
        
        return result
    
    def _log_violation(self, violation: Violation):
        """Violation'ı kaydet."""
        self.violations.append(violation)
        
        if self.config.log_violations:
            log_dir = Path(self.config.violation_log_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Append mode
            log_path = Path(self.config.violation_log_path)
            logs = []
            if log_path.exists():
                try:
                    with open(log_path, 'r') as f:
                        logs = json.load(f)
                except (json.JSONDecodeError, IOError):
                    logs = []
            
            logs.append({
                'type': violation.violation_type.value,
                'action': violation.action_type.name if isinstance(violation.action_type, ActionType) else str(violation.action_type),
                'details': violation.details,
                'timestamp': violation.timestamp,
                'severity': violation.severity
            })
            
            with open(log_path, 'w') as f:
                json.dump(logs, f, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Guard istatistikleri."""
        return {
            'total_violations': len(self.violations),
            'actions_remaining': self.general_limiter.remaining(),
            'api_calls_remaining': self.api_limiter.remaining(),
            'pending_escalations': len([
                e for e in self.escalation.pending_escalations 
                if e['status'] == 'pending'
            ]),
            'rollback_history_size': len(self.rollback.history),
        }
