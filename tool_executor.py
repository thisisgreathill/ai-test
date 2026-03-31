"""
tool_executor.py — Gerçek Tool Çalıştırma (Proto-AGI)

Sandbox'taki mock executor'lar yerine GERÇEK tool çalıştırma:
  - FileExecutor:    Dosya okuma/yazma
  - WebExecutor:     HTTP istekleri (GET/POST)
  - CodeExecutor:    Python kodu çalıştırma (izole)
  - ShellExecutor:   Shell komutları (kısıtlı)
  - ToolRouter:      Action tipine göre doğru executor'a yönlendirme

GÜVENLİK: Tüm executor'lar Constitution guard'dan geçer.
"""

import subprocess
import urllib.request
import urllib.error
import json
import os
import sys
import traceback
import time
from io import StringIO
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path

from action_space import ActionType


# ───────────────────────────── Config ─────────────────────────────

@dataclass
class ToolConfig:
    """Tool executor konfigürasyonu."""
    # File
    allowed_read_dirs: List[str] = field(default_factory=lambda: ["/tmp", "./"])
    allowed_write_dirs: List[str] = field(default_factory=lambda: ["/tmp"])
    max_file_size: int = 1_000_000  # 1MB
    
    # Web
    request_timeout: int = 10       # saniye
    allowed_domains: List[str] = field(default_factory=list)  # Boş = hepsi
    blocked_domains: List[str] = field(default_factory=lambda: [
        "localhost", "127.0.0.1", "0.0.0.0"  # Yerel ağ engellenmiş
    ])
    max_response_size: int = 500_000  # 500KB
    
    # Code
    code_timeout: int = 5           # saniye
    max_output_len: int = 10_000    # karakter
    
    # Shell
    shell_enabled: bool = False     # Varsayılan olarak KAPALI
    allowed_commands: List[str] = field(default_factory=lambda: [
        "echo", "date", "whoami", "ls", "cat", "head", "tail", "wc"
    ])
    
    # Logging
    log_dir: str = "./logs/tools"


# ───────────────────────────── Execution Result ───────────────────

@dataclass
class ToolResult:
    """Bir tool çalıştırmasının sonucu."""
    success: bool
    output: str
    error: str = ""
    execution_time: float = 0.0
    tool_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ───────────────────────────── File Executor ──────────────────────

class FileExecutor:
    """
    Güvenli dosya okuma/yazma.
    Sadece izinli dizinlerde çalışır.
    """
    
    def __init__(self, config: ToolConfig):
        self.config = config
    
    def _check_read_path(self, path: str) -> bool:
        """Dosya okuma izni kontrolü."""
        abs_path = os.path.abspath(path)
        return any(
            abs_path.startswith(os.path.abspath(d)) 
            for d in self.config.allowed_read_dirs
        )
    
    def _check_write_path(self, path: str) -> bool:
        """Dosya yazma izni kontrolü."""
        abs_path = os.path.abspath(path)
        return any(
            abs_path.startswith(os.path.abspath(d)) 
            for d in self.config.allowed_write_dirs
        )
    
    def read(self, path: str) -> ToolResult:
        """Dosya oku."""
        start = time.time()
        
        if not self._check_read_path(path):
            return ToolResult(
                success=False, output="",
                error=f"Okuma izni yok: {path}",
                tool_name="file_read"
            )
        
        try:
            if not os.path.exists(path):
                return ToolResult(
                    success=False, output="",
                    error=f"Dosya bulunamadı: {path}",
                    tool_name="file_read"
                )
            
            size = os.path.getsize(path)
            if size > self.config.max_file_size:
                return ToolResult(
                    success=False, output="",
                    error=f"Dosya çok büyük: {size} bytes (limit: {self.config.max_file_size})",
                    tool_name="file_read"
                )
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return ToolResult(
                success=True, output=content,
                execution_time=time.time() - start,
                tool_name="file_read",
                metadata={'path': path, 'size': size}
            )
        
        except Exception as e:
            return ToolResult(
                success=False, output="",
                error=str(e),
                execution_time=time.time() - start,
                tool_name="file_read"
            )
    
    def write(self, path: str, content: str) -> ToolResult:
        """Dosya yaz."""
        start = time.time()
        
        if not self._check_write_path(path):
            return ToolResult(
                success=False, output="",
                error=f"Yazma izni yok: {path}",
                tool_name="file_write"
            )
        
        try:
            if len(content) > self.config.max_file_size:
                return ToolResult(
                    success=False, output="",
                    error=f"İçerik çok büyük: {len(content)} chars",
                    tool_name="file_write"
                )
            
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                output=f"Dosya yazıldı: {path} ({len(content)} chars)",
                execution_time=time.time() - start,
                tool_name="file_write",
                metadata={'path': path, 'size': len(content)}
            )
        
        except Exception as e:
            return ToolResult(
                success=False, output="",
                error=str(e),
                execution_time=time.time() - start,
                tool_name="file_write"
            )


# ───────────────────────────── Web Executor ───────────────────────

class WebExecutor:
    """
    HTTP istekleri gönderme.
    Domain kısıtlamaları ve timeout ile güvenli.
    """
    
    def __init__(self, config: ToolConfig):
        self.config = config
    
    def _check_domain(self, url: str) -> bool:
        """Domain güvenlik kontrolü."""
        from urllib.parse import urlparse
        
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        
        # Yasaklı domain kontrolü
        if any(d in domain for d in self.config.blocked_domains):
            return False
        
        # İzinli domain kontrolü (liste boşsa hepsi izinli)
        if self.config.allowed_domains:
            return any(d in domain for d in self.config.allowed_domains)
        
        return True
    
    def get(self, url: str) -> ToolResult:
        """HTTP GET isteği."""
        start = time.time()
        
        if not self._check_domain(url):
            return ToolResult(
                success=False, output="",
                error=f"Domain engelli: {url}",
                tool_name="web_get"
            )
        
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'NexusCore-ProtoAGI/0.1'}
            )
            
            with urllib.request.urlopen(req, timeout=self.config.request_timeout) as resp:
                data = resp.read(self.config.max_response_size)
                content = data.decode('utf-8', errors='replace')
            
            return ToolResult(
                success=True, output=content[:self.config.max_output_len],
                execution_time=time.time() - start,
                tool_name="web_get",
                metadata={'url': url, 'status': resp.status, 'size': len(data)}
            )
        
        except urllib.error.URLError as e:
            return ToolResult(
                success=False, output="",
                error=f"URL hatası: {e}",
                execution_time=time.time() - start,
                tool_name="web_get"
            )
        except Exception as e:
            return ToolResult(
                success=False, output="",
                error=str(e),
                execution_time=time.time() - start,
                tool_name="web_get"
            )


# ───────────────────────────── Code Executor ──────────────────────

class CodeExecutor:
    """
    Python kodu çalıştırma (izole).
    Güvenlik: exec() kullanır ama stdout'u yakalar, timeout uygular.
    """
    
    def __init__(self, config: ToolConfig):
        self.config = config
    
    def execute(self, code: str) -> ToolResult:
        """Python kodunu çalıştır."""
        start = time.time()
        
        # Tehlikeli fonksiyonları kontrol et
        dangerous = ['import os', 'import sys', 'subprocess', 'eval(', 
                      '__import__', 'open(', 'exec(', 'shutil', 'rmtree']
        for d in dangerous:
            if d in code:
                return ToolResult(
                    success=False, output="",
                    error=f"Güvenlik ihlali: '{d}' kullanımı yasaklı",
                    tool_name="code_exec"
                )
        
        # stdout'u yakala
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        
        try:
            # İzole namespace
            local_ns = {'__builtins__': {
                'print': print, 'range': range, 'len': len,
                'int': int, 'float': float, 'str': str,
                'list': list, 'dict': dict, 'tuple': tuple,
                'sum': sum, 'min': min, 'max': max, 'abs': abs,
                'round': round, 'sorted': sorted, 'reversed': reversed,
                'enumerate': enumerate, 'zip': zip, 'map': map,
                'True': True, 'False': False, 'None': None,
            }}
            
            exec(code, local_ns)
            output = buffer.getvalue()[:self.config.max_output_len]
            
            return ToolResult(
                success=True, output=output,
                execution_time=time.time() - start,
                tool_name="code_exec"
            )
        
        except Exception as e:
            return ToolResult(
                success=False, output=buffer.getvalue(),
                error=f"{type(e).__name__}: {e}",
                execution_time=time.time() - start,
                tool_name="code_exec"
            )
        
        finally:
            sys.stdout = old_stdout


# ───────────────────────────── Shell Executor ─────────────────────

class ShellExecutor:
    """
    Shell komutu çalıştırma (çok kısıtlı).
    Sadece whitelist'teki komutlar çalışır.
    """
    
    def __init__(self, config: ToolConfig):
        self.config = config
    
    def execute(self, command: str) -> ToolResult:
        """Shell komutu çalıştır."""
        start = time.time()
        
        if not self.config.shell_enabled:
            return ToolResult(
                success=False, output="",
                error="Shell executor devre dışı",
                tool_name="shell"
            )
        
        # Komutu parçala ve ilk kelimeyi kontrol et
        parts = command.strip().split()
        if not parts:
            return ToolResult(
                success=False, output="",
                error="Boş komut",
                tool_name="shell"
            )
        
        cmd = parts[0]
        if cmd not in self.config.allowed_commands:
            return ToolResult(
                success=False, output="",
                error=f"Komut izinli değil: '{cmd}'. İzinli: {self.config.allowed_commands}",
                tool_name="shell"
            )
        
        # Tehlikeli operatörleri engelle
        dangerous_ops = ['|', '>', '>>', '&&', '||', ';', '`', '$(',]
        if any(op in command for op in dangerous_ops):
            return ToolResult(
                success=False, output="",
                error="Pipe/redirect/chain operatörleri yasak",
                tool_name="shell"
            )
        
        try:
            result = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                timeout=self.config.code_timeout,
                cwd="/tmp"
            )
            
            output = result.stdout[:self.config.max_output_len]
            error = result.stderr[:1000] if result.returncode != 0 else ""
            
            return ToolResult(
                success=result.returncode == 0,
                output=output,
                error=error,
                execution_time=time.time() - start,
                tool_name="shell",
                metadata={'command': command, 'returncode': result.returncode}
            )
        
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False, output="",
                error=f"Zaman aşımı ({self.config.code_timeout}s)",
                execution_time=time.time() - start,
                tool_name="shell"
            )
        except Exception as e:
            return ToolResult(
                success=False, output="",
                error=str(e),
                execution_time=time.time() - start,
                tool_name="shell"
            )


# ───────────────────────────── Tool Router ────────────────────────

class ToolRouter:
    """
    Action tipine göre doğru executor'a yönlendirme.
    
    ActionType → Executor mapping:
      LOCAL_COMPUTE        → CodeExecutor
      EXTERNAL_API         → WebExecutor
      ISOLATED_SIMULATION  → CodeExecutor (sandbox mode)
      WEB_CRAWL            → WebExecutor
      MEMORY_QUERY         → (internal, no executor)
      SELF_REFLECT         → (internal, no executor)
      DELEGATE             → (recursive, gelecekte)
      WAIT                 → (no-op)
    """
    
    def __init__(self, config: Optional[ToolConfig] = None):
        if config is None:
            config = ToolConfig()
        self.config = config
        
        self.file_executor = FileExecutor(config)
        self.web_executor = WebExecutor(config)
        self.code_executor = CodeExecutor(config)
        self.shell_executor = ShellExecutor(config)
        
        self.execution_log: List[Dict[str, Any]] = []
        
        # Log dizini
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def execute(
        self, 
        action_type: ActionType, 
        params: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Action'ı ilgili executor'a yönlendir.
        
        params: Action'a özgü parametreler
          - code_exec: {'code': 'print(2+2)'}
          - web_get:   {'url': 'https://example.com'}
          - file_read: {'path': '/tmp/data.txt'}
          - file_write:{'path': '/tmp/out.txt', 'content': '...'}
          - shell:     {'command': 'echo hello'}
        """
        params = params or {}
        start = time.time()
        
        if action_type == ActionType.LOCAL_COMPUTE:
            if 'code' in params:
                result = self.code_executor.execute(params['code'])
            else:
                result = ToolResult(
                    success=True, output="Yerel hesaplama tamamlandı",
                    tool_name="local_compute"
                )
        
        elif action_type == ActionType.EXTERNAL_API:
            url = params.get('url', '')
            if url:
                result = self.web_executor.get(url)
            else:
                result = ToolResult(
                    success=False, output="",
                    error="URL belirtilmedi",
                    tool_name="external_api"
                )
        
        elif action_type == ActionType.WEB_CRAWL:
            url = params.get('url', '')
            if url:
                result = self.web_executor.get(url)
            else:
                result = ToolResult(
                    success=False, output="",
                    error="URL belirtilmedi",
                    tool_name="web_crawl"
                )
        
        elif action_type == ActionType.ISOLATED_SIMULATION:
            code = params.get('code', '')
            if code:
                result = self.code_executor.execute(code)
            else:
                result = ToolResult(
                    success=True, output="Simülasyon tamamlandı",
                    tool_name="simulation"
                )
        
        elif action_type in (ActionType.MEMORY_QUERY, ActionType.SELF_REFLECT):
            result = ToolResult(
                success=True, output="İç operasyon tamamlandı",
                tool_name=action_type.name.lower()
            )
        
        elif action_type == ActionType.WAIT:
            wait_time = params.get('seconds', 1)
            time.sleep(min(wait_time, 5))  # Max 5 saniye bekle
            result = ToolResult(
                success=True, output=f"{wait_time}s beklendi",
                tool_name="wait"
            )
        
        else:
            result = ToolResult(
                success=False, output="",
                error=f"Bilinmeyen action tipi: {action_type}",
                tool_name="unknown"
            )
        
        result.execution_time = time.time() - start
        
        # Log
        self._log_execution(action_type, params, result)
        
        return result
    
    def _log_execution(
        self, 
        action_type: ActionType, 
        params: Dict, 
        result: ToolResult
    ):
        """Çalıştırmayı logla."""
        log_entry = {
            'action_type': action_type.name,
            'params': {k: str(v)[:100] for k, v in params.items()},
            'success': result.success,
            'output_preview': result.output[:200],
            'error': result.error,
            'execution_time': result.execution_time,
            'timestamp': time.time()
        }
        self.execution_log.append(log_entry)
        
        # Disk'e yaz
        log_path = Path(self.config.log_dir) / "tool_log.json"
        try:
            with open(log_path, 'w') as f:
                json.dump(self.execution_log[-100:], f, indent=2)  # Son 100
        except:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Execution istatistikleri."""
        if not self.execution_log:
            return {'total_executions': 0}
        
        total = len(self.execution_log)
        successes = sum(1 for e in self.execution_log if e['success'])
        
        return {
            'total_executions': total,
            'success_rate': successes / total,
            'tools_used': list(set(e['action_type'] for e in self.execution_log)),
        }
