"""Runtime configuration, read from environment variables (and an optional .env file).

Nothing secret is hard-coded; see `.env.example` at the repository root.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Minimal .env loader (KEY=VALUE lines). Real environment variables win."""
    for candidate in (BACKEND_DIR / ".env", BACKEND_DIR.parent / ".env"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """All tunables in one place. Instantiate via `get_settings()`."""

    data_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("REPROSCOPE_DATA_DIR", BACKEND_DIR / "data")).resolve())
    max_pdf_mb: int = field(default_factory=lambda: _int("REPROSCOPE_MAX_PDF_MB", 30))
    max_zip_mb: int = field(default_factory=lambda: _int("REPROSCOPE_MAX_ZIP_MB", 100))
    max_repo_unpacked_mb: int = field(default_factory=lambda: _int("REPROSCOPE_MAX_REPO_MB", 500))
    git_clone_timeout_s: int = field(default_factory=lambda: _int("REPROSCOPE_GIT_TIMEOUT_S", 120))
    allowed_git_hosts: tuple[str, ...] = field(default_factory=lambda: tuple(
        h.strip() for h in os.environ.get(
            "REPROSCOPE_ALLOWED_GIT_HOSTS", "github.com,gitlab.com,bitbucket.org").split(",") if h.strip()))

    # LLM: provider is configurable. "groq" and "openai_compatible" speak the OpenAI
    # chat-completions protocol; without a key the app runs in labelled "dev" mode.
    llm_provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "groq").lower())
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"))
    llm_base_url: str = field(default_factory=lambda: os.environ.get(
        "LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"))
    llm_api_key: str | None = field(default_factory=lambda: (
        os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY") or None))
    # Strict JSON-schema decoding. Groq supports it for openai/gpt-oss-* models; set
    # false for models that only do best-effort JSON (responses are still validated).
    llm_strict_json: bool = field(default_factory=lambda: os.environ.get(
        "LLM_STRICT_JSON", "true").lower() in ("1", "true", "yes"))
    llm_timeout_s: int = field(default_factory=lambda: _int("LLM_TIMEOUT_S", 120))
    llm_max_retries: int = field(default_factory=lambda: _int("LLM_MAX_RETRIES", 4))

    # Sandbox
    sandbox_cpus: float = field(default_factory=lambda: float(os.environ.get("SANDBOX_CPUS", "2")))
    sandbox_memory: str = field(default_factory=lambda: os.environ.get("SANDBOX_MEMORY", "4g"))
    sandbox_timeout_s: int = field(default_factory=lambda: _int("SANDBOX_TIMEOUT_S", 600))
    sandbox_python: str = field(default_factory=lambda: os.environ.get("SANDBOX_PYTHON", "3.11"))

    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o.strip() for o in os.environ.get(
            "REPROSCOPE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "reproscope.db"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def llm_mode(self) -> str:
        """'live' when a real provider is configured, otherwise 'dev' (sample outputs)."""
        if self.llm_provider in ("groq", "openai_compatible") and self.llm_api_key:
            return "live"
        return "dev"

    @property
    def llm_cache_dir(self) -> Path:
        return self.data_dir / "llm_cache"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return _settings


def override_settings(settings: Settings) -> None:
    """Used by tests to point the app at a temporary data directory."""
    global _settings
    _settings = settings
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
