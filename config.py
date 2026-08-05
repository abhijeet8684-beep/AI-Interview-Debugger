"""Project configuration and constants.

This module centralizes filesystem paths and configuration defaults used across
the project. Keep it lightweight and avoid environment-specific side effects.

Design decisions
- Use pathlib.Path for path manipulation (cross-platform)
- Keep values as plain dataclasses / constants so importing this module is cheap
  and safe in tests.
- If needed later, replace the Config dataclass with pydantic.BaseSettings to
  support env-var overrides.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Final

ROOT: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"

# Deterministic diagnostic thresholds used by the rule engine.
LLM_TIMEOUT_MS: Final[float] = 5_000.0
TOOL_TIMEOUT_MS: Final[float] = 5_000.0
MAX_RETRIES: Final[int] = 2
MAX_TOTAL_TOKENS: Final[int] = 10_000
LONG_INTERVIEW_SECONDS: Final[float] = 1_800.0
MIN_TRANSCRIPT_TURNS: Final[int] = 4


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration.

    Keep defaults here. For production use, consider loading overridable values
    from environment variables using pydantic.BaseSettings.
    """

    project_name: str = "AI-Interview-Debugger"
    root_dir: Path = ROOT
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    random_seed: int = 42


DEFAULT_CFG = Config()


if __name__ == "__main__":
    # Quick sanity check when running directly
    print(f"Project root: {ROOT}")
    print(f"Raw data dir: {RAW_DATA_DIR}")
