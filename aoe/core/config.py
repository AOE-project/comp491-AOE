"""
core/config.py — Loads environment variables and exposes a typed Settings object
plus a factory for the LangChain chat model.
All modules read configuration and obtain the LLM from here.
"""

from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file (aoe/.env), not the CWD
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-5.4-nano"
    gurobi_license_path: str = ""
    log_level: str = "INFO"
    log_output_dir: str = "./logs"
    use_dummy_analyser: bool = False
    use_dummy_code_generator: bool = False
    use_dummy_regeneration: bool = False

    # Error testing (debug flow validation)
    test_error_injection: bool = False
    test_error_type: str = "syntax_error"  # syntax_error | runtime_error | modeling_error | unknown_error

    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), env_file_encoding="utf-8")


def load_settings() -> Settings:
    return Settings()


def get_llm_client() -> ChatOpenAI:
    settings = load_settings()
    return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)