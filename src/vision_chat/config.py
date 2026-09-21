import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # llama-server (OpenAI-compatible) connection
    LLAMA_BASE_URL: str = "http://127.0.0.1:8081/v1"
    LLAMA_API_KEY: str = "sk-no-key-required"
    MODEL_NAME: str = "Qwen3-VL-2B-Instruct"
    MAX_TOKENS: int = 256

    # LangSmith tracing (optional; disabled unless both tracing and key are set)
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "vision-chat"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # .env also carries Makefile-only variables (HOST, PORT, ...).
        extra="ignore",
    )

    def apply_langsmith_env(self) -> None:
        """Expose LangSmith settings as real environment variables.

        langchain-core reads LANGSMITH_* from os.environ (with an internal
        cache on first use), so values from .env must be exported before the
        first chain call. Already-set environment variables win, and tracing
        stays off unless both LANGSMITH_TRACING=true and an API key exist.
        """
        if os.environ.get("LANGSMITH_TRACING") is None:
            if self.LANGSMITH_TRACING and self.LANGSMITH_API_KEY:
                os.environ["LANGSMITH_TRACING"] = "true"
            else:
                os.environ["LANGSMITH_TRACING"] = "false"

        for field in ("LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"):
            if os.environ.get(field) is None and getattr(self, field):
                os.environ[field] = getattr(self, field)


settings = Settings()
settings.apply_langsmith_env()
