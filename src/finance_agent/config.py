from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    sec_user_agent: str = "FinanceAgent contact@example.com"
    chroma_persist_dir: str = ".chroma"
    log_level: str = "INFO"


settings = Settings()
