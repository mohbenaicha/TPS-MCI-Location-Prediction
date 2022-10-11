import logging
import sys
import os
from types import FrameType
from typing import List, cast

from loguru import logger
from pydantic import AnyHttpUrl, BaseSettings


# contains key api and logging settings
class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO  # logging levels are type int

class DB_Settings(BaseSettings):
    # Unlike, the model package, the api will read some some configuration
    # specs from env variables from .env that are set in docker 
    # compose yaml/ tox ini files with sensistive config kept in a secure store

    SQLALCH_DB_URI = (f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
                       f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/mci-app-db")
    REDIS_URI = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"

# export DB_HOST='localhost' && export DB_USER='postgres' && export DB_PASSWORD='password' && export DB_PORT='30006' && export APP_PORT='8001' && export REDIS_HOST='localhost' && export REDIS_PORT='6379'

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    APP_PORT: int = os.getenv('APP_PORT')
    PROM_METRICS_PORT: int = os.getenv('PROM_METRICS_PORT')
    # Meta
    logging: LoggingSettings = LoggingSettings()
    db_settings: DB_Settings = DB_Settings()

    # BACKEND_CORS_ORIGINS is a comma-separated list of origins
    # e.g: http://localhost,http://localhost:4200,http://localhost:3000
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # type: ignore
        "http://localhost:8000",  # type: ignore
        "https://localhost:3000",  # type: ignore
        "https://localhost:8000",  # type: ignore
    ]

    APP_NAME: str = "MCI Prediction API"

    class Config:
        case_sensitive = True


class InterceptHandler(logging.Handler):
    """
    intercepts logging logs for each uvicorn webserver logger
    """

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


def setup_app_logging(config: Settings) -> None:
    """Prepare custom logging for our application."""

    LOGGERS = ("uvicorn.asgi", "uvicorn.access")
    logging.getLogger().handlers = [InterceptHandler()]
    for logger_name in LOGGERS:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(level=config.logging.LOGGING_LEVEL)]

    handler = [{"sink": sys.stderr, "level": config.logging.LOGGING_LEVEL}]
    logger.configure(handlers=handler)


settings = Settings()
