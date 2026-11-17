"""Veritabanı bağlantı, şema oluşturma ve toplu yükleme yönetimi.

Varsayılan arka uç SQLite'tır (WAL modu + foreign key zorlaması). URL değiştirilerek
PostgreSQL vb. kullanılabilir; bu modül dışında hiçbir yer engine oluşturmamalıdır.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from sqlalchemy import Engine, create_engine, event, insert
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings
from src.database.models import Account, AccountDevice, Alert, Base, Device, Investigation, Transaction

# Tüm ORM tablolarının `Base.metadata.create_all` içinde kayıtlı olduğundan emin ol
_ORM_MODELS = (Alert, Investigation)

if TYPE_CHECKING:
    from src.generator.synthetic_data import SyntheticDataset

logger = logging.getLogger(__name__)

# Foreign key bağımlılık sırası (yükleme bu sırayla yapılır)
TABLE_LOAD_ORDER: tuple[tuple[type[Base], str], ...] = (
    (Account, "accounts"),
    (Device, "devices"),
    (AccountDevice, "account_devices"),
    (Transaction, "transactions"),
)


def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def create_db_engine(url: str | None = None, *, echo: bool | None = None) -> Engine:
    """Yeni bir Engine oluşturur. SQLite dosyasının klasörü yoksa yaratılır."""
    settings = get_settings()
    url = url or settings.database.sqlalchemy_url
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite" and parsed.database not in (None, "", ":memory:"):
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url, echo=settings.database.echo if echo is None else echo)
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _sqlite_pragmas)
    return engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Uygulama genelinde paylaşılan varsayılan Engine."""
    return create_db_engine()


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Commit/rollback'i otomatik yöneten oturum bağlamı."""
    factory = sessionmaker(bind=engine or get_engine(), expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine | None = None, *, drop_existing: bool = False) -> Engine:
    """Tüm tabloları oluşturur (idempotent). `drop_existing=True` önce siler."""
    engine = engine or get_engine()
    if drop_existing:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    logger.info("Şema hazır: %s", ", ".join(sorted(Base.metadata.tables)))
    return engine


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame'i NaN/NaT yerine None içeren saf Python kayıtlarına çevirir."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def persist_dataset(
    dataset: SyntheticDataset, engine: Engine | None = None, *, chunk_size: int = 5_000
) -> dict[str, int]:
    """Üretilen veri setini tek bir transaction içinde, FK sırasına uygun yükler."""
    engine = engine or get_engine()
    counts: dict[str, int] = {}
    with engine.begin() as conn:
        for model, attr in TABLE_LOAD_ORDER:
            records = dataframe_to_records(getattr(dataset, attr))
            for start in range(0, len(records), chunk_size):
                conn.execute(insert(model), records[start : start + chunk_size])
            counts[model.__tablename__] = len(records)
            logger.info("%s: %d satır yüklendi", model.__tablename__, len(records))
    return counts


def load_table(table: str, engine: Engine | None = None) -> pd.DataFrame:
    """Bir tabloyu DataFrame olarak okur"""
    if table not in Base.metadata.tables:
        raise KeyError(f"Bilinmeyen tablo: {table}")
    return pd.read_sql_table(table, engine or get_engine())
