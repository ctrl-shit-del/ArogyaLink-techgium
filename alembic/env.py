import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy import create_engine
from alembic import context

from backend.config.database import Base
from backend.models.orm import PatientORM, VitalsHistoryORM, AlertEventORM

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

database_url = os.getenv("DATABASE_URL", "postgresql://synera:synera_dev@localhost:5432/synera")
if "+asyncpg" in database_url:
    database_url = database_url.replace("+asyncpg", "")
config.set_main_option("sqlalchemy.url", database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = config.get_main_option("sqlalchemy.url")
    connectable = create_engine(url, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
