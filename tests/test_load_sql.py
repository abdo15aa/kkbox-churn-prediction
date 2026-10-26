"""Tests unitaires pour le pipeline d'échantillonnage et de chargement SQLite."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parent.parent / "src" / "01_load_sql.py"
spec = importlib.util.spec_from_file_location("load_sql_module", MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Impossible de charger le module à l'adresse {MODULE_PATH}")
load_sql_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_sql_module)

configure_sqlite = load_sql_module.configure_sqlite
create_database_indexes = load_sql_module.create_database_indexes
extract_sample_msno = load_sql_module.extract_sample_msno
stream_csv_to_sqlite = load_sql_module.stream_csv_to_sqlite




@pytest.fixture
def mock_csv_data(tmp_path: Path) -> tuple[Path, Path]:
    """Crée des fichiers CSV miniatures pour tester la logique de streaming et filtrage."""
    train_file = tmp_path / "mock_train.csv"
    train_df = pd.DataFrame(
        {
            "msno": [f"user_{i}" for i in range(100)],
            "is_churn": [i % 2 for i in range(100)],
        }
    )
    train_df.to_csv(train_file, index=False)

    members_file = tmp_path / "mock_members.csv"
    members_df = pd.DataFrame(
        {
            "msno": [f"user_{i}" for i in range(200)],
            "city": [1] * 200,
            "bd": [25] * 200,
        }
    )
    members_df.to_csv(members_file, index=False)

    return train_file, members_file


def test_extract_sample_msno_reproducibility(mock_csv_data: tuple[Path, Path]) -> None:
    """Vérifie que l'échantillonnage est strictement déterministe avec la même seed."""
    train_file, _ = mock_csv_data

    sample_df1, msnos1 = extract_sample_msno(train_file, sample_size=20, seed=42, chunk_size=30)
    sample_df2, msnos2 = extract_sample_msno(train_file, sample_size=20, seed=42, chunk_size=30)

    assert len(sample_df1) == 20
    assert len(msnos1) == 20
    assert msnos1 == msnos2
    pd.testing.assert_frame_equal(sample_df1, sample_df2)


def test_stream_csv_to_sqlite(tmp_path: Path, mock_csv_data: tuple[Path, Path]) -> None:
    """Vérifie que le filtrage par chunks n'insère que les membres cibles."""
    train_file, members_file = mock_csv_data
    db_file = tmp_path / "test.db"

    conn = sqlite3.connect(db_file)
    configure_sqlite(conn)

    _, target_msnos = extract_sample_msno(train_file, sample_size=15, seed=42, chunk_size=25)
    inserted_count = stream_csv_to_sqlite(
        csv_path=members_file,
        table_name="members",
        conn=conn,
        target_msnos=target_msnos,
        chunk_size=30,
    )

    assert inserted_count == 15

    # Vérification en base SQL
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), COUNT(DISTINCT msno) FROM members;")
    rows, distinct = cursor.fetchone()
    assert rows == 15
    assert distinct == 15

    # Test création d'index sur table existante
    create_database_indexes(conn)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_members_msno';")
    assert cursor.fetchone() is not None

    conn.close()
