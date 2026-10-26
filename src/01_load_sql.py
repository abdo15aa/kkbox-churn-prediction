"""Pipeline de chargement et d'échantillonnage KKBox vers SQLite.

Ce script extrait un échantillon aléatoire reproductible de 50 000 membres (msno)
depuis `train_v2.csv` (graine 42) et charge l'ensemble des données associées
depuis les fichiers bruts vers une base de données relationnelle SQLite locale (`kkbox.db`).

Contraintes de conception :
- Échantillonnage représentatif fixé à 50 000 msno (seed=42).
- Aucun fichier complet n'est chargé en mémoire vive : lecture stricte par chunks (morceaux).
- Filtrage en mémoire au vol (streaming) avant toute écriture en base.
- Optimisations SQLite (WAL, cache mémoire, création des index après insertion).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration de la journalisation (logs)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kkbox_loader")

# Paramètres globaux du projet
SEED: int = 42
SAMPLE_SIZE: int = 50_000
CHUNK_SIZE: int = 100_000

# Chemins absolus dérivés du chemin du script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "kkbox.db"

TRAIN_CSV = DATA_RAW_DIR / "train_v2.csv"
MEMBERS_CSV = DATA_RAW_DIR / "members_v3.csv"
TRANSACTIONS_CSV = DATA_RAW_DIR / "transactions_v2.csv"
USER_LOGS_CSV = DATA_RAW_DIR / "user_logs_v2.csv"


def configure_sqlite(conn: sqlite3.Connection) -> None:
    """Configure les paramètres de performance SQLite pour les opérations de chargement massif.

    Args:
        conn: Connexion SQLite active.
    """
    logger.info("Application des optimisations PRAGMA pour SQLite...")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -64000;")  # Alloue 64 Mo de cache mémoire


def extract_sample_msno(
    train_path: Path,
    sample_size: int = SAMPLE_SIZE,
    seed: int = SEED,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[pd.DataFrame, set[str]]:
    """Tire un échantillon uniforme de 50 000 membres depuis train_v2 sans charger le fichier complet.

    Méthode :
    1. Premier passage léger pour compter le nombre total d'enregistrements N.
    2. Tirage sans remise de 50 000 indices de lignes (0 <= idx < N) avec le générateur aléatoire fixé (seed=42).
    3. Deuxième passage en lecture par morceaux (chunks) : seules les lignes sélectionnées
       sont conservées en mémoire.

    Args:
        train_path: Chemin vers le fichier train_v2.csv.
        sample_size: Nombre de membres à échantillonner (par défaut 50 000).
        seed: Graine aléatoire garantissant la stricte reproductibilité.
        chunk_size: Taille de chaque bloc de lecture en nombre de lignes.

    Returns:
        Un tuple contenant :
        - DataFrame contenant les 50 000 labels (msno, is_churn).
        - Set des 50 000 msno pour des filtres instantanés O(1).
    """
    if not train_path.exists():
        raise FileNotFoundError(f"Le fichier source {train_path} est introuvable.")

    logger.info("Comptage des lignes de %s pour un tirage uniforme...", train_path.name)
    total_rows = 0
    with open(train_path, encoding="utf-8") as f:
        next(f)  # Ignore l'en-tête (msno, is_churn)
        for _ in f:
            total_rows += 1

    logger.info("Total d'enregistrements dans %s : %d", train_path.name, total_rows)
    if total_rows < sample_size:
        raise ValueError(f"Taille de fichier insuffisante ({total_rows} < {sample_size}).")

    rng = np.random.default_rng(seed)
    sampled_indices = set(rng.choice(total_rows, size=sample_size, replace=False))

    logger.info("Extraction par chunks des %d membres échantillonnés...", sample_size)
    sampled_dfs: list[pd.DataFrame] = []
    current_index = 0

    for chunk in pd.read_csv(train_path, chunksize=chunk_size):
        chunk_len = len(chunk)
        chunk_indices = np.arange(current_index, current_index + chunk_len)
        mask = [idx in sampled_indices for idx in chunk_indices]
        if any(mask):
            sampled_dfs.append(chunk[mask])
        current_index += chunk_len

    sample_df = pd.concat(sampled_dfs, ignore_index=True)
    sample_msnos = set(sample_df["msno"].unique())

    logger.info("Échantillon final : %d msno uniques et distincts obtenus.", len(sample_msnos))
    return sample_df, sample_msnos


def stream_csv_to_sqlite(
    csv_path: Path,
    table_name: str,
    conn: sqlite3.Connection,
    target_msnos: set[str],
    chunk_size: int = CHUNK_SIZE,
) -> int:
    """Lit un fichier CSV volumineux par chunks, filtre sur target_msnos et insère dans SQLite.

    Args:
        csv_path: Chemin du fichier CSV source.
        table_name: Nom de la table cible dans SQLite.
        conn: Connexion SQLite active.
        target_msnos: Ensemble des identifiants membres à conserver.
        chunk_size: Nombre de lignes lues à chaque itération.

    Returns:
        Nombre total de lignes insérées dans la table SQLite.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    logger.info("Réinitialisation de la table '%s'...", table_name)
    conn.execute(f"DROP TABLE IF EXISTS {table_name};")

    total_inserted = 0
    logger.info("Traitement par morceaux de %s vers la table '%s'...", csv_path.name, table_name)

    with tqdm(desc=f"Chargement {table_name}", unit=" chunks") as pbar:
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
            # Filtrage vectorisé O(1) par élément via le set Python
            filtered_chunk = chunk[chunk["msno"].isin(target_msnos)]
            if not filtered_chunk.empty:
                filtered_chunk.to_sql(
                    table_name,
                    conn,
                    if_exists="append",
                    index=False,
                )
                total_inserted += len(filtered_chunk)
            pbar.update(1)

    logger.info("Table '%s' complétée : %d lignes insérées.", table_name, total_inserted)
    return total_inserted


def create_database_indexes(conn: sqlite3.Connection) -> None:
    """Crée les index sur la colonne 'msno' dans chaque table présente.

    Note pédagogique : La création des index s'effectue TOUJOURS après les insertions massives.
    Insérer des millions de lignes dans une table déjà indexée oblige le moteur à rééquilibrer
    l'arbre B-tree à chaque écriture, multipliant le temps d'exécution par 5 à 10.
    """
    tables = ["train_labels", "members", "transactions", "user_logs_sample"]
    logger.info("Création des index sur la colonne 'msno' pour accélérer les jointures SQL...")

    cursor = conn.cursor()
    for table in tables:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
        if not cursor.fetchone():
            logger.warning("Table '%s' absente, création de l'index ignorée.", table)
            continue

        t0 = time.time()
        index_name = f"idx_{table}_msno"
        logger.info("Création de l'index %s sur %s(msno)...", index_name, table)
        conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}(msno);")
        logger.info("Index %s créé en %.2f secondes.", index_name, time.time() - t0)


def display_database_summary(conn: sqlite3.Connection) -> None:
    """Affiche un récapitulatif clair du volume de données et du nombre de membres distincts."""
    tables = ["train_labels", "members", "transactions", "user_logs_sample"]

    print("\n" + "=" * 70)
    print("           RÉCAPITULATIF DE LA BASE SQLITE KKBOX (kkbox.db)")
    print("=" * 70)
    print(f"{'Table':<20} | {'Nombre de lignes':<20} | {'msno distincts':<20}")
    print("-" * 70)

    cursor = conn.cursor()
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        row_count = cursor.fetchone()[0]

        cursor.execute(f"SELECT COUNT(DISTINCT msno) FROM {table};")
        distinct_msno = cursor.fetchone()[0]

        print(f"{table:<20} | {row_count:<20,d} | {distinct_msno:<20,d}".replace(",", " "))

    print("=" * 70 + "\n")


def main() -> None:
    """Point d'entrée principal du script d'initialisation et de chargement."""
    start_total_time = time.time()
    logger.info("Démarrage du processus de chargement SQLite KKBOX...")

    # S'assurer que le dossier parent data/ existe
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Connexion à SQLite
    conn = sqlite3.connect(DB_PATH)

    try:
        configure_sqlite(conn)

        # 1. Échantillon 50 000 msno depuis train_v2.csv
        sample_df, sample_msnos = extract_sample_msno(
            train_path=TRAIN_CSV,
            sample_size=SAMPLE_SIZE,
            seed=SEED,
            chunk_size=CHUNK_SIZE,
        )

        # Insertion de train_labels
        logger.info("Insertion des 50 000 étiquettes dans 'train_labels'...")
        conn.execute("DROP TABLE IF EXISTS train_labels;")
        sample_df.to_sql("train_labels", conn, if_exists="replace", index=False)

        # 2. Chargement filtré de members_v3.csv
        stream_csv_to_sqlite(
            csv_path=MEMBERS_CSV,
            table_name="members",
            conn=conn,
            target_msnos=sample_msnos,
            chunk_size=CHUNK_SIZE,
        )

        # 3. Chargement filtré de transactions_v2.csv
        stream_csv_to_sqlite(
            csv_path=TRANSACTIONS_CSV,
            table_name="transactions",
            conn=conn,
            target_msnos=sample_msnos,
            chunk_size=CHUNK_SIZE,
        )

        # 4. Chargement filtré de user_logs_v2.csv (fichier volumineux de ~1.43 Go)
        # On peut utiliser des chunks de 200 000 lignes pour accélérer la boucle
        stream_csv_to_sqlite(
            csv_path=USER_LOGS_CSV,
            table_name="user_logs_sample",
            conn=conn,
            target_msnos=sample_msnos,
            chunk_size=200_000,
        )

        # 5. Création des index
        create_database_indexes(conn)

        # 6. Affichage du bilan de la base
        display_database_summary(conn)

        elapsed = time.time() - start_total_time
        logger.info("Traitement complet achevé avec succès en %.1f secondes (%.1f min).", elapsed, elapsed / 60)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
