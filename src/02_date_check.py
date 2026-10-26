"""Script d'analyse temporelle et de validation des dates (Étape 2).

Ce script se connecte en LECTURE SEULE à la base SQLite (`data/kkbox.db`)
et génère un audit complet de la chronologie des données :
1. Taux de churn global de l'échantillon.
2. Bornes temporelles (MIN / MAX) des transactions, expirations et logs.
3. Distribution des dates d'expiration par mois (AAAAMM) pour churn=0 vs churn=1.
4. Échantillon de trajectoires individuelles (10 non-churners, 10 churners) avec l'historique complet de leurs transactions.
5. Distribution de la date de dernière transaction par mois pour churn=0 vs churn=1.
6. Distribution du nombre de transactions par membre.
7. Impact sur le churn des données manquantes (membres sans profil, sans transaction, sans logs).

L'objectif est d'identifier la frontière temporelle étanche (CUTOFF_DATE)
pour éliminer tout risque de fuite de données (Data Leakage).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# Chemins
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "kkbox.db"
SEED = 42


def get_readonly_connection(db_path: Path) -> sqlite3.Connection:
    """Ouvre une connexion SQLite en lecture seule (sécurité garantie).

    Args:
        db_path: Chemin vers le fichier de base de données.

    Returns:
        Objet sqlite3.Connection configuré en lecture seule.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Base de données introuvable : {db_path}")
    # Ouverture en mode URI ro (Read-Only)
    uri = f"file:{db_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def format_date_int(d: int | str | None) -> str:
    """Formate un entier AAAAMMJJ en chaîne AAAA-MM-JJ."""
    if d is None:
        return "N/A"
    s = str(d).strip()
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def print_section(title: str, step_num: int) -> None:
    """Affiche un en-tête de section standardisé."""
    print("\n" + "=" * 80)
    print(f"[{step_num}/7] {title}")
    print("=" * 80)


def check_global_churn_rate(conn: sqlite3.Connection) -> None:
    """1. Affiche le taux de churn global de l'échantillon de 50 000 membres."""
    print_section("TAUX DE CHURN GLOBAL DE L'ÉCHANTILLON", 1)
    query = """
    SELECT
        COUNT(*) as total_membres,
        SUM(is_churn) as nb_churn,
        COUNT(*) - SUM(is_churn) as nb_fidele,
        ROUND(AVG(is_churn) * 100.0, 2) as churn_rate_pct
    FROM train_labels;
    """
    df = pd.read_sql_query(query, conn)
    total = int(df["total_membres"].iloc[0])
    churn = int(df["nb_churn"].iloc[0])
    fidele = int(df["nb_fidele"].iloc[0])
    pct = float(df["churn_rate_pct"].iloc[0])

    print(f"Total membres analysés   : {total:,}".replace(",", " "))
    print(f"Membres fidèles (churn=0): {fidele:,} ({100 - pct:.2f} %)".replace(",", " "))
    print(f"Membres perdus  (churn=1): {churn:,} ({pct:.2f} %)".replace(",", " "))
    print("-" * 50)
    print(f"-> Déséquilibre de classe : ~1 churner pour {(100 - pct) / pct:.1f} clients fidèles.")


def check_min_max_dates(conn: sqlite3.Connection) -> None:
    """2. Affiche les dates MIN et MAX dans chaque table temporelle."""
    print_section("BORNES TEMPORELLES (MIN ET MAX) DES DONNÉES", 2)

    query_tx = """
    SELECT
        MIN(transaction_date) as min_tx,
        MAX(transaction_date) as max_tx,
        MIN(membership_expire_date) as min_exp,
        MAX(membership_expire_date) as max_exp
    FROM transactions;
    """
    df_tx = pd.read_sql_query(query_tx, conn)

    query_logs = """
    SELECT
        MIN(date) as min_log,
        MAX(date) as max_log
    FROM user_logs_sample;
    """
    df_logs = pd.read_sql_query(query_logs, conn)

    query_members = """
    SELECT
        MIN(registration_init_time) as min_reg,
        MAX(registration_init_time) as max_reg
    FROM members;
    """
    df_members = pd.read_sql_query(query_members, conn)

    print(f"{'Champ temporel':<32} | {'Date Min':<12} | {'Date Max':<12}")
    print("-" * 62)
    print(f"{'Inscriptions (registration_init)':<32} | {format_date_int(df_members['min_reg'].iloc[0]):<12} | {format_date_int(df_members['max_reg'].iloc[0]):<12}")
    print(f"{'Transactions (transaction_date)':<32} | {format_date_int(df_tx['min_tx'].iloc[0]):<12} | {format_date_int(df_tx['max_tx'].iloc[0]):<12}")
    print(f"{'Expirations (membership_expire)':<32} | {format_date_int(df_tx['min_exp'].iloc[0]):<12} | {format_date_int(df_tx['max_exp'].iloc[0]):<12}")
    print(f"{'Logs écoute (date)':<32} | {format_date_int(df_logs['min_log'].iloc[0]):<12} | {format_date_int(df_logs['max_log'].iloc[0]):<12}")


def check_expire_date_distribution(conn: sqlite3.Connection) -> None:
    """3. Distribution de membership_expire_date par mois (AAAAMM) selon is_churn."""
    print_section("DISTRIBUTION DES DATES D'EXPIRATION PAR MOIS (AAAAMM)", 3)

    query = """
    WITH tx_with_labels AS (
        SELECT
            t.msno,
            substr(CAST(t.membership_expire_date AS TEXT), 1, 6) as expire_month,
            l.is_churn
        FROM transactions t
        JOIN train_labels l ON t.msno = l.msno
    )
    SELECT
        expire_month,
        SUM(CASE WHEN is_churn = 0 THEN 1 ELSE 0 END) as nb_tx_fidele,
        SUM(CASE WHEN is_churn = 1 THEN 1 ELSE 0 END) as nb_tx_churn,
        COUNT(*) as total_tx,
        ROUND(AVG(is_churn) * 100.0, 2) as churn_rate_pct
    FROM tx_with_labels
    GROUP BY expire_month
    ORDER BY expire_month;
    """
    df = pd.read_sql_query(query, conn)
    print(f"{'Mois Expire':<12} | {'Tx Fidele (0)':<14} | {'Tx Churn (1)':<14} | {'Total Tx':<10} | {'Taux Churn':<10}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(
            f"{row['expire_month']:<12} | "
            f"{int(row['nb_tx_fidele']):<14,d} | "
            f"{int(row['nb_tx_churn']):<14,d} | "
            f"{int(row['total_tx']):<10,d} | "
            f"{row['churn_rate_pct']:>6.2f} %".replace(",", " ")
        )


def check_sample_member_trajectories(conn: sqlite3.Connection) -> None:
    """4. Affiche l'historique de transactions pour 10 fidèles et 10 churners (seed fixe)."""
    print_section("TRAJECTOIRES DÉTAILLÉES : 10 NON-CHURNERS VS 10 CHURNERS", 4)

    # Récupération des msno ayant au moins 1 transaction
    query_candidates = """
    SELECT DISTINCT l.msno, l.is_churn
    FROM train_labels l
    JOIN transactions t ON l.msno = t.msno;
    """
    df_cand = pd.read_sql_query(query_candidates, conn)

    fideles = df_cand[df_cand["is_churn"] == 0]["msno"].to_numpy()
    churners = df_cand[df_cand["is_churn"] == 1]["msno"].to_numpy()

    rng = np.random.default_rng(SEED)
    sample_fideles = rng.choice(fideles, size=min(10, len(fideles)), replace=False)
    sample_churners = rng.choice(churners, size=min(10, len(churners)), replace=False)

    def print_trajectories_for_members(msno_list: np.ndarray, label_desc: str) -> None:
        print(f"\n--- Échantillon de {len(msno_list)} membres : {label_desc} ---")
        for i, msno in enumerate(msno_list, 1):
            query_m = f"""
            SELECT transaction_date, membership_expire_date, payment_plan_days,
                   plan_list_price, actual_amount_paid, is_auto_renew, is_cancel
            FROM transactions
            WHERE msno = '{msno}'
            ORDER BY transaction_date ASC;
            """
            txs = pd.read_sql_query(query_m, conn)
            short_id = msno[:12] + "..."
            print(f"\n[{i:02d}] Membre {short_id} ({len(txs)} transaction(s)) :")
            for _, r in txs.iterrows():
                print(
                    f"    Tx Date: {format_date_int(r['transaction_date'])} | "
                    f"Expire Date: {format_date_int(r['membership_expire_date'])} | "
                    f"Jours: {r['payment_plan_days']:>3} | "
                    f"Auto-Renew: {r['is_auto_renew']} | "
                    f"Cancel: {r['is_cancel']} | "
                    f"Montant: {r['actual_amount_paid']:>3}"
                )

    print_trajectories_for_members(sample_fideles, "FIDÈLES (is_churn = 0)")
    print_trajectories_for_members(sample_churners, "CHURNERS (is_churn = 1)")


def check_last_transaction_distribution(conn: sqlite3.Connection) -> None:
    """5. Distribution de la date de la DERNIÈRE transaction par membre, par mois."""
    print_section("DATE DE LA DERNIÈRE TRANSACTION PAR MEMBRE (PAR MOIS)", 5)

    query = """
    WITH last_tx_per_member AS (
        SELECT
            t.msno,
            l.is_churn,
            substr(CAST(MAX(t.transaction_date) AS TEXT), 1, 6) as last_tx_month
        FROM transactions t
        JOIN train_labels l ON t.msno = l.msno
        GROUP BY t.msno, l.is_churn
    )
    SELECT
        last_tx_month,
        SUM(CASE WHEN is_churn = 0 THEN 1 ELSE 0 END) as nb_fidele,
        SUM(CASE WHEN is_churn = 1 THEN 1 ELSE 0 END) as nb_churn,
        COUNT(*) as total_membres,
        ROUND(AVG(is_churn) * 100.0, 2) as churn_rate_pct
    FROM last_tx_per_member
    GROUP BY last_tx_month
    ORDER BY last_tx_month;
    """
    df = pd.read_sql_query(query, conn)
    print(f"{'Dernier Mois':<12} | {'Membres Fidele':<16} | {'Membres Churn':<14} | {'Total Membres':<14} | {'Taux Churn':<10}")
    print("-" * 75)
    for _, row in df.iterrows():
        print(
            f"{row['last_tx_month']:<12} | "
            f"{int(row['nb_fidele']):<16,d} | "
            f"{int(row['nb_churn']):<14,d} | "
            f"{int(row['total_membres']):<14,d} | "
            f"{row['churn_rate_pct']:>6.2f} %".replace(",", " ")
        )


def check_transaction_count_distribution(conn: sqlite3.Connection) -> None:
    """6. Distribution du nombre de transactions par membre (0, 1, 2, 3+)."""
    print_section("DISTRIBUTION DU NOMBRE DE TRANSACTIONS PAR MEMBRE", 6)

    query = """
    WITH tx_counts AS (
        SELECT
            l.msno,
            l.is_churn,
            COUNT(t.transaction_date) as nb_tx
        FROM train_labels l
        LEFT JOIN transactions t ON l.msno = t.msno
        GROUP BY l.msno, l.is_churn
    )
    SELECT
        CASE
            WHEN nb_tx = 0 THEN '0 transaction'
            WHEN nb_tx = 1 THEN '1 transaction'
            WHEN nb_tx = 2 THEN '2 transactions'
            ELSE '3 transactions ou +'
        END as categorie,
        SUM(CASE WHEN is_churn = 0 THEN 1 ELSE 0 END) as nb_fidele,
        SUM(CASE WHEN is_churn = 1 THEN 1 ELSE 0 END) as nb_churn,
        COUNT(*) as total_membres,
        ROUND(AVG(is_churn) * 100.0, 2) as churn_rate_pct,
        MIN(nb_tx) as sort_key
    FROM tx_counts
    GROUP BY categorie
    ORDER BY sort_key ASC;
    """
    df = pd.read_sql_query(query, conn)
    print(f"{'Catégorie':<20} | {'Fidèles (0)':<12} | {'Churners (1)':<12} | {'Total':<10} | {'Taux Churn':<10}")
    print("-" * 72)
    for _, row in df.iterrows():
        print(
            f"{row['categorie']:<20} | "
            f"{int(row['nb_fidele']):<12,d} | "
            f"{int(row['nb_churn']):<12,d} | "
            f"{int(row['total_membres']):<10,d} | "
            f"{row['churn_rate_pct']:>6.2f} %".replace(",", " ")
        )


def check_churn_by_data_presence(conn: sqlite3.Connection) -> None:
    """7. Compare le taux de churn selon la présence ou l'absence de profil, transaction et logs."""
    print_section("IMPACT SUR LE CHURN DES DONNÉES MANQUANTES (PROFIL / TX / LOGS)", 7)

    query = """
    SELECT
        'Tous membres (Référence globale)' as groupe,
        COUNT(*) as total_membres,
        ROUND(AVG(is_churn) * 100.0, 2) as churn_rate_pct
    FROM train_labels

    UNION ALL

    SELECT
        'SANS profil dans members' as groupe,
        COUNT(*) as total_membres,
        ROUND(AVG(l.is_churn) * 100.0, 2) as churn_rate_pct
    FROM train_labels l
    LEFT JOIN members m ON l.msno = m.msno
    WHERE m.msno IS NULL

    UNION ALL

    SELECT
        'AVEC profil dans members' as groupe,
        COUNT(*) as total_membres,
        ROUND(AVG(l.is_churn) * 100.0, 2) as churn_rate_pct
    FROM train_labels l
    INNER JOIN members m ON l.msno = m.msno

    UNION ALL

    SELECT
        'SANS aucune transaction' as groupe,
        COUNT(*) as total_membres,
        ROUND(AVG(l.is_churn) * 100.0, 2) as churn_rate_pct
    FROM train_labels l
    LEFT JOIN (SELECT DISTINCT msno FROM transactions) t ON l.msno = t.msno
    WHERE t.msno IS NULL

    UNION ALL

    SELECT
        'AVEC au moins 1 transaction' as groupe,
        COUNT(*) as total_membres,
        ROUND(AVG(l.is_churn) * 100.0, 2) as churn_rate_pct
    FROM train_labels l
    INNER JOIN (SELECT DISTINCT msno FROM transactions) t ON l.msno = t.msno

    UNION ALL

    SELECT
        'SANS aucun log écoute' as groupe,
        COUNT(*) as total_membres,
        ROUND(AVG(l.is_churn) * 100.0, 2) as churn_rate_pct
    FROM train_labels l
    LEFT JOIN (SELECT DISTINCT msno FROM user_logs_sample) u ON l.msno = u.msno
    WHERE u.msno IS NULL

    UNION ALL

    SELECT
        'AVEC logs écoute' as groupe,
        COUNT(*) as total_membres,
        ROUND(AVG(l.is_churn) * 100.0, 2) as churn_rate_pct
    FROM train_labels l
    INNER JOIN (SELECT DISTINCT msno FROM user_logs_sample) u ON l.msno = u.msno;
    """
    df = pd.read_sql_query(query, conn)
    print(f"{'Groupe analysé':<35} | {'Nombre de membres':<18} | {'Taux Churn':<10}")
    print("-" * 68)
    for _, row in df.iterrows():
        print(
            f"{row['groupe']:<35} | "
            f"{int(row['total_membres']):<18,d} | "
            f"{row['churn_rate_pct']:>6.2f} %".replace(",", " ")
        )
    print("-" * 68)


def main() -> None:
    """Exécute les 7 volets d'audit temporel sur la base SQLite."""
    conn = get_readonly_connection(DB_PATH)
    try:
        check_global_churn_rate(conn)
        check_min_max_dates(conn)
        check_expire_date_distribution(conn)
        check_sample_member_trajectories(conn)
        check_last_transaction_distribution(conn)
        check_transaction_count_distribution(conn)
        check_churn_by_data_presence(conn)
        print("\n" + "=" * 80)
        print("Fin de l'audit temporel. Données prêtes pour l'analyse de la date de coupure.")
        print("=" * 80 + "\n")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
