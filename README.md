# KKBox Churn Prediction — De la donnée à la décision

Ce projet prédit l'attrition (*churn*) des abonnés du service de streaming musical KKBOX. Il combine un pipeline de Machine Learning (LightGBM, Random Forest, Régression Logistique), une base de données relationnelle SQLite, un dashboard d'aide à la décision Power BI, ainsi qu'une démarche MLOps (tests pytest, linter ruff, conteneurisation Docker et CI GitHub Actions).

---

## 📁 Structure du projet

```text
kkbox-churn/
├── data/
│   ├── raw/                 # Fichiers sources CSV (exclus de git)
│   ├── processed/           # Datasets nettoyés et features
│   └── kkbox.db             # Base SQLite locale échantillonnée
├── sql/                     # Scripts et requêtes d'analyse métier
├── notebooks/               # Explorations et visualisations EDA
├── src/                     # Code source modulaire du projet
│   ├── __init__.py
│   └── 01_load_sql.py       # Pipeline d'échantillonnage et chargement SQLite
├── models/                  # Modèles entraînés et artefacts
├── dashboard/               # Fichiers Power BI (.pbix) et exports
├── tests/                   # Tests unitaires et d'intégration pytest
│   ├── __init__.py
│   └── test_load_sql.py
├── Dockerfile               # Conteneurisation du projet
├── .github/workflows/ci.yml # Intégration continue
├── pyproject.toml           # Configuration ruff, pytest et métadonnées
├── requirements.txt         # Dépendances Python
├── .gitignore               # Fichiers et dossiers exclus du suivi git
└── README.md
```

---

## 🛠️ Installation et environnement

L'environnement virtuel utilise **Python 3.11** sous `venv/`.

### 1. Activation de l'environnement virtuel

- **Sous Windows (PowerShell) :**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Sous Windows (CMD) :**
  ```cmd
  venv\Scripts\activate.bat
  ```
- **Sous Linux / macOS :**
  ```bash
  source venv/bin/activate
  ```

### 2. Installation des dépendances

```bash
pip install -r requirements.txt
```

---

## 🚀 Étape 1 : Chargement et échantillonnage SQLite

Pour charger l'échantillon représentatif de 50 000 membres et insérer les données filtrées dans `data/kkbox.db` sans saturer la mémoire vive :

```powershell
python src/01_load_sql.py
```

Le script :
1. Tire 50 000 `msno` dans `data/raw/train_v2.csv` avec la graine `seed = 42`.
2. Lit par morceaux (*chunks*) `members_v3.csv`, `transactions_v2.csv` et `user_logs_v2.csv` (1,43 Go) en ne conservant que ces 50 000 membres.
3. Insère les données dans `data/kkbox.db` (`train_labels`, `members`, `transactions`, `user_logs_sample`).
4. Crée un index sur la colonne `msno` de chaque table pour accélérer les futures jointures.
5. Affiche un bilan des volumétries et du nombre d'utilisateurs distincts.

---

## 🧪 Exécution des tests et contrôle qualité

Pour vérifier le bon fonctionnement des tests unitaires :
```powershell
pytest -v
```

Pour vérifier le style et la conformité du code :
```powershell
ruff check .
```
