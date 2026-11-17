# KKBox Churn Prediction — De la donnée à la décision

Ce projet industriel prédit l'attrition (*churn*) des abonnés du service de streaming musical KKBOX. Il implémente un pipeline de Machine Learning de bout en bout : staging relationnel multi-cohortes étanche, feature engineering sans fuite temporelle, modélisation supervisée (Régression Logistique, LightGBM, XGBoost), interprétabilité SHAP, simulation financière de rétention et scoring de production par segmentation de risque.

---

## 📁 Structure du projet

```text
kkbox-churn-prediction-challenge/
├── data/
│   ├── raw/                           # Fichiers sources CSV bruts (exclus de git)
│   ├── features/                      # Datasets de features calculées (cohort_a, cohort_b)
│   ├── dashboard/                     # Tables prêtes pour Power BI / Dashboard
│   └── kkbox_cohorts_staging.db       # Base SQLite de staging multi-cohortes (4.02 Go)
├── models/                            # Modèles sérialisés et préprocesseurs (.joblib)
│   ├── baseline_logistic_regression.joblib
│   ├── churn_preprocessor.joblib
│   ├── lightgbm_churn_model.joblib
│   ├── xgboost_churn_model.joblib     # Modèle champion (ROC-AUC = 0.9017)
│   └── model_card.md                  # Fiche descriptive du modèle
├── outputs/                           # Fichiers d'inférence de production
│   └── customer_churn_scores.csv      # Scores (50 000 abonnés) et segments de risque
├── dashboard/                         # Guide de connexion et intégration Power BI
│   └── POWERBI_GUIDE.md
├── app/                               # Tableau de bord interactif de visualisation
│   └── dashboard.py
├── reports/                           # Rapports techniques, audits et métriques
│   ├── figures/                       # Graphiques SHAP, courbes ROC, PR et Lift
│   ├── final_report.md                # Rapport de synthèse final
│   └── stage[1-6]_*.md / *.json
├── src/                               # Code source du pipeline de production
│   ├── __init__.py
│   ├── 06_build_staging_db.py         # Étape 1 : Construction base SQLite de staging (v1 + v2)
│   ├── 07_build_features.py           # Étape 2 : Feature engineering point-in-time étanche (34 features)
│   ├── 08_eda.py                      # Étape 3A : Analyse exploratoire des données (EDA)
│   ├── 08_train_baseline.py           # Étape 3B : Entraînement baseline Régression Logistique
│   ├── 09_train_advanced_models.py    # Étape 4 : Entraînement LightGBM & XGBoost + optimisation seuil
│   ├── 10_evaluate_oot.py             # Étape 5A : Évaluation indépendante hors-temps (OOT Cohorte B)
│   ├── 10_model_interpretability.py   # Étape 5B : Interprétabilité globale et locale SHAP
│   ├── 11_business_evaluation.py      # Étape 5C : Simulation ROI de rétention et courbes de gains
│   └── 12_production_scoring.py       # Étape 6 : Pipeline de scoring batch et segmentation de risque
├── .github/workflows/                 # Intégration Continue
│   └── ci.yml                         # CI GitHub Actions
├── Dockerfile                         # Conteneurisation de production
├── docker-compose.yml                 # Déploiement dashboard / scoring
├── requirements.txt                   # Dépendances Python
└── .gitignore                         # Fichiers et dossiers exclus du suivi git
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

## 🚀 Pipeline de production (Étapes 1 à 6)

Le pipeline complet s'exécute de manière séquentielle et reproductible :

### Étape 1 : Construction de la base de staging (`src/06_build_staging_db.py`)
Construit la base SQLite relationnelle indexée `data/kkbox_cohorts_staging.db` (4,02 Go) en intégrant les fichiers `v1` et `v2`, avec échantillonnage stratifié reproductible (`seed=42`) de deux cohortes indépendantes (Cohorte A d'entraînement à cutoff 28/02/2017 et Cohorte B d'évaluation finale à cutoff 31/03/2017).

```powershell
python src/06_build_staging_db.py
```

### Étape 2 : Feature Engineering étanche (`src/07_build_features.py`)
Calcule 36 features comportementales (RFM, vélocité d'écoute, logs à fenêtres glissantes 30j/60j/90j, autorenew) avec respect strict de la date frontière (*point-in-time*) pour éliminer tout risque de fuite de données (*data leakage*). Génère `cohort_a_features.csv` et `cohort_b_features.csv`.

```powershell
python src/07_build_features.py
```

### Étape 3 : Modèle Baseline (`src/08_train_baseline.py`)
Entraîne une Régression Logistique avec imputation médiane et standardisation robuste.

```powershell
python src/08_train_baseline.py
```

### Étape 4 : Modèles Avancés & Optimisation du Seuil (`src/09_train_advanced_models.py`)
Entraîne LightGBM et XGBoost avec `scale_pos_weight` calculé sur les données d'entraînement. Sélectionne le seuil de décision optimal maximisant le F1-score sur la validation Cohorte A. Évalue sans biais sur la Cohorte B (out-of-time test set).

```powershell
python src/09_train_advanced_models.py
```

### Étape 5 : Interprétabilité SHAP & Évaluation Métier (`src/10_...` & `src/11_...`)
Calcule les valeurs SHAP (TreeExplainer), classe les leviers de désabonnement (`auto_renew_churn_risk`, `discount_per_day`, `pct_25`), et simule le retour sur investissement (ROI) de campagnes de rétention ciblant les percentiles de risque (Top 5%, 10%, 20%).

```powershell
python src/10_model_interpretability.py
python src/11_business_evaluation.py
```

### Étape 6 : Scoring Batch & Segmentation Client (`src/12_production_scoring.py`)
Scénario de production : charge le modèle champion XGBoost et le préprocesseur sérialisé, prédit les probabilités de churn, classe les clients du plus au moins risqué, et assigne les segments de rétention (`High risk`: Top 5%, `Medium risk`: 5–20%, `Low risk`: 80% restants).

```powershell
python src/12_production_scoring.py
```

---

## 📊 Tableau de bord d'aide à la décision & Power BI

Les données prêtes pour le reporting décisionnel sont exportées dans `data/dashboard/` :
- `predictions.csv` : scores individuels et catégorie de risque pour chaque abonné.
- `segment_summary.csv` : synthèse agrégée des segments (High, Medium, Low Risk).
- `feature_importance.csv` : importance des variables explicatives SHAP & XGBoost.
- `model_metrics.csv` : indicateurs de performance clés (ROC-AUC, PR-AUC, F1).

Consultez le guide détaillé [dashboard/POWERBI_GUIDE.md](file:///d:/Documents/projects/kkbox-churn-prediction-challenge/dashboard/POWERBI_GUIDE.md) pour les instructions de connexion pas à pas dans Microsoft Power BI.

---

## 🧪 Contrôle qualité & CI

Pour vérifier la conformité du code :
```powershell
ruff check src/
```
