# Big Data & Machine Learning — TPs M2TL

Fatima Zahra - M2TL Digital Campus

## 🎯 Fil rouge

Prédiction du **churn client** sur le dataset public **Telco Customer Churn** (IBM). L'objectif métier : identifier en amont les clients d'un opérateur télécom susceptibles de résilier pour cibler une campagne de rétention.

Chaque TP couvre une étape du pipeline ML, de l'exploration des données à la modélisation, avec un transfert d'apprentissage progressif d'un TP à l'autre.

## 📁 Structure du repo

```
├── data/                       # Splits TP1 + splits featurisés TP3
│   ├── X_train.csv, X_val.csv, X_test.csv
│   ├── y_train.csv, y_val.csv, y_test.csv
│   └── X_train_fe.csv, X_val_fe.csv, X_test_fe.csv
├── artifacts/                  # Best model + manifeste (TP4)
│   ├── best_model.joblib
│   └── manifest.json
├── TP1.ipynb                   # EDA & preprocessing
├── TP2.ipynb                   # Baseline LogReg & métriques
├── TP3.ipynb                   # Pipeline pro & course de modèles (S2 matin)
├── TP4.ipynb                   # Tuning, seuil métier & best model (S2 après-midi)
├── TP5.ipynb                   # Segmentation client non supervisée (S3 matin)
├── TP6.ipynb                   # Cluster comme feature (S3 après-midi)
├── TP7.ipynb                   # API FastAPI — du modèle au service web (S4 matin)
├── TP8.ipynb                   # Tests API — pytest, TestClient, OpenAPI (S4 après-midi)
├── .gitignore
└── README.md
```

## 📚 TP1 — EDA & Preprocessing initial

**Notebook :** `TP1.ipynb`

EDA du dataset Telco Customer Churn (7 043 clients × 21 colonnes) et split stratifié train/val/test (70/15/15).

**Résultats clés :**

- Taux de churn global : **26,54 %** (déséquilibre ~3:1)
- Top prédicteurs : `Contract`, `InternetService`, `PaymentMethod`, `tenure`
- Multicolinéarité détectée : `TotalCharges ≈ tenure × MonthlyCharges` (r = 0,9996)
- Splits sauvegardés dans `data/` pour réutilisation au TP2

## 📚 TP2 — Baseline Logistic Regression & Métriques

**Notebook :** `TP2.ipynb`

Construction d'un baseline de classification (régression logistique) avec pipeline scikit-learn complet (preprocessing + modèle), évaluation multi-métriques et interprétation des coefficients.

**Résultats clés :**

- Baseline LogReg @ seuil 0.5 : Accuracy 0.81, F1 0.62, ROC-AUC 0.85, PR-AUC 0.63 (vs Dummy 0.73)
- Top features confirmant l'EDA : `tenure` (coef négatif fort), `Contract_Two year`, `InternetService_Fiber optic`
- Modèle bien calibré (Brier score nettement sous la baseline aléatoire)
- `class_weight="balanced"` fait passer le recall de 0.59 → 0.81 au prix de la precision
- Anomalies à corriger en TP3 : multicolinéarité `TotalCharges`/`tenure`/`MonthlyCharges`, 6 features redondantes `"No internet service"`

## 📚 TP3 — Pipeline pro & course de modèles

**Notebook :** `TP3.ipynb`

Feature engineering, cross-validation stratifiée multi-métriques, tracking MLflow, et comparaison de 5 familles de modèles pour identifier les 2 candidats à tuner.

**Résultats clés :**

- 4 features ingénierées ajoutées : `tenure_group`, `services_count`, `has_internet`, `avg_charge_per_month`
- CV 5-fold stratifiée multi-métriques (F1, ROC-AUC, PR-AUC)
- Course de 5 modèles (F1 mean ± std sur 5 folds) :
  - LogReg : **0.5828 ± 0.0114**
  - DecisionTree : 0.5512 ± 0.0192
  - RandomForest : 0.5641 ± 0.0163
  - GradientBoosting : **0.5892 ± 0.0161**
  - HistGradientBoosting : 0.5646 ± 0.0093
- Course très serrée en tête : tous les bons modèles convergent vers F1 ≈ 0.58
- Tracking complet via MLflow (expérience `churn-models-zoo`)
- 2 candidats retenus pour tuning : **LogReg** (rapide, stable, interprétable) + **HistGradientBoosting** (potentiel via tuning + rapide vs GradientBoosting classique)

## 📚 TP4 — Tuning, seuil métier & best model

**Notebook :** `TP4.ipynb`

GridSearchCV sur 2 modèles, optimisation du seuil selon une fonction de coût métier, choix raisonné du best model, permutation importance et évaluation finale (1 seule fois) sur le test set.

**Résultats clés :**

- LogReg tuned : F1 CV = **0.6308** (+0.048 vs default), best params `C=0.01, class_weight="balanced"`
- HGBT tuned : F1 CV = **0.6293** (+0.065 vs default), best params `lr=0.05, max_depth=5, max_iter=200, class_weight="balanced"`
- Insight clé : `class_weight="balanced"` est le levier dominant (+4.5 pts F1), mathématiquement équivalent à baisser le seuil
- Fonction de coût métier : gain = 55 € × TP - 5 € × FP (hypothèse 30 % succès rétention, LTV 200 €, appel 5 €)
- Seuils optimaux : HGBT @ **0.15** (12 775 €), LogReg @ 0.25 (12 675 €) — bien sous le 0.5 par défaut
- **Best model retenu** : `HistGradientBoosting` tuned @ seuil 0.15
- Top 3 features (permutation importance) : `Contract` (0.123), `tenure` (0.097), `InternetService` (0.021)
- **Évaluation finale test set** : F1 0.5509, PR-AUC 0.6448, **Recall 94.3 %**, **Gain métier 12 495 €** (265/281 churners détectés)
- Modèle persisté dans `artifacts/best_model.joblib` + manifeste JSON pour la session 4 (API FastAPI)

## 📚 TP5 — Segmentation client non supervisée (clustering & PCA)

**Notebook :** `TP5.ipynb`

Segmentation des clients via **k-means** et **DBSCAN**, visualisation 2D par **PCA**, et profiling métier des clusters.

**Points clés :**

- Préprocesseur (StandardScaler + OneHotEncoder) **fitté sur train uniquement** pour éviter la fuite
- Choix de K via **elbow method** (inertie) + **silhouette score** sur sous-échantillon de 2 000 points
- Décision pédagogique : **K=4** retenu (compromis lisibilité métier vs silhouette max à K=2)
- Visualisation 2D des clusters via `PCA(n_components=2)`
- **Profiling** des clusters (tenure, MonthlyCharges, services_count, churn rate, top Contract / InternetService / PaymentMethod)
- Test de **DBSCAN** avec `eps` choisi via k-distance plot — illustre pourquoi DBSCAN convient moins aux données tabulaires homogènes type Telco

## 📚 TP6 — Réinjection des clusters comme feature

**Notebook :** `TP6.ipynb`

Construction d'une feature `cluster_id` SANS data leakage, comparaison rigoureuse en CV "avec vs sans cluster" sur le best model S2 (HGBT), et test de PCA comme réduction de dimension.

**Points clés :**

- Fonction `add_cluster_feature` qui **fit sur train uniquement** (préprocesseur + k-means), puis **predict** sur val/test
- Comparaison 5-fold CV stratifiée du HGBT tuné de S2 : **avec vs sans** `cluster_id` (F1, PR-AUC, ΔF1 par fold)
- Décision argumentée : on adopte ou rejette la feature selon le ΔF1 (significativité statistique + pratique)
- Bonus PCA (`n_components=10`) injectée dans le pipeline — confirme que PCA dégrade les modèles d'arbres
- Évaluation finale sur le test set avec la feature cluster (seuil métier S2)
- Artefacts sauvegardés : `artifacts/kmeans.joblib` + `artifacts/best_model_with_cluster.joblib`

## 📚 TP7 — Du modèle au service web : construire l'API

**Notebook :** `TP7.ipynb` (manuel pas-à-pas) — **Projet :** `churn_api/`

Construction d'une API FastAPI pour exposer le best model du TP4. Architecture en couches (schemas / model / routes), validation Pydantic stricte, chargement modèle one-shot au démarrage via lifespan, doc auto-générée via Swagger.

**Résultats clés :**

- 5 endpoints REST : `/` (redirect docs), `/health`, `/model-info`, `/predict`, `/predict-batch`
- Validation Pydantic stricte sur 19 features (Literal pour les enums, Field bounds pour les numériques)
- Pattern lifespan : modèle HGBT chargé une seule fois au démarrage (pas par requête)
- Séparation logique métier (`model.py`) / routes HTTP (`main.py`) / validation I/O (`schemas.py`)
- `add_engineered_features` synchronisée avec celle du TP3 (sync critique entre training et inférence)
- API testable via Swagger UI sur `http://localhost:8000/docs`
- Sortie cohérente entre CLI local et environnement Colab (reproductibilité totale)

## 📚 TP8 — Tester l'API : pytest, TestClient, OpenAPI

**Notebook :** `TP8.ipynb`

Tests automatisés de l'API du TP7 via TestClient de FastAPI, validation de la gestion d'erreurs Pydantic, exploration du schéma OpenAPI auto-généré.

**Résultats clés :**

- 4 catégories d'erreurs interceptées en 422 (jamais de 500 sur input invalide) :
  - Champ manquant → message localisé (`loc: ['body', 'tenure']`)
  - Valeur hors enum Literal → liste des valeurs acceptées
  - Hors bornes Field(ge=, le=) → contrainte explicite
  - Mauvais type → coercion intelligente Pydantic
- Schéma OpenAPI auto-généré déclare 200 + 422 pour chaque endpoint, sans doc manuelle
- `$ref` vers `ClientFeatures` partagée (pattern DRY appliqué à la doc)
- Pattern batch endpoint : 1000× plus rapide qu'une boucle de requêtes individuelles
- 10 tests pytest passent : 5 endpoints + 4 cas d'erreur + schéma OpenAPI
