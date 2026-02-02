# Guide de la Structure du Projet - Système de Prédiction des Carences en Vitamines

Cette documentation explique le rôle de chaque dossier du projet `projet_vitamin_IA`, son contenu et comment l'utiliser. Travailler en équipe signifie que chaque membre doit savoir ce qui appartient à chaque section.

---

## 📋 Table des matières

1. [Structure Générale du Projet](#structure-générale-du-projet)
2. [Définitions des Dossiers](#définitions-des-dossiers)
3. [Types de Fichiers et Conventions](#types-de-fichiers-et-conventions)
4. [Distribution des Tâches par Équipe](#distribution-des-tâches-par-équipe)

---

## 🏗️ Structure Générale du Projet

```
projet_vitamin_IA/
├── data/                    # Gestion des données
├── src/                     # Code de production (modules Python)
├── notebooks/               # Exploration et expérimentation (Jupyter)
├── models/                  # Modèles d'apprentissage automatique entraînés
├── app/                     # Application web (Flask/FastAPI)
├── visualizations/          # Graphiques et tableaux de bord Tableau
├── tests/                   # Tests unitaires
├── docs/                    # Documentation
├── requirements.txt         # Dépendances Python
├── .gitignore              # Fichiers ignorés par Git
├── README.md               # Résumé du projet
└── LICENSE                 # Licence MIT
```

---

## 📁 Définitions des Dossiers

### 1. **data/** - GESTION DES DONNÉES
**Qu'est-ce qu'il contient ?** Fichiers de données brutes et traitées

**Sous-dossiers :**
```
data/
├── raw/                    # Données originales - NE PAS TOUCHER
│   ├── vitamin_deficiency_dataset.csv  (4000 dossiers patients, 34 colonnes)
│   └── CIQUAL_food_database.csv        (3484 aliments, 84 colonnes)
│
├── processed/              # Données nettoyées et prêtes
│   ├── X_train.csv         (ensemble d'entraînement)
│   ├── X_test.csv          (ensemble de test)
│   ├── y_train.csv         (variable cible d'entraînement)
│   └── y_test.csv          (variable cible de test)
│
└── external/               # Données de référence externes
    ├── vitamin_daily_requirements.csv
    └── nutrient_groups.csv
```

**Règles Importantes :**
- ⛔ **JAMAIS** modifier les fichiers dans `raw/`
- ✅ Le dossier `processed/` contient les données nettoyées (écrites par Python)
- 📊 Avant d'ouvrir les CSV dans Excel, vérifier la version sur GitHub

**Membres de l'équipe qui l'utilisent :**
- Analyste de données : exploration (EDA)
- Ingénieur ML : entraînement des modèles

---

### 2. **src/** - CODE DE PRODUCTION (CRUCIAL !)
**Qu'est-ce qu'il contient ?** Modules Python réutilisables et propres

**Fichiers :**
```
src/
├── __init__.py             # Définition du paquet Python
├── preprocessing.py        # Fonctions de nettoyage des données
├── feature_engineering.py  # Création de nouvelles variables
├── model.py                # Entraînement et prédiction du modèle
├── recommendation.py       # Recommandations nutritionnelles avec CIQUAL
└── utils.py                # Fonctions utilitaires
```

**Chaque fichier, qu'est-ce qu'il fait ?**

**preprocessing.py :**
```python
# load_data()           - Charge les fichiers CSV
# clean_missing_values()- Traite les valeurs manquantes
# encode_categorical()  - Convertit les catégories en nombres
# scale_features()      - Normalise les données (0-1)
```

**feature_engineering.py :**
```python
# create_bmi_category()     - BMI → catégories (faible, normal, élevé)
# calculate_symptom_score() - Poids des symptômes
# create_interactions()     - Combinaisons de variables
```

**model.py :**
```python
# train_model()         - Entraîne le modèle
# evaluate_model()      - Mesure la performance (accuracy, F1, etc.)
# predict()             - Prédiction pour nouveau patient
```

**recommendation.py :**
```python
# get_foods_by_vitamin()    - Carence → aliments recommandés
# calculate_nutrition()      - Score de correspondance nutritionnelle
# rank_recommendations()     - Classe les recommandations
```

**utils.py :**
```python
# save_model()          - Sauvegarde le modèle (.pkl)
# load_model()          - Charge un modèle entraîné
# log_metrics()         - Enregistre les métriques
```
# Guide Complet de la Structure du Projet - Système de Prédiction des Carences en Vitamines

Ce document détaille la structure réelle du projet `projet_vitamin_IA` pour notre équipe de développement. Chaque dossier, sous-dossier et fichier a un rôle spécifique dans le workflow de travail.

---

## 📋 Table des Matières

1. [Structure Générale Complète](#structure-générale-complète)
2. [Descriptions Détaillées des Dossiers](#descriptions-détaillées-des-dossiers)
3. [Types de Fichiers et Conventions](#types-de-fichiers-et-conventions)
4. [Distribution des Tâches par Équipe](#distribution-des-tâches-par-équipe)
5. [Bonnes Pratiques](#bonnes-pratiques)

---

## 🏗️ Structure Générale Complète

```
projet_vitamin_IA/
│
├── 📁 app/                          # APPLICATION WEB
│   ├── .gitkeep
│   ├── app.py                       # Application principale (Flask/FastAPI)
│   ├── config.py                    # Configuration
│   ├── requirements.txt             # Dépendances web
│   ├── templates/                   # Fichiers HTML
│   │   ├── index.html              # Page d'accueil
│   │   ├── results.html            # Résultats de prédiction
│   │   └── recommendations.html    # Recommandations nutritionnelles
│   └── static/                      # Ressources statiques
│       ├── css/
│       │   └── style.css
│       ├── js/
│       │   └── script.js
│       └── images/
│           └── logo.png
│
├── 📁 data/                         # GESTION DES DONNÉES
│   ├── data_csv/                    # Données brutes en CSV
│   │   ├── .gitkeep
│   │   ├── vitamin_deficiency_dataset.csv    (4000 dossiers, 34 colonnes)
│   │   └── CIQUAL_food_database.csv          (3484 aliments, 84 colonnes)
│   │
│   ├── external/                    # Données de référence externes
│   │   ├── vitamin_daily_requirements.csv
│   │   ├── food_groups.csv
│   │   └── .gitkeep
│   │
│   ├── processed/                   # Données nettoyées et traitées
│   │   ├── X_train.csv             # Ensemble d'entraînement (features)
│   │   ├── X_test.csv              # Ensemble de test (features)
│   │   ├── y_train.csv             # Cibles d'entraînement
│   │   ├── y_test.csv              # Cibles de test
│   │   └── .gitkeep
│   │
│   └── raw/                         # Données originales INTACTES
│       └── .gitkeep
│
├── 📁 database_sql/                 # SCHÉMA ET REQUÊTES SQL
│   ├── .gitkeep
│   ├── schema.sql                   # Création de toutes les tables
│   ├── vitamin_deficiency.sql       # Table des patients/carences
│   ├── ciqual_foods.sql             # Table des aliments CIQUAL
│   ├── vitamin_requirements.sql     # Table des besoins nutritionnels
│   ├── queries.sql                  # Requêtes SQL réutilisables
│   ├── insert_data.sql              # Scripts d'insertion de données
│   └── README.md                    # Documentation de la base de données
│
├── 📁 docs/                         # DOCUMENTATION
│   ├── .gitkeep
│   ├── GUIDE_STRUCTURE_PROJET_FR.md # Ce fichier
│   ├── methodology.md               # Méthodologie ML
│   ├── API.md                       # Documentation des endpoints API
│   ├── data_dictionary.md           # Dictionnaire des données CSV
│   ├── database_schema.md           # Schéma détaillé de la base de données
│   ├── model_performance.md         # Métriques et performance
│   └── deployment.md                # Guide de déploiement
│
├── 📁 models/                       # MODÈLES ENTRAÎNÉS
│   ├── .gitkeep
│   ├── model_v1.pkl                 # Premier modèle (Random Forest)
│   ├── model_v2.pkl                 # Deuxième modèle (XGBoost)
│   ├── scaler.pkl                   # Normaliseur de données
│   ├── label_encoder.pkl            # Encodeur de variables catégories
│   ├── feature_names.pkl            # Noms des features utilisées
│   └── model_registry.json          # Registre de versions
│
├── 📁 notebooks/                    # EXPLORATION ET EXPÉRIMENTATION
│   ├── .gitkeep
│   ├── 01_eda.ipynb                 # Analyse Exploratoire des Données
│   ├── 02_data_cleaning.ipynb       # Nettoyage et préparation
│   ├── 03_model_training.ipynb      # Entraînement et comparaison
│   ├── 04_recommendations.ipynb     # Recommandations CIQUAL
│   └── 05_web_app_testing.ipynb     # Tests de l'application
│
├── 📁 src/                          # CODE PYTHON DE PRODUCTION
│   ├── __init__.py                  # Initialisation du paquet
│   ├── .gitkeep
│   ├── preprocessing.py             # Nettoyage des données
│   ├── feature_engineering.py       # Création de features
│   ├── model.py                     # Entraînement et prédiction
│   ├── recommendation.py            # Moteur de recommandations
│   ├── database.py                  # Connexion à la base de données
│   └── utils.py                     # Fonctions utilitaires
│
├── 📁 test/                         # TESTS UNITAIRES
│   ├── .gitkeep
│   ├── test_preprocessing.py        # Tests du nettoyage
│   ├── test_model.py                # Tests du modèle
│   ├── test_recommendation.py       # Tests des recommandations
│   ├── test_database.py             # Tests de la base de données
│   └── test_app.py                  # Tests de l'application web
│
├── 📁 visualizations/               # GRAPHIQUES ET DASHBOARDS
│   ├── .gitkeep
│   ├── dashboard.twbx               # Dashboard Tableau interactif
│   ├── plots/                       # Graphiques PNG/PDF
│   │   ├── vitamin_distribution.png
│   │   ├── symptom_frequency.png
│   │   ├── model_performance.png
│   │   ├── heatmap_correlation.png
│   │   └── .gitkeep
│   └── analysis_report.pdf          # Rapport avec graphiques
│
├── .gitattributes                   # Paramètres Git
├── .gitignore                       # Fichiers à ignorer par Git
├── .gitkeep                         # Marqueur de dossier
├── CONTRIBUTING.md                  # Guide de contribution
├── LICENSE                          # Licence MIT
├── README.md                        # Résumé du projet
├── requirements.txt                 # Dépendances Python générales
├── Roadmap_Project.md              # Plan 12 semaines
└── project_structure.md             # Vue d'ensemble de la structure
```

---

## 📁 Descriptions Détaillées des Dossiers

### 1. **app/** - APPLICATION WEB

**Rôle :** Interface utilisateur pour les prédictions et recommandations

#### app.py
```python
from flask import Flask, render_template, request, jsonify
from src.model import predict
from src.recommendation import get_recommendations
from src.database import get_patient_history

app = Flask(__name__)

@app.route('/')
def home():
    """Page d'accueil avec formulaire"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """API de prédiction"""
    patient_data = request.form.to_dict()
    prediction = predict(patient_data)
    recommendations = get_recommendations(prediction)
    
    return render_template('results.html',
                          prediction=prediction,
                          recommendations=recommendations)

@app.route('/api/patient/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    """Récupérer l'historique d'un patient"""
    patient = get_patient_history(patient_id)
    return jsonify(patient)
```

#### templates/index.html
```html
<!-- Formulaire de saisie patient -->
<form method="POST" action="/predict">
    <input type="number" name="age" required>
    <input type="text" name="gender" required>
    <input type="number" name="bmi" required>
    <!-- Plus de champs... -->
    <button type="submit">Prédire</button>
</form>
```

#### templates/results.html
```html
<!-- Affichage des résultats -->
<h2>Résultats de la Prédiction</h2>
<p>Maladie : {{ prediction.disease }}</p>
<p>Confiance : {{ prediction.confidence }}%</p>
<table>
    <!-- Recommandations nutritionnelles -->
</table>
```

#### static/css/style.css
```css
/* Styles de l'application */
body { font-family: Arial, sans-serif; }
.result { color: green; }
```

#### static/js/script.js
```javascript
// Interactivité et validation
document.getElementById('form').addEventListener('submit', function(e) {
    // Valider les données
    // Afficher un loader
});
```

**Responsable :** Développeur Web
**Utilise :** `src/`, `models/`, `database_sql/`

---

### 2. **data/** - GESTION DES DONNÉES

**Rôle :** Stocker et organiser les données du projet

#### **data/data_csv/** - Données Brutes
- `vitamin_deficiency_dataset.csv` : 4000 patients, 34 colonnes
  - Colonnes : age, gender, bmi, symptoms_count, disease_diagnosis, etc.
  - **NE JAMAIS MODIFIER DIRECTEMENT**

- `CIQUAL_food_database.csv` : 3484 aliments, 84 colonnes
  - Colonnes : alim_code, alim_nom_eng, vitamin_a, vitamin_d, iron, etc.
  - Source officielle : ANSES (anses.fr)

#### **data/processed/** - Données Traitées
```python
# Exemple de création
from src.preprocessing import load_data, clean_data

df = load_data('data/data_csv/vitamin_deficiency_dataset.csv')
cleaned_df = clean_data(df)

# Sauvegarder
cleaned_df.to_csv('data/processed/cleaned_vitamin_data.csv')

# Créer X et y
X_train.to_csv('data/processed/X_train.csv')
y_train.to_csv('data/processed/y_train.csv')
```

#### **data/external/** - Données Externes
- `vitamin_daily_requirements.csv` : Besoins quotidiens en vitamines
- `food_groups.csv` : Classification des aliments

**Responsable :** Analyste de données, Ingénieur ML
**Règle :** Ne jamais modifier `data_csv/`, utiliser `processed/` et `external/`

---

### 3. **database_sql/** - BASE DE DONNÉES SQL

**Rôle :** Définir et gérer la structure de la base de données

**IMPORTANT :** Ce dossier contient les schémas SQL pour les environnements SQL/PostgreSQL/MySQL

#### schema.sql
```sql
-- Créer la base de données
CREATE DATABASE vitamin_deficiency_db;
USE vitamin_deficiency_db;

-- Importer les schémas des autres fichiers
SOURCE vitamin_deficiency.sql;
SOURCE ciqual_foods.sql;
SOURCE vitamin_requirements.sql;
```

#### vitamin_deficiency.sql
```sql
-- Table des patients et diagnostics
CREATE TABLE patients (
    patient_id INT PRIMARY KEY AUTO_INCREMENT,
    age INT NOT NULL,
    gender VARCHAR(10) NOT NULL,
    bmi DECIMAL(5, 2) NOT NULL,
    smoking_status VARCHAR(20),
    alcohol_consumption INT,
    exercise_level VARCHAR(20),
    diet_type VARCHAR(30),
    sun_exposure INT,
    income_level VARCHAR(20),
    latitude_region DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE vitamin_measurements (
    measurement_id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id INT NOT NULL,
    vitamin_a_percent_rda DECIMAL(5, 2),
    vitamin_c_percent_rda DECIMAL(5, 2),
    vitamin_d_percent_rda DECIMAL(5, 2),
    vitamin_e_percent_rda DECIMAL(5, 2),
    vitamin_b12_percent_rda DECIMAL(5, 2),
    folate_percent_rda DECIMAL(5, 2),
    hemoglobin_g_dl DECIMAL(5, 2),
    serum_vitamin_d_ng_ml DECIMAL(7, 2),
    serum_vitamin_b12_pg_ml DECIMAL(8, 2),
    serum_folate_ng_ml DECIMAL(7, 2),
    measurement_date DATE,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE diagnoses (
    diagnosis_id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id INT NOT NULL,
    disease_diagnosis VARCHAR(50) NOT NULL,
    symptoms_count INT,
    symptoms_list TEXT,
    has_night_blindness BOOLEAN,
    has_fatigue BOOLEAN,
    has_bleeding_gums BOOLEAN,
    has_bone_pain BOOLEAN,
    has_muscle_weakness BOOLEAN,
    has_numbness_tingling BOOLEAN,
    has_memory_problems BOOLEAN,
    has_pale_skin BOOLEAN,
    has_multiple_deficiencies BOOLEAN,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
```

#### ciqual_foods.sql
```sql
-- Table de la base CIQUAL
CREATE TABLE ciqual_foods (
    food_id INT PRIMARY KEY AUTO_INCREMENT,
    alim_code VARCHAR(20) NOT NULL UNIQUE,
    alim_nom_eng VARCHAR(100) NOT NULL,
    alim_nom_sci VARCHAR(100),
    alim_grp_code VARCHAR(10),
    alim_grp_nom_eng VARCHAR(50),
    energy_kJ INT,
    energy_kcal INT,
    water DECIMAL(5, 2),
    protein DECIMAL(5, 2),
    fat DECIMAL(5, 2),
    carbohydrate DECIMAL(5, 2),
    sugars DECIMAL(5, 2),
    fiber DECIMAL(5, 2),
    vitamin_a DECIMAL(7, 2),
    vitamin_c DECIMAL(7, 2),
    vitamin_d DECIMAL(7, 3),
    vitamin_b12 DECIMAL(7, 2),
    vitamin_b9 DECIMAL(7, 2),
    iron DECIMAL(7, 2),
    calcium DECIMAL(7, 2),
    magnesium DECIMAL(7, 2),
    zinc DECIMAL(7, 2),
    salt DECIMAL(5, 2)
);

CREATE INDEX idx_vitamin_a ON ciqual_foods(vitamin_a);
CREATE INDEX idx_vitamin_c ON ciqual_foods(vitamin_c);
CREATE INDEX idx_vitamin_d ON ciqual_foods(vitamin_d);
```

#### vitamin_requirements.sql
```sql
-- Table des besoins nutritionnels quotidiens
CREATE TABLE vitamin_requirements (
    requirement_id INT PRIMARY KEY AUTO_INCREMENT,
    age_min INT,
    age_max INT,
    gender VARCHAR(10),
    vitamin_a_rda INT,
    vitamin_c_rda INT,
    vitamin_d_rda INT,
    vitamin_b12_rda INT,
    folate_rda INT,
    iron_rda INT,
    calcium_rda INT
);
```

#### queries.sql
```sql
-- Requêtes réutilisables

-- 1. Trouver les patients avec carences en vitamine A
SELECT p.patient_id, p.age, vm.vitamin_a_percent_rda
FROM patients p
JOIN vitamin_measurements vm ON p.patient_id = vm.patient_id
WHERE vm.vitamin_a_percent_rda < 100
ORDER BY vm.vitamin_a_percent_rda ASC;

-- 2. Recommander des aliments riches en vitamine D
SELECT alim_nom_eng, vitamin_d
FROM ciqual_foods
WHERE vitamin_d > 10
ORDER BY vitamin_d DESC
LIMIT 10;

-- 3. Historique des diagnostics d'un patient
SELECT d.diagnosis_id, d.disease_diagnosis, d.prediction_date
FROM diagnoses d
WHERE d.patient_id = ? 
ORDER BY d.prediction_date DESC;
```

#### insert_data.sql
```sql
-- Scripts d'insertion des données CSV
-- À exécuter après la création des tables

-- Importer les données CSV
LOAD DATA INFILE '/path/to/vitamin_deficiency_dataset.csv'
INTO TABLE patients
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE '/path/to/CIQUAL_food_database.csv'
INTO TABLE ciqual_foods
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
```

#### README.md (database_sql/README.md)
```markdown
# Documentation de la Base de Données

## Vue d'ensemble
Cette base de données gère :
- Informations des patients
- Mesures de vitamines
- Diagnostics et symptômes
- Base alimentaire CIQUAL

## Installation

### 1. Créer la base de données
```bash
mysql -u root -p < database_sql/schema.sql
```

### 2. Insérer les données
```bash
mysql -u root -p vitamin_deficiency_db < database_sql/insert_data.sql
```

### 3. Vérifier l'installation
```bash
mysql -u root -p -e "USE vitamin_deficiency_db; SHOW TABLES;"
```

## Tables Principales

### patients
- Informations démographiques et mode de vie
- Colonnes : age, gender, bmi, smoking_status, etc.

### vitamin_measurements
- Mesures de vitamines sériques
- Colonnes : vitamin_a_percent_rda, vitamin_d_ng_ml, etc.

### diagnoses
- Diagnostics et symptômes
- Colonnes : disease_diagnosis, symptoms_count, etc.

### ciqual_foods
- Base de données nutritionnelle officielle
- Colonnes : vitamin_a, vitamin_c, iron, calcium, etc.

## Requêtes Courantes

Voir `queries.sql` pour les exemples.
```

**Responsable :** Ingénieur ML, DBA
**Utilise :** Python via `src/database.py`

---

### 4. **docs/** - DOCUMENTATION

**Rôle :** Documenter tous les aspects du projet

#### GUIDE_STRUCTURE_PROJET_FR.md
```
Ce fichier - Guide complet de la structure
```

#### methodology.md
```markdown
# Méthodologie Machine Learning

## Approche
1. Exploration des données (EDA)
2. Préparation et nettoyage
3. Feature engineering
4. Entraînement et validation
5. Déploiement

## Modèles testés
- Logistic Regression
- Random Forest
- Gradient Boosting (XGBoost)
- Neural Networks

## Métrique de sélection
Accuracy, Precision, Recall, F1-score, ROC-AUC
```

#### API.md
```markdown
# Documentation de l'API

## POST /predict
Faire une prédiction

Request:
```json
{
  "age": 35,
  "gender": "M",
  "bmi": 24.5,
  "symptoms_count": 3
}
```

Response:
```json
{
  "disease": "Vitamin_A_Deficiency",
  "confidence": 0.89
}
```

## GET /api/patient/{id}
Récupérer l'historique
```

#### data_dictionary.md
```markdown
# Dictionnaire des Données

## Vitamin Deficiency Dataset (4000 x 34)

| Colonne | Type | Description |
|---------|------|-------------|
| age | int | Âge du patient (18-100) |
| gender | str | Sexe (M/F) |
| bmi | float | Index de masse corporelle |
| smoking_status | str | Fumeur (Yes/No) |
| vitamin_a_percent_rda | float | % besoins quotidiens |
| disease_diagnosis | str | Maladie identifiée |
```

#### database_schema.md
```markdown
# Schéma de la Base de Données

## Diagramme ER

patients
├── vitamin_measurements
├── diagnoses
└── ciqual_foods

## Tables
- patients (8 colonnes)
- vitamin_measurements (10 colonnes)
- diagnoses (13 colonnes)
- vitamin_requirements (10 colonnes)
- ciqual_foods (24 colonnes)
```

#### model_performance.md
```markdown
# Performance des Modèles

## Model v2 (XGBoost) - MEILLEUR
- Accuracy: 0.887
- Precision: 0.85
- Recall: 0.88
- F1-score: 0.865
- ROC-AUC: 0.92
```

#### deployment.md
```markdown
# Guide de Déploiement

## Production
1. Exécuter tests: `pytest tests/`
2. Copier modèle: `cp models/model_v2.pkl prod/`
3. Démarrer app: `python app/app.py`
4. Vérifier: `curl http://localhost:5000/`
```

**Responsable :** Tous (maintenance collective)

---

### 5. **models/** - MODÈLES ENTRAÎNÉS

**Rôle :** Stocker les modèles d'IA sérialisés

```python
# Sauvegarder un modèle
import pickle
with open('models/model_v2.pkl', 'wb') as f:
    pickle.dump(model, f)

# Charger un modèle
with open('models/model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

# Faire une prédiction
prediction = model.predict(X_test)
```

#### model_registry.json
```json
{
  "current_model": "model_v2",
  "models": [
    {
      "name": "model_v1",
      "type": "RandomForest",
      "accuracy": 0.82,
      "created_at": "2026-02-01",
      "status": "deprecated"
    },
    {
      "name": "model_v2",
      "type": "XGBoost",
      "accuracy": 0.887,
      "created_at": "2026-02-02",
      "status": "active"
    }
  ]
}
```

**Responsable :** Ingénieur ML
**⚠️ Règle :** NE PAS committer sur GitHub (dans .gitignore)

---

### 6. **notebooks/** - EXPLORATION ET EXPÉRIMENTATION

**Rôle :** Développer, tester et explorer avant la production

#### 01_eda.ipynb
```python
# Charger et explorer
import pandas as pd
df = pd.read_csv('data/data_csv/vitamin_deficiency_dataset.csv')

# Statistiques
df.describe()
df.info()
df.isnull().sum()

# Visualisations
df['vitamin_a_percent_rda'].hist()
df.corr().heatmap()
```

#### 02_data_cleaning.ipynb
```python
# Importer les fonctions de preprocessing
from src.preprocessing import clean_missing_values, scale_features

# Tester
cleaned_df = clean_missing_values(df)
scaled_df = scale_features(cleaned_df)

# Sauvegarder
cleaned_df.to_csv('data/processed/cleaned_data.csv')
```

#### 03_model_training.ipynb
```python
# Importer et entraîner
from src.model import train_model, evaluate_model

# Préparer données
X = cleaned_df.drop('disease_diagnosis', axis=1)
y = cleaned_df['disease_diagnosis']

# Entraîner
model = train_model(X, y)
metrics = evaluate_model(model, X_test, y_test)
print(metrics)
```

#### 04_recommendations.ipynb
```python
# Recommandations nutritionnelles
from src.recommendation import get_foods_by_vitamin

disease = "Vitamin_A_Deficiency"
foods = get_foods_by_vitamin(disease)
print(foods)
```

#### 05_web_app_testing.ipynb
```python
# Tester l'application
import requests

response = requests.post('http://localhost:5000/predict', {
    'age': 35,
    'vitamin_a_percent_rda': 60,
    'symptoms_count': 3
})
print(response.json())
```

**Responsable :** Analyste de données, Ingénieur ML
**Workflow :** Tester → Valider → Copier dans `src/` → Utiliser en production

---

### 7. **src/** - CODE PYTHON DE PRODUCTION

**Rôle :** Code réutilisable, testé et en production

#### preprocessing.py
```python
"""
Module de préparation des données.
Fonctions de nettoyage et transformation.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    """
    Charger les données depuis un fichier CSV.
    
    Parameters:
    -----------
    filepath : str
        Chemin vers le fichier CSV
    
    Returns:
    --------
    pd.DataFrame
        Données chargées
    """
    return pd.read_csv(filepath)

def clean_missing_values(df, strategy='drop'):
    """
    Traiter les valeurs manquantes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Données brutes
    strategy : str
        'drop' ou 'mean' ou 'median'
    
    Returns:
    --------
    pd.DataFrame
        Données nettoyées
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())

def encode_categorical(df, categorical_cols):
    """
    Convertir variables catégories en nombres.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Données
    categorical_cols : list
        Colonnes à encoder
    
    Returns:
    --------
    pd.DataFrame, dict
        Données encodées et encodeurs
    """
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

def scale_features(df, numerical_cols):
    """
    Normaliser les données (0-1).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Données
    numerical_cols : list
        Colonnes à normaliser
    
    Returns:
    --------
    pd.DataFrame, scaler
        Données normalisées et scaler
    """
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler
```

#### feature_engineering.py
```python
"""
Module de création de nouvelles variables.
"""

import pandas as pd
import numpy as np

def create_bmi_category(df):
    """
    Créer catégories BMI.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Données avec colonne 'bmi'
    
    Returns:
    --------
    pd.DataFrame
        Données avec colonne 'bmi_category'
    """
    df['bmi_category'] = pd.cut(df['bmi'], 
                                 bins=[0, 18.5, 25, 30, float('inf')],
                                 labels=['underweight', 'normal', 'overweight', 'obese'])
    return df

def calculate_symptom_score(df):
    """
    Calculer score de sévérité des symptômes.
    """
    symptom_cols = [c for c in df.columns if c.startswith('has_')]
    df['symptom_severity'] = df[symptom_cols].sum(axis=1)
    return df

def create_interaction_features(df):
    """
    Créer interactions entre variables.
    """
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['symptoms_age_interaction'] = df['symptom_severity'] * df['age']
    return df
```

#### model.py
```python
"""
Module d'entraînement et prédiction.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

def train_model(X, y, model_type='xgboost'):
    """
    Entraîner un modèle de classification.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features d'entraînement
    y : pd.Series
        Cibles
    model_type : str
        Type de modèle
    
    Returns:
    --------
    model
        Modèle entraîné
    """
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(max_depth=6, learning_rate=0.1, random_state=42)
    
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Évaluer la performance du modèle.
    
    Parameters:
    -----------
    model : model
        Modèle entraîné
    X_test : pd.DataFrame
        Features de test
    y_test : pd.Series
        Cibles de test
    
    Returns:
    --------
    dict
        Métriques
    """
    predictions = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, average='weighted'),
        'recall': recall_score(y_test, predictions, average='weighted'),
        'f1': f1_score(y_test, predictions, average='weighted')
    }

def predict(model, X):
    """
    Faire une prédiction.
    
    Parameters:
    -----------
    model : model
        Modèle entraîné
    X : pd.DataFrame
        Données à prédire
    
    Returns:
    --------
    dict
        Prédiction et confiance
    """
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X).max()
    
    return {
        'disease': prediction,
        'confidence': float(confidence)
    }

def save_model(model, filepath):
    """
    Sauvegarder un modèle.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """
    Charger un modèle.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)
```

#### recommendation.py
```python
"""
Module de recommandations nutritionnelles.
Utilise CIQUAL pour matcher les carences.
"""

import pandas as pd

def get_foods_by_vitamin(disease, ciqual_df):
    """
    Trouver aliments riches en vitamine pour une carence.
    
    Parameters:
    -----------
    disease : str
        Maladie (ex: Vitamin_A_Deficiency)
    ciqual_df : pd.DataFrame
        Base CIQUAL
    
    Returns:
    --------
    list
        Aliments recommandés
    """
    vitamin_map = {
        'Vitamin_A_Deficiency': 'vitamin_a',
        'Vitamin_C_Deficiency': 'vitamin_c',
        'Vitamin_D_Deficiency': 'vitamin_d',
        'Vitamin_B12_Deficiency': 'vitamin_b12'
    }
    
    vitamin = vitamin_map.get(disease)
    if not vitamin:
        return []
    
    # Trouver les aliments riches
    foods = ciqual_df[ciqual_df[vitamin] > ciqual_df[vitamin].quantile(0.75)]
    foods = foods.sort_values(vitamin, ascending=False).head(10)
    
    return foods[['alim_nom_eng', vitamin]].to_dict('records')

def calculate_nutrition_score(foods, requirements):
    """
    Calculer score de correspondance nutritionnelle.
    """
    return len(foods) / max(1, len(requirements))

def rank_recommendations(recommendations):
    """
    Classer les recommandations par pertinence.
    """
    return sorted(recommendations, 
                  key=lambda x: x.get('score', 0), 
                  reverse=True)
```

#### database.py
```python
"""
Module de connexion et requêtes SQL.
"""

import sqlite3
import pandas as pd

class DatabaseConnection:
    def __init__(self, db_path):
        """
        Initialiser la connexion.
        """
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """
        Créer la connexion.
        """
        self.conn = sqlite3.connect(self.db_path)
    
    def query(self, sql, params=None):
        """
        Exécuter une requête.
        """
        if not self.conn:
            self.connect()
        
        if params:
            result = pd.read_sql_query(sql, self.conn, params=params)
        else:
            result = pd.read_sql_query(sql, self.conn)
        
        return result
    
    def insert_patient(self, patient_data):
        """
        Insérer un patient.
        """
        sql = """
        INSERT INTO patients 
        (age, gender, bmi, smoking_status)
        VALUES (?, ?, ?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(sql, patient_data)
        self.conn.commit()
    
    def close(self):
        """
        Fermer la connexion.
        """
        if self.conn:
            self.conn.close()

def get_patient_history(patient_id):
    """
    Récupérer l'historique d'un patient.
    """
    db = DatabaseConnection('vitamin_db.sqlite')
    db.connect()
    
    sql = """
    SELECT d.disease_diagnosis, d.prediction_date
    FROM diagnoses d
    WHERE d.patient_id = ?
    ORDER BY d.prediction_date DESC
    """
    
    result = db.query(sql, (patient_id,))
    db.close()
    
    return result.to_dict('records')
```

#### utils.py
```python
"""
Fonctions utilitaires générales.
"""

import json
import logging
from datetime import datetime

def save_metrics(metrics, filepath):
    """
    Sauvegarder les métriques.
    """
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_config(filepath):
    """
    Charger la configuration.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def setup_logging(log_file):
    """
    Configurer les logs.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def log_prediction(prediction, confidence, timestamp=None):
    """
    Enregistrer une prédiction.
    """
    logger = setup_logging('logs/predictions.log')
    if timestamp is None:
        timestamp = datetime.now()
    
    logger.info(f"Prediction: {prediction}, Confidence: {confidence}, Time: {timestamp}")
```

**Responsable :** Ingénieur ML
**Règle :** Code testé, documenté et réutilisable

---

### 8. **test/** - TESTS UNITAIRES

**Rôle :** Valider que le code fonctionne correctement

#### test_preprocessing.py
```python
"""
Tests du module preprocessing.
"""

import pytest
import pandas as pd
from src.preprocessing import clean_missing_values, scale_features

def test_clean_missing_values():
    """Tester le nettoyage des valeurs manquantes."""
    df = pd.DataFrame({
        'age': [25, None, 35],
        'vitamin_a': [100, 200, None]
    })
    
    cleaned = clean_missing_values(df)
    assert cleaned.isnull().sum().sum() == 0

def test_scale_features():
    """Tester la normalisation."""
    df = pd.DataFrame({'feature': [0, 50, 100]})
    scaled, _ = scale_features(df, ['feature'])
    
    assert scaled['feature'].min() >= -1
    assert scaled['feature'].max() <= 1
```

#### test_model.py
```python
"""
Tests du module model.
"""

import pytest
from sklearn.datasets import make_classification
from src.model import train_model, evaluate_model

def test_train_model():
    """Tester l'entraînement."""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    model = train_model(X, y, model_type='random_forest')
    assert model is not None

def test_evaluate_model():
    """Tester l'évaluation."""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = train_model(X, y)
    
    metrics = evaluate_model(model, X, y)
    assert 'accuracy' in metrics
    assert metrics['accuracy'] > 0
```

#### test_app.py
```python
"""
Tests de l'application web.
"""

import pytest
from app.app import app

@pytest.fixture
def client():
    """Client de test."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home(client):
    """Tester la page d'accueil."""
    response = client.get('/')
    assert response.status_code == 200

def test_predict(client):
    """Tester l'API de prédiction."""
    data = {
        'age': 35,
        'gender': 'M',
        'bmi': 24.5
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
```

**Exécuter les tests :**
```bash
pytest test/
```

**Responsable :** Ingénieur ML, Développeur Web

---

### 9. **visualizations/** - GRAPHIQUES ET DASHBOARDS

**Rôle :** Créer des visualisations pour l'analyse et la présentation

#### dashboard.twbx (Tableau)
```
Tableur interactif avec :
- Distribution des vitamines
- Fréquence des symptômes
- Performance du modèle
- Analyse démographique
- Carte géographique
```

#### plots/ (Graphiques PNG)
```python
# Exemple de création avec Matplotlib
import matplotlib.pyplot as plt

# Histogramme
df['vitamin_a_percent_rda'].hist(bins=50)
plt.savefig('visualizations/plots/vitamin_distribution.png', dpi=600)

# Heatmap de corrélation
import seaborn as sns
sns.heatmap(df.corr())
plt.savefig('visualizations/plots/heatmap_correlation.png', dpi=600)
```

**Responsable :** Analyste de données, Tableau Specialist

---

## 📄 Types de Fichiers et Conventions

### Noms de Fichiers Python
```
✅ Correct :
- preprocessing.py          # snake_case
- feature_engineering.py
- test_preprocessing.py

❌ Incorrect :
- PreProcessing.py          # CamelCase
- Feature-Engineering.py    # Tirets
- tests_preprocessing.py    # Mauvais ordre
```

### Noms de Notebooks
```
✅ Correct :
01_eda.ipynb
02_data_cleaning.ipynb
03_model_training.ipynb

❌ Incorrect :
eda_final.ipynb
data_cleaning(v2).ipynb
model_v5_real_FINAL.ipynb
```

### Noms SQL
```
✅ Correct :
schema.sql
vitamin_deficiency.sql
queries.sql

❌ Incorrect :
Schema.SQL
database_create.sql
q1.sql
```

### Docstrings Python
```python
✅ Correct :
def clean_vitamin_data(df):
    """
    Nettoyer les données de vitamines.
    Supprime valeurs manquantes et aberrantes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Données brutes
    
    Returns:
    --------
    pd.DataFrame
        Données nettoyées
    """
    pass

❌ Incorrect :
def clean(d):
    # clean data
    pass
```

---

## 👥 Distribution des Tâches par Équipe

### **Analyste de Données**
**Dossiers :** `data/data_csv/` → `notebooks/` → `visualizations/`

**Tâches :**
- Charger et explorer les données (`01_eda.ipynb`)
- Identifier valeurs manquantes et aberrantes
- Créer graphiques et dashboards
- Écrire documentation (`data_dictionary.md`)

**Fichiers utilisés :**
```
data/data_csv/
notebooks/01_eda.ipynb
visualizations/dashboard.twbx
docs/data_dictionary.md
```

---

### **Ingénieur ML**
**Dossiers :** `src/` → `models/` → `notebooks/` → `test/`

**Tâches :**
- Écrire fonctions préparation (`src/preprocessing.py`)
- Créer features (`src/feature_engineering.py`)
- Entraîner modèles (`notebooks/03_model_training.ipynb`)
- Sauvegarder meilleur modèle (`models/model_v2.pkl`)
- Écrire tests (`test/test_model.py`)

**Fichiers utilisés :**
```
src/preprocessing.py
src/feature_engineering.py
src/model.py
src/recommendation.py
notebooks/02_data_cleaning.ipynb
notebooks/03_model_training.ipynb
models/model_v2.pkl
test/test_model.py
```

---

### **Développeur Web**
**Dossiers :** `app/` → `src/` → `models/` → `test/`

**Tâches :**
- Créer application Flask/FastAPI (`app/app.py`)
- Écrire templates HTML (`app/templates/`)
- Styliser CSS/JS (`app/static/`)
- Appeler fonctions ML (`src/model.py`)
- Écrire tests web (`test/test_app.py`)

**Fichiers utilisés :**
```
app/app.py
app/templates/
app/static/css/
app/static/js/
src/model.py
src/recommendation.py
models/model_v2.pkl
test/test_app.py
docs/API.md
```

---

### **Ingénieur BD (Database)**
**Dossiers :** `database_sql/` → `src/database.py`

**Tâches :**
- Créer schéma SQL (`database_sql/schema.sql`)
- Écrire requêtes (`database_sql/queries.sql`)
- Implémenter connexion Python (`src/database.py`)
- Documenter BD (`docs/database_schema.md`)

**Fichiers utilisés :**
```
database_sql/schema.sql
database_sql/vitamin_deficiency.sql
database_sql/ciqual_foods.sql
database_sql/queries.sql
src/database.py
docs/database_schema.md
```

---

### **Analyste Tableau**
**Dossiers :** `visualizations/` + `data/processed/`

**Tâches :**
- Créer dashboard Tableau
- Graphiques interactifs
- Connecter données `data/processed/`

**Fichiers utilisés :**
```
visualizations/dashboard.twbx
data/processed/
```

---

### **Chef de Projet**
**Dossiers :** Tous les dossiers

**Tâches :**
- Suivre avancement Roadmap
- Vérifier commits GitHub
- Maintenir documentation
- Coordonner l'équipe

**Fichiers utilisés :**
```
Roadmap_Project.md
README.md
docs/
.gitignore
```

---

## ✅ Bonnes Pratiques

### Écrire du Code Python

```python
# ✅ BON CODE

def calculate_vitamin_deficiency(patient_data):
    """
    Calculer le score de carence en vitamines.
    
    Parameters:
    -----------
    patient_data : dict
        Données du patient
    
    Returns:
    --------
    float
        Score de carence (0-100)
    
    Raises:
    -------
    ValueError
        Si données manquantes
    """
    required_fields = ['vitamin_a', 'vitamin_d', 'vitamin_b12']
    
    for field in required_fields:
        if field not in patient_data:
            raise ValueError(f"Champ manquant : {field}")
    
    score = sum([
        patient_data['vitamin_a'],
        patient_data['vitamin_d'],
        patient_data['vitamin_b12']
    ]) / 3
    
    return score

# Utilisation
try:
    score = calculate_vitamin_deficiency(patient)
    print(f"Score : {score}")
except ValueError as e:
    print(f"Erreur : {e}")
```

### Écrire des Notebooks

```markdown
# Titre du Notebook
## Section 1 : Exploration

### Charger les données
```python
import pandas as pd

df = pd.read_csv('data/data_csv/vitamin_deficiency_dataset.csv')
print(df.shape)
print(df.head())
```

### Analyser les données
- 4000 patients
- 34 colonnes
- 15% de valeurs manquantes

## Conclusion
...
```

### Nommage des Variables

```python
# ✅ Clair et explicite
patient_age = 35
vitamin_a_level = 85.5
has_symptom_fatigue = True

# ❌ Vague
x = 35
va = 85.5
has_fat = True
```

---

## 🎯 Résumé - Vue d'Ensemble

| Dossier | Contenu | Responsable | Outputs |
|---------|---------|-------------|---------|
| **app/** | Code web | Développeur Web | Application fonctionnelle |
| **data/** | Données CSV | Analyste données | data/processed/ |
| **database_sql/** | Schémas SQL | Ingénieur BD | Base de données |
| **docs/** | Documentation | Tous | Guides et références |
| **models/** | Modèles ML | Ingénieur ML | model_v2.pkl |
| **notebooks/** | Exploration | Analyste + ML | Notebooks validés |
| **src/** | Code Python | Ingénieur ML | Modules réutilisables |
| **test/** | Tests | ML + Web | Tests passants |
| **visualizations/** | Graphiques | Analyste Tableau | Dashboard + PNG |

---

## 📞 Problèmes Courants et Solutions

### "Où je mets mes données ?"
→ `data/data_csv/` pour les données brutes
→ `data/processed/` pour les données nettoyées

### "Comment j'utilise le modèle dans l'app ?"
```python
from src.model import load_model
model = load_model('models/model_v2.pkl')
prediction = model.predict(data)
```

### "Mes tests ne passent pas"
```bash
pytest test/ -v  # Verbose
pytest test/test_model.py -v  # Test spécifique
```

### "Où je documente mon code ?"
→ Docstrings dans le code
→ `docs/` pour la documentation générale

### "Comment je sauvegarde le modèle ?"
```python
from src.model import save_model
save_model(model, 'models/model_v3.pkl')
```

---

## 🚀 Démarrage

**Semaine 1 :**
1. Cloner le repo
2. Créer venv
3. Installer dépendances : `pip install -r requirements.txt`
4. Charger données : `data/data_csv/`
5. Créer BD : `mysql < database_sql/schema.sql`
6. Commencer exploration : `notebooks/01_eda.ipynb`

**Semaine 2-12 :**
- Suivre le Roadmap_Project.md
- Utiliser cette structure
- Committer régulièrement sur GitHub

---

*Ce guide a été créé pour le projet vitamin_IA - 12 semaines*
*Dernière mise à jour : Février 2026*
*Tous les membres doivent connaître ce document.*
**Règles Importantes :**
- ✅ Les fonctions doivent être **simples et faire une seule tâche**
- ✅ Écrire des docstrings : `"""Que fait cette fonction?"""`
- ✅ Gérer les erreurs (try-except)
- ⛔ Ne pas utiliser de variables globales

**Membres de l'équipe qui l'utilisent :**
- Ingénieur ML : écrira les modèles
- Développeur web : appellera ces fonctions dans l'app web

**Exemple d'utilisation :**
```python
from src.preprocessing import load_data, clean_missing_values
from src.model import train_model, predict

# Charger et nettoyer les données
data = load_data('data/raw/vitamin_deficiency_dataset.csv')
clean_data = clean_missing_values(data)

# Entraîner le modèle
model = train_model(clean_data)

# Faire une prédiction
prediction = predict(model, new_patient_data)
```

---

### 3. **notebooks/** - EXPLORATION ET EXPÉRIMENTATION
**Qu'est-ce qu'il contient ?** Fichiers Jupyter Notebook (analyse, tests, apprentissage)

**Fichiers :**
```
notebooks/
├── 01_eda.ipynb                 # Analyse Exploratoire des Données
├── 02_data_cleaning.ipynb       # Processus de Nettoyage
├── 03_model_training.ipynb      # Entraînement et Comparaison des Modèles
├── 04_recommendations.ipynb     # Moteur de Recommandations Nutritionnelles
└── 05_web_app_testing.ipynb     # Tests de l'Application Web
```

**Que doit faire chaque notebook ?**

**01_eda.ipynb :**
```
- Charger les données
- Statistiques de base (moyenne, écart-type, min, max)
- Montrer les valeurs manquantes (heatmap)
- Tracer les distributions (histogramme, box-plot)
- Analyser la matrice de corrélation
```

**02_data_cleaning.ipynb :**
```
- Tester les étapes de nettoyage
- Essayer différentes stratégies (suppression vs. imputation)
- Comparer les résultats
- Tester les fonctions à écrire dans src/preprocessing.py
```

**03_model_training.ipynb :**
```
- Essayer différents modèles (RF, XGBoost, etc.)
- Optimiser les hyperparamètres
- Comparer les performances
- Sélectionner le meilleur modèle et le sauvegarder
```

**Règles Importantes :**
- 📝 Écrire des explications en Markdown
- 🔍 Utiliser beaucoup de graphiques (clarté)
- ✅ Le code doit être propre et commenté
- 🚫 Ne pas nettoyer les données directement ici → utiliser `src/`
- 💾 Ne pas committer les fichiers `.ipynb_checkpoints/`

**Membres de l'équipe qui l'utilisent :**
- Analyste de données : explorations
- Ingénieur ML : tests des modèles

**Workflow :**
```
Tester dans le notebook → Copier les bonnes fonctions dans src/ → 
Utiliser src/ dans l'app web
```

---

### 4. **models/** - MODÈLES ENTRAÎNÉS
**Qu'est-ce qu'il contient ?** Fichiers .pkl et .joblib (modèles d'IA entraînés)

**Fichiers :**
```
models/
├── model_v1.pkl            # Modèle Random Forest
├── model_v2.pkl            # Modèle XGBoost (meilleur)
├── scaler.pkl              # Normaliseur de données (preprocessing)
├── label_encoder.pkl       # Encodeur de variables catégories
├── .gitkeep                # Marqueur de dossier vide
└── model_registry.json     # Versions et performances des modèles
```

**Comment l'utiliser ?**
```python
import pickle

# Charger le modèle
with open('models/model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

# Faire une prédiction
prediction = model.predict(new_data)
```

**Règles Importantes :**
- 📌 Utiliser des numéros de version (v1, v2, v3...)
- 📝 Mettre à jour `model_registry.json` (quand, performance)
- ⛔ NE PAS committer les modèles sur GitHub (.gitignore existe)
- 💾 App en production ? Copier le dernier modèle

**Membres de l'équipe qui l'utilisent :**
- Ingénieur ML : entraîne et sauvegarde
- Développeur web : utilise le modèle dans l'app

---

### 5. **app/** - APPLICATION WEB
**Qu'est-ce qu'il contient ?** Application Flask/FastAPI (interface utilisateur)

**Fichiers :**
```
app/
├── app.py                   # Application principale Flask/FastAPI
├── config.py                # Configuration (BD, API keys, etc.)
├── requirements.txt         # Dépendances (Flask, SQLAlchemy, etc.)
├── templates/               # Fichiers HTML
│   ├── index.html          # Page principale (formulaire patient)
│   ├── results.html        # Résultats de prédiction
│   └── recommendations.html # Recommandations nutritionnelles
└── static/                  # CSS, JavaScript, images
    ├── css/
    │   └── style.css        # Styles
    ├── js/
    │   └── script.js        # Interactivité
    └── images/
        └── logo.png
```

**Que fait app.py ?**
```python
from flask import Flask, render_template, request
from src.model import predict
from src.recommendation import get_recommendations

app = Flask(__name__)

@app.route('/')
def home():
    # Afficher la page principale
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # Récupérer les informations du patient
    patient_data = request.form.to_dict()
    
    # Faire la prédiction
    prediction = predict(patient_data)
    
    # Obtenir les recommandations
    recommendations = get_recommendations(prediction)
    
    # Afficher les résultats
    return render_template('results.html', 
                          prediction=prediction,
                          recommendations=recommendations)
```

**Que doit faire templates/index.html ?**
- Formulaire d'informations patient (âge, sexe, BMI, symptômes)
- Bouton de validation
- Validation des erreurs

**Que doit faire templates/results.html ?**
- Maladie prédite
- Score de confiance (%)
- Tableau de recommandations nutritionnelles
- Bouton "Retour"

**Règles Importantes :**
- 🔐 Valider les entrées utilisateur (âge > 0 ?)
- 🚀 Rapide (< 2 secondes)
- 📱 Responsive (mobile-friendly)
- 🐛 Messages d'erreur clairs

**Membres de l'équipe qui l'utilisent :**
- Développeur web : code et templates
- Designer UI/UX : HTML/CSS

---

### 6. **visualizations/** - GRAPHIQUES ET DASHBOARDS
**Qu'est-ce qu'il contient ?** Fichiers Tableau et graphiques statiques

**Fichiers :**
```
visualizations/
├── dashboard.twbx           # Dashboard Tableau (interactif)
├── plots/                   # Graphiques PNG/PDF
│   ├── vitamin_distribution.png
│   ├── symptom_frequency.png
│   ├── model_performance.png
│   └── heatmap_correlation.png
├── analysis_report.pdf      # Rapport avec graphiques
└── .gitkeep
```

**Contenu du Dashboard Tableau :**
- Démographie des patients (âge, sexe)
- Fréquence des carences en vitamines
- Relation symptôme-maladie
- Métriques de performance du modèle
- Analyse géographique (carte)

**Graphiques Statiques :**
- Créés avec Matplotlib/Seaborn
- À inclure dans le rapport
- Format PNG (600 DPI qualité)

**Règles Importantes :**
- 📊 Chaque graphique doit avoir un titre et une légende
- 🎨 Couleurs accessibles (daltonisme)
- 📈 L'axe Y commence à 0 (pas de tromperie)

**Membres de l'équipe qui l'utilisent :**
- Analyste de données : crée le dashboard
- Rapportage : intègre les graphiques

---

### 7. **tests/** - TESTS UNITAIRES
**Qu'est-ce qu'il contient ?** Tests Python (pytest, unittest)

**Fichiers :**
```
tests/
├── test_preprocessing.py    # Tests du nettoyage
├── test_model.py            # Tests du modèle
├── test_recommendation.py   # Tests des recommandations
├── test_app.py              # Tests de l'app web
└── .gitkeep
```

**Exemple : test_preprocessing.py :**
```python
import pytest
from src.preprocessing import clean_missing_values, scale_features

def test_clean_missing_values():
    # Données de test
    data = pd.DataFrame({
        'age': [25, None, 35],
        'vitamin_a': [100, 200, None]
    })
    
    # Nettoyer
    cleaned = clean_missing_values(data)
    
    # Vérifier qu'il n'y a plus de valeurs manquantes
    assert cleaned.isnull().sum().sum() == 0

def test_scale_features():
    data = pd.DataFrame({'feature': [0, 50, 100]})
    scaled = scale_features(data)
    
    # Doit être entre 0-1
    assert scaled['feature'].min() >= 0
    assert scaled['feature'].max() <= 1
```

**Pourquoi les tests sont importants ?**
- ✅ Trouve les erreurs après l'écriture
- ✅ Les modifications de code ne cassent rien
- ✅ L'équipe travaille en confiance

**Membres de l'équipe qui l'utilisent :**
- Ingénieur ML : écrit les tests
- QA : exécute tous les tests

**Exécuter les tests :**
```bash
pytest tests/
```

---

### 8. **docs/** - DOCUMENTATION
**Qu'est-ce qu'il contient ?** Explications détaillées écrites sur le projet

**Fichiers :**
```
docs/
├── methodology.md           # Méthodologie ML
├── API.md                   # Documentation des endpoints API
├── data_dictionary.md       # Dictionnaire des données
├── model_performance.md     # Détails des métriques
└── deployment.md            # Étapes de déploiement
```

**Exemple : data_dictionary.md :**
```markdown
## Dataset Vitamin Deficiency

| Nom Colonne | Type | Description | Exemple |
|-------------|------|-------------|---------|
| age | int | Âge du patient | 25 |
| vitamin_a_percent_rda | float | Vitamine A (% besoins quotidiens) | 85.5 |
| has_night_blindness | bool | Symptôme cécité nocturne ? | True |
| disease_diagnosis | str | Maladie diagnostiquée | Vitamin_A_Deficiency |
```

**Exemple : API.md :**
```markdown
## API de Prédiction

### POST /predict
Faire une prédiction de maladie

Requête :
```json
{
  "age": 35,
  "vitamin_a_percent_rda": 60,
  "symptoms_count": 3
}
```

Réponse :
```json
{
  "disease": "Vitamin_A_Deficiency",
  "confidence": 0.89
}
```

**Membres de l'équipe qui l'utilisent :**
- Tout le monde : lit la documentation
- Nouvelles recrues : apprennent rapidement

---

## 📄 Types de Fichiers et Conventions

### Noms de Fichiers Python
```
✅ Correct :
- preprocessing.py          # snake_case
- feature_engineering.py
- calculate_vitamin_score.py

❌ Incorrect :
- PreProcessing.py          # CamelCase
- Feature-Engineering.py    # Tirets
- calc_vit_score.py         # Abréviations
```

### Noms de Notebooks
```
✅ Correct :
01_eda.ipynb
02_data_cleaning.ipynb
03_model_training.ipynb

❌ Incorrect :
EDA.ipynb
data_cleaning(final).ipynb
model_training_v5_final_REAL.ipynb
```

### Messages de Commit Git
```
✅ Correct :
git commit -m "Add preprocessing functions: clean_missing_values, scale_features"
git commit -m "Train XGBoost model with accuracy 0.87"
git commit -m "Create web app API endpoints for prediction"

❌ Incorrect :
git commit -m "update"
git commit -m "fix bug"
git commit -m "asdfgh"
```

---

## 👥 Distribution des Tâches par Équipe

### Analyste de Données
**Responsable de :**
- Examiner les données dans `data/raw/`
- Écrire `notebooks/01_eda.ipynb`
- Trouver les valeurs manquantes et aberrantes
- Préparer graphiques et statistiques

**Fichiers utilisés :**
```
data/raw/ → notebooks/01_eda.ipynb → visualizations/plots/
```

### Ingénieur ML
**Responsable de :**
- Fonctions de nettoyage → `src/preprocessing.py`
- Feature engineering → `src/feature_engineering.py`
- Entraînement modèle → `notebooks/03_model_training.ipynb`
- Sauvegarder le meilleur modèle → `models/model_vX.pkl`
- Écrire les tests → `tests/test_model.py`

**Fichiers utilisés :**
```
data/processed/ + src/ + notebooks/ + models/ + tests/
```

### Développeur Web
**Responsable de :**
- Créer l'app Flask/FastAPI → `app/app.py`
- Écrire les templates HTML → `app/templates/`
- Styles CSS/JS → `app/static/`
- Tests web → `tests/test_app.py`
- Documentation API → `docs/API.md`

**Fichiers utilisés :**
```
src/ + models/ + app/ + tests/
```

### Analyste Tableau
**Responsable de :**
- Créer le dashboard → `visualizations/dashboard.twbx`
- Graphiques interactifs
- Graphiques prêts pour la présentation

**Fichiers utilisés :**
```
data/processed/ → visualizations/
```

### Chef de Projet
**Responsable de :**
- Suivi du Roadmap
- Vérifier les commits GitHub
- Rappeler les deadlines
- Mettre à jour la documentation

**Fichiers utilisés :**
```
Roadmap_Project.md + docs/ + README.md
```

---

## ✅ Bonnes Pratiques (Best Practices)

### Écrire du Code Python
```python
# ✅ Correct
def clean_vitamin_data(df):
    """
    Nettoie les données de carences en vitamines.
    Supprime les valeurs manquantes et les valeurs aberrantes.
    
    Paramètres :
    -----------
    df : pd.DataFrame
        Ensemble de données à nettoyer
    
    Retour :
    --------
    pd.DataFrame
        Ensemble de données nettoyé
    """
    df = df.dropna()
    df = df[df['age'] > 0]
    return df

# ❌ Incorrect
def clean(d):
    d = d.dropna()
    return d
```

### Écrire des Notebooks
```markdown
# 1. Titre et Description
## Exploration des Données - Dataset Vitamin Deficiency
Dans ce notebook, nous examinons le dataset de carences...

# 2. Code
```python
df = pd.read_csv('data/raw/vitamin_deficiency_dataset.csv')
```

# 3. Conclusion
- Le dataset contient 4000 lignes et 34 colonnes
- 15% de valeurs manquantes
- Distribution des symptômes...
```

---

## 📞 Questions et Aide

**Si vous vous posez des questions sur :**
- Où mettre un fichier ? → Consultez cette documentation
- Comment écrire le code ? → Regardez les exemples dans `src/`
- Git ? → `git help`
- Performance du modèle ? → Demandez à l'ingénieur ML
- App cassée ? → Appelez le développeur web

---

## 🎯 Résumé

| Dossier | Qui ? | Quoi ? |
|---------|-------|--------|
| **data/** | Analyste de données | Gestion des données |
| **src/** | Ingénieur ML | Code réutilisable et propre |
| **notebooks/** | Analyste + ML | Exploration et tests |
| **models/** | Ingénieur ML | Modèles entraînés |
| **app/** | Développeur Web | Application web |
| **visualizations/** | Analyste Tableau | Graphiques |
| **tests/** | ML + Web | Tests automatisés |
| **docs/** | Toute l'équipe | Documentation |

**Ensemble, nous créerons un excellent projet ! 🚀**

---

*Ce guide a été créé pour le projet de prédiction des carences en vitamines sur 10 semaines.*
*Mise à jour : Février 2026*