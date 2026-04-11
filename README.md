# VitaIA — Prédiction des carences en vitamines par Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Live Demo](https://img.shields.io/badge/Demo-vitaminia.streamlit.app-27ae60?logo=streamlit&logoColor=white)](https://vitaminia.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-serdarvarl%2Fprojet__vitamin__IA-181717?logo=github)](https://github.com/serdarvarl/projet_vitamin_IA)

**Projet universitaire — L3 MIASHS · Université Paul-Valéry Montpellier 3 · 2025–2026**

---

## Démo en ligne

**[vitaminia.streamlit.app](https://vitaminia.streamlit.app)**

L'application est déployée sur Streamlit Community Cloud et accessible sans installation.

---

## Présentation

VitaIA est un système de classification automatique des carences en vitamines développé dans le cadre d'un projet de Machine Learning en L3 MIASHS. À partir des données cliniques, biologiques et comportementales d'un patient, le modèle prédit parmi cinq diagnostics :

| Diagnostic | Description |
|---|---|
| **Healthy** | Aucune carence détectée |
| **Anemia** | Carence en fer / vitamine B12 / folate |
| **Rickets_Osteomalacia** | Carence en vitamine D / calcium |
| **Night_Blindness** | Carence en vitamine A (rétinol) |
| **Scurvy** | Carence sévère en vitamine C |

Une fois la carence identifiée, l'application propose automatiquement des aliments correcteurs issus de la base officielle **CIQUAL (ANSES)**.

> **Avertissement :** Cet outil est à usage pédagogique. Il ne remplace pas un avis médical professionnel.

---

## Performances du modèle final (v5)

| Métrique | Valeur |
|---|---|
| F1-Score pondéré (holdout 30 %) | **94,06 %** |
| Exactitude (holdout 30 %) | **94,00 %** |
| F1 Scorbut | **1,00** (détection parfaite) |
| Algorithme | Random Forest (340 arbres) |
| Variables | 30 (biomarqueurs + symptômes + lifestyle) |
| Protocole | Holdout propre — jeu de test jamais vu lors du développement |

L'étude d'ablation à 4 dimensions a établi la hiérarchie des contributions :
- **Biomarqueurs sériques** : composante critique (−17,4 % F1 sans eux)
- **Variables lifestyle** : gain le plus élevé (+5,3 % F1, résultat contre-intuitif)
- **SMOTE** : protection du rappel sur classes rares (Scorbut, Cécité nocturne)
- **Normalisation** : effet neutre pour Random Forest

---

## Démarche : Analyse exploratoire avant le Machine Learning

Avant d'attaquer la modélisation, une phase d'analyse exploratoire des données (EDA) a été réalisée dans deux notebooks dédiés :

### `data_vis.ipynb` — Exploration du dataset principal
- Distribution de la variable cible (`disease_diagnosis`) : identification du déséquilibre sévère (ratio 15,9:1 entre Healthy et Scurvy)
- Visualisation des distributions des biomarqueurs par classe (boxplots, violin plots)
- Analyse des apports nutritionnels moyens par type de carence
- Corrélations entre variables (heatmap) : détection des variables redondantes
- Distribution des variables catégorielles (genre, régime alimentaire, exposition solaire...)

### `data2_vis.ipynb` — Analyse approfondie et préparation
- Étude des outliers sur les biomarqueurs sériques
- Analyse des symptômes cliniques : fréquence par classe de maladie
- Vérification des valeurs manquantes et des incohérences
- Visualisation des corrélations entre variables lifestyle et diagnostics
- Identification des features les plus discriminantes avant modélisation

Cette étape d'EDA a guidé directement nos choix : elle a révélé le déséquilibre des classes (→ SMOTE), la forte variance des biomarqueurs entre classes (→ StandardScaler), et la pertinence potentielle des variables lifestyle (→ confirmée par l'étude d'ablation).

---

## Architecture du système

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────────┐
│  Données Patient    │────▶│  Random Forest       │────▶│  Carence Prédite        │
│  (Kaggle, 4K lignes)│     │  340 arbres, 30 var. │     │  (ex : Vitamine D)      │
└─────────────────────┘     └──────────────────────┘     └───────────┬─────────────┘
                                                                     │
                                                                     ▼
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────────┐
│  Base CIQUAL        │────▶│  Filtrage &           │────▶│  Recommandations        │
│  (ANSES, aliments)  │     │  Classement Nutritif  │     │  Alimentaires (Top-N)   │
└─────────────────────┘     └──────────────────────┘     └─────────────────────────┘
```

---

## Sources de données

### Dataset 1 — Vitamin Deficiency Disease Prediction Dataset (Kaggle)
- **Source :** [Kaggle — nudratabbas](https://www.kaggle.com/datasets/nudratabbas/vitamin-deficiency-disease-prediction-dataset)
- **Taille :** 4 000 observations, 34 variables réparties en 4 groupes :
  - **Socio-démographiques / comportementales (10)** : âge, genre, IMC, tabac, alcool, activité physique, type de régime, exposition solaire, revenu, région
  - **Apports nutritionnels en % AJR (8)** : vitamines A, C, D, E, B12, folate, calcium, fer
  - **Biomarqueurs sériques (4)** : hémoglobine, vitamine D sérique, B12 sérique, folate sérique
  - **Symptômes cliniques (8)** : cécité nocturne, fatigue, gingivorragies, douleurs osseuses, faiblesse musculaire, engourdissements, troubles mnésiques, pâleur cutanée
- **Variable cible :** `disease_diagnosis` (5 classes)
- **Déséquilibre :** ratio 15,9:1 (Healthy vs Scurvy) → justifie SMOTE

### Dataset 2 — Table CIQUAL (ANSES)
- **Source :** [ciqual.anses.fr](https://ciqual.anses.fr/#/cms/telechargement/node/20)
- Base officielle française de composition nutritionnelle des aliments
- Utilisée pour générer les recommandations alimentaires par carence

---

## Structure du projet

```
projet_vitamin_IA/
│
├── README.md                          # Ce fichier
├── .gitignore
│
├── data_csv/
│   └── raw/
│       ├── vitamin_deficiency_disease_dataset_20260123.ods   # Dataset Kaggle
│       └── Table_Ciqual_V2.ods                               # Base CIQUAL ANSES
│
├── notebooks/
│   ├── data_vis.ipynb                 # EDA — Exploration & visualisation du dataset principal (Kaggle)
│   ├── data2_vis.ipynb                # EDA — Analyse approfondie, distributions, corrélations, outliers
│   ├── modeles_IA.ipynb               # v1 — Premier modèle Random Forest (baseline)
│   ├── modeles_IA_v2.ipynb            # v2 — Comparaison RF / SVM / k-NN
│   ├── modeles_IA_v3.ipynb            # v3 — Étude d'ablation (4 configurations)
│   ├── modeles_IA_v4.ipynb            # v4 — Configuration étendue + variables lifestyle
│   ├── modeles_IA_v5_holdout_final.ipynb  # v5 — Évaluation finale non biaisée (holdout propre)
│   └── models_final/                  # Modèles sérialisés (pkl)
│       ├── random_forest_final.pkl    # Modèle v5 (340 arbres, 30 variables)
│       ├── scaler_final.pkl           # StandardScaler (ajusté sur train)
│       ├── label_encoder_final.pkl    # LabelEncoder (5 classes)
│       └── feature_cols_final.pkl     # Liste ordonnée des 30 features
│
├── docs/
│   └── rapport_IA.tex                 # Rapport LaTeX complet (partie IA)
│
└── streamlit/
    ├── app.py                         # Page d'accueil (Accueil)
    ├── pages/
    │   ├── 1_Diagnostic.py            # Formulaire de diagnostic + CIQUAL
    │   └── 2_Presentation.py          # Métriques, graphiques, ablation
    ├── assets/
    │   └── logos/
    │       └── logo2_pilule_neurone.svg
    └── .streamlit/
        └── config.toml                # Thème Streamlit (light, navy)
```

---

## Installation locale

### Prérequis
- Python 3.10 ou supérieur
- pip

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/serdarvarl/projet_vitamin_IA.git
cd projet_vitamin_IA

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install streamlit scikit-learn imbalanced-learn pandas numpy plotly odfpy

# 4. Lancer l'application Streamlit
cd streamlit
streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`.

> **Note :** Les fichiers `.pkl` du modèle sont déjà présents dans `notebooks/models_final/` et seront chargés automatiquement. Si vous souhaitez ré-entraîner le modèle, exécutez le notebook `modeles_IA_v5_holdout_final.ipynb`.

---

## Technologies utilisées

| Catégorie | Outil | Usage |
|---|---|---|
| Langage | Python 3.10+ | Développement complet |
| Machine Learning | scikit-learn | Random Forest, SVM, k-NN, StandardScaler, LabelEncoder |
| Rééchantillonnage | imbalanced-learn | SMOTE (Synthetic Minority Over-sampling Technique) |
| Traitement données | pandas, NumPy | Manipulation et prétraitement |
| Visualisation | Plotly, matplotlib, seaborn | Graphiques interactifs et statiques |
| Application web | Streamlit | Interface utilisateur déployée |
| Données alimentaires | CIQUAL / ANSES | Recommandations nutritionnelles |
| Gestion de version | Git / GitHub | Versionnage, collaboration |
| Déploiement | Streamlit Community Cloud | Hébergement en ligne |

---

## Composition de l'équipe

| Membre | GitHub | Contributions |
|---|---|---|
| **Ibnmtar Hazem** | [@IbnmtarHazem](https://github.com/IbnmtarHazem) | Prétraitement et exploration des données, implémentation k-NN et SVM, validation croisée (StratifiedKFold) |
| **Moutchachou Lydia** | [@lydiamtch](https://github.com/lydiamtch) | Visualisation de la base CIQUAL, étude d'ablation, réalisation de l'évaluation finale (holdout v5), contribution au rapport |
| **Varol Serdar** | [@serdarvarl](https://github.com/serdarvarl) | Déploiement Streamlit Cloud, visualisation de la base principale, Machine Learning, modélisation sous Orange Data Mining, gestion GitHub et aspects techniques |
| **Bekakria Ahmed** | [@ahmed-abc73](https://github.com/ahmed-abc73) | Développement Machine Learning, développement web (Streamlit), choix et justification de l'algorithme, implémentation SMOTE, rédaction du rapport |

---

## Références

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357.
- Cortes, C., & Vapnik, V. (1995). *Support-vector networks*. Machine Learning, 20(3), 273–297.
- Cover, T., & Hart, P. (1967). *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory, 13(1), 21–27.
- Scikit-learn developers. (2024). *scikit-learn: Machine Learning in Python*. [scikit-learn.org](https://scikit-learn.org)
- ANSES. (2020). *Table de composition nutritionnelle des aliments Ciqual 2020*. [ciqual.anses.fr](https://ciqual.anses.fr)
- Dataset Kaggle : Nudrat Abbas. *Vitamin Deficiency Disease Prediction Dataset*. [kaggle.com](https://www.kaggle.com/datasets/nudratabbas/vitamin-deficiency-disease-prediction-dataset)

---

## Licence

Projet réalisé dans le cadre d'un cours universitaire de Machine Learning — L3 MIASHS, Université Paul-Valéry Montpellier 3, 2025–2026. Usage pédagogique uniquement.
