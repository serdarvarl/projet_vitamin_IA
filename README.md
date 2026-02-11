
# 

## Objectif de prédiction :

Développer un système capable de prédire les carences en vitamines à partir de données cliniques, biologiques et comportementales, puis proposer automatiquement des solutions nutritionnelles ciblées basées sur la base CIQUAL.


## Problamatique 

Aide à la Décision Médicale (Curatif) : Prédire le type de carence en vitamines
chez un patient à partir de ses symptômes cliniques, marqueurs biologiques et apports
nutritionnels actuels. Une fois la carence identifiée, le système recommande automatiquement
des aliments riches en la vitamine manquante via la base CIQUAL.


### Base de données 1: Vitamin Deficiency Disease Prediction Dataset [Kaggle](https://www.kaggle.com/datasets/nudratabbas/vitamin-deficiency-disease-prediction-dataset?select=vitamin_deficiency_disease_dataset_20260123.csv)

Vitamin Deficiency Disease Prediction Dataset
Contient 4 catégories de variables pour chaque patient :
 
*   Facteurs sociodémographiques et comportementaux (10): age, gender, BMI,
smoking_status, alcohol_consumption, exercise_level, diet_type, sun_exposure,
income_level, latitude_region
*   Apports nutritionnels en % RDA (8): vitamin_a, vitamin_c, vitamin_d, vitamin_e,
vitamin_b12, folate, calcium, iron
*   Marqueurs biologiques (4): hemoglobin, serum_vitamin_d, serum_vitamin_b12,
serum_folate (mesures sanguines réelles)
*   Données cliniques et symptomatiques (11): symptoms_count, symptoms_list,
has_night_blindness, has_fatigue, has_bleeding_gums, has_bone_pain,
has_muscle_weakness, has_numbness_tingling, has_memory_problems, has_pale_skin,
disease_diagnosis (RÉPONSE À PRÉDIRE), has_multiple_deficiencies

## Base de données 2: Table CIQUAL - Composition Nutritionnelle des Aliments (ANSES) [Ciqual](https://ciqual.anses.fr/#/cms/telechargement/node/20)
Base officielle française extrêmement détaillée contenant la composition de milliers d'aliments
avec:
*   Identificateurs : codes aliments, groupes alimentaires (viandes, produits laitiers, fruits,
légumes, etc.)
*   Macronutriments : énergie, eau, protéines, glucides, lipides, fibres, acides gras détaillés
*   Minéraux (12) : Calcium, Fer, Magnésium, Zinc, Cuivre, Iode, Phosphore, Potassium,
Sélénium, Sodium, Chlorure, Manganèse
*   Vitamines (18) : Vitamine A (+ rétinol, beta-carotène), Vitamine C, Vitamine D (D2, D3),
Vitamine E, Vitamines K (K1, K2), et toutes les vitamines B (B1, B2, B3, B5, B6, B9/Folates,
B12)
__________



## Architecture du Système

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────────┐
│  Données Patient    │────▶│  Random Forest       │────▶│  Carence Prédite        │
│  (Kaggle, 4K lignes)│     │  Classifier          │     │  (ex : Vitamine D)      │
└─────────────────────┘     └──────────────────────┘     └───────────┬─────────────┘
                                                                     │
                                                                     ▼
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────────┐
│  Base CIQUAL        │────▶│  Filtrage &           │────▶│  Recommandations        │
│  (Aliments FR)      │     │  Classement Nutritif  │     │  Alimentaires (Top-N)   │
└─────────────────────┘     └──────────────────────┘     └─────────────────────────┘
```

---


## Technologies Utilisées

| Catégorie | Outils |
|-----------|--------|
| Langage | Python 3 |
| Machine Learning | scikit-learn (Random Forest) |
| Traitement de données | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| ML Visuel | Orange Data Mining |
| Gestion de version | Git / GitHub |

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/serdarvarl/projet_vitamin_IA.git
cd projet_vitamin_IA

# Installer les dépendances
pip install -r requirements.txt
```

---

## Composition du Groupe

| Membre | Rôle |
|--------|------|
| Lydia Moutchachou |_À compléter_ |
| Hazem Ibnmtar | _À compléter_ |
| Ahmed Bekakria | _À compléter_ |
| Serdar Varol | _À compléter_ |


---


## Licence

Projet réalisé dans le cadre d'un cours universitaire de Machine Learning.