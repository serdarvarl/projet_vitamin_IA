# Guide — Lancer l'application VitaIA

## Prérequis

- **Python 3.10+** installé sur la machine
- **Git** installé
- Le dépôt cloné en local :
  ```bash
  git clone https://github.com/serdarvarl/projet_vitamin_IA.git
  cd projet_vitamin_IA
  ```

---

## Installation (une seule fois)

### 1. Créer l'environnement virtuel

```bash
python -m venv .venv
```

### 2. Activer l'environnement

**Windows (PowerShell) :**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Git Bash / MINGW64) :**
```bash
source .venv/Scripts/activate
```

**Mac / Linux :**
```bash
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r streamlit/requirements.txt
```

---

## Générer le modèle ML (une seule fois)

Le modèle `.pkl` doit être généré avant de lancer l'app.

1. Ouvre **Jupyter** :
   ```bash
   jupyter notebook
   ```
2. Lance le notebook `notebooks/modeles_IA_v3.ipynb` en entier
   *(ou `modeles_IA_v5_holdout_final.ipynb` pour le modèle final)*
3. Les fichiers `.pkl` sont automatiquement sauvegardés dans `notebooks/models_v3/`

---

## Lancer l'application

Depuis la racine du projet (`projet_vitamin_IA/`) :

**Windows (PowerShell) :**
```powershell
.venv\Scripts\streamlit run streamlit\app.py
```

**Windows (Git Bash) :**
```bash
.venv/Scripts/streamlit run streamlit/app.py
```

**Mac / Linux :**
```bash
.venv/bin/streamlit run streamlit/app.py
```

Le navigateur s'ouvre automatiquement sur **http://localhost:8501**

---

## Structure de l'application

```
streamlit/
├── app.py                  → Page d'accueil (métriques, équipe)
└── pages/
    ├── 1_Diagnostic.py     → Formulaire patient + prédiction + recommandations CIQUAL
    └── 2_Presentation.py   → Résultats de l'étude (graphiques, ablation)
```

---

## Fichiers nécessaires

| Fichier | Rôle | Où |
|---|---|---|
| `notebooks/models_v3/best_model_v3.pkl` | Modèle Random Forest | Généré par le notebook |
| `notebooks/models_v3/label_encoder_v3.pkl` | Encodage des classes | Généré par le notebook |
| `notebooks/models_v3/feature_cols_v3.pkl` | Liste des features | Généré par le notebook |
| `data_csv/raw/Table_Ciqual_V2.ods` | Base alimentaire CIQUAL/ANSES | Déjà dans le repo |
| `docs/*.png` | Figures pour la page Résultats | Déjà dans le repo |

---

## Problèmes courants

| Erreur | Solution |
|---|---|
| `ModuleNotFoundError: streamlit` | Relancer `pip install streamlit` dans le venv activé |
| `ModuleNotFoundError: plotly` | `pip install plotly` |
| `Aucun modèle trouvé` | Lancer le notebook pour générer les `.pkl` |
| `streamlit: command not found` | Utiliser `.venv/Scripts/streamlit run ...` (chemin complet) |
| Page blanche au démarrage | Appuyer sur **F5** dans le navigateur |

---

## Mettre à jour le code

```bash
git pull
```

Puis relancer l'app normalement.
