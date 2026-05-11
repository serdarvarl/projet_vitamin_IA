Tamam! İşte guide:

---

## 🚀 VitaIA — Guide de démarrage rapide

### 1. Pull et installation

```bash

cd streamlit
pip install -r requirements.txt
```

### 2. Générer les fichiers modèle

Les `.pkl` ne sont pas sur GitHub. Il faut les générer depuis le notebook :

- Ouvrir `notebooks/modeles_IA_v3.ipynb` dans Jupyter
- Exécuter toutes les cellules (**Run All**)
- Les 3 fichiers sont automatiquement sauvegardés dans `models_v3/` :
  - `best_model_v3.pkl`
  - `label_encoder_v3.pkl`
  - `feature_cols_v3.pkl`

> ⚠️ Cette étape prend quelques minutes (entraînement du modèle).

### 3. Lancer l'application

```bash
streamlit run app.py
```

Le navigateur s'ouvre sur `http://localhost:8501`

---

### Pages disponibles

| Page | Description |
|------|-------------|
| 🏠 Accueil | Présentation générale et statistiques |
| 🩺 Diagnostic | Formulaire patient → prédiction → recommandations CIQUAL |
| 📊 Présentation | Méthodologie, performances, équipe |

---

> ⚠️ Si tu vois un warning sklearn version → normal, le modèle fonctionne quand même.