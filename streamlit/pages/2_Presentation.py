import streamlit as st
import plotly.graph_objects as go

# TODO (équipe) : ajoutez votre CSS ici
# st.markdown("<style> ... </style>", unsafe_allow_html=True)

st.title("📊 Présentation du projet")
st.markdown("---")

# ── Résumé ────────────────────────────────────────────────────────────────────
st.header("Le projet")

# TODO (équipe) : modifiez ce texte
st.markdown("""
**VitaIA** est un système d'aide à la décision médicale développé dans le cadre
d'un projet universitaire de Machine Learning.

**Objectif :** Prédire les carences en vitamines à partir de données cliniques et biologiques,
puis recommander des aliments adaptés via la base CIQUAL (ANSES).
""")

st.markdown("---")

# ── Architecture ──────────────────────────────────────────────────────────────
st.header("Architecture du système")

st.markdown("""
```
Données patient (21 features)
        ↓
Random Forest Classifier
(StandardScaler → SMOTE → RF)
        ↓
Carence prédite
(Anémie / Scorbut / Rachitisme / Cécité nocturne / Sain)
        ↓
Recommandations CIQUAL
(Top-N aliments riches en la vitamine manquante)
```
""")

st.markdown("---")

# ── Performances ──────────────────────────────────────────────────────────────
st.header("Performances du modèle")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Test Accuracy",  "94.25%")
col2.metric("Test F1-Score",  "94.31%")
col3.metric("CV F1-Score",    "96.01%")
col4.metric("N estimators",   "340 arbres")

st.markdown("**Comparaison des modèles :**")

models    = ["RF v2 (manuel)", "RF v3 (tuned)", "SVM v3", "KNN v3"]
f1_scores = [0.9444,           0.9431,          0.8926,   0.8208]

fig = go.Figure(go.Bar(
    x=models,
    y=f1_scores,
    text=[f"{v:.1%}" for v in f1_scores],
    textposition="outside",
))
fig.update_layout(
    yaxis=dict(range=[0.7, 1.0], title="F1-Score", tickformat=".0%"),
    title="F1-Score par modèle (test set)",
    height=350,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Données ───────────────────────────────────────────────────────────────────
st.header("Sources de données")

d1, d2 = st.columns(2)

with d1:
    st.subheader("Kaggle — Vitamin Deficiency Dataset")
    st.markdown("""
    - 4 000 patients · 34 colonnes · 5 classes
    - 10 facteurs socio-démographiques
    - 8 apports nutritionnels (%AJR)
    - 4 marqueurs biologiques
    - 9 symptômes cliniques
    """)

with d2:
    st.subheader("CIQUAL — Table ANSES")
    st.markdown("""
    - Base officielle française
    - Composition nutritionnelle de milliers d'aliments
    - 18 vitamines · 12 minéraux · macronutriments
    - Groupes alimentaires détaillés
    """)

st.markdown("---")

# ── Équipe ────────────────────────────────────────────────────────────────────
st.header("L'équipe")

# TODO (équipe) : complétez les rôles et ajoutez vos photos si vous voulez
team = [
    {"nom": "Lydia Moutchachou", "role": "TODO"},
    {"nom": "Hazem Ibnmtar",     "role": "TODO"},
    {"nom": "Ahmed Bekakria",    "role": "TODO"},
    {"nom": "Serdar Varol",      "role": "TODO"},
]

cols = st.columns(len(team))
for col, membre in zip(cols, team):
    col.markdown(f"**{membre['nom']}**")
    col.caption(membre["role"])

st.markdown("---")
st.caption("Projet universitaire · Machine Learning · Données Kaggle & CIQUAL (ANSES) · Licence MIT")
