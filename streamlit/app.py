import streamlit as st

st.set_page_config(
    page_title="VitaIA",
    page_icon="💊",
    layout="wide",
)

# ── TODO (équipe) : ajoutez votre CSS ici ─────────────────────────────────────
# st.markdown("<style> ... </style>", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────

st.title("💊 VitaIA")
st.subheader("Prédiction des carences en vitamines par IA")

st.markdown("""
Bienvenue sur **VitaIA**, un outil d'aide à la décision médicale basé sur le Machine Learning.

**Comment ça marche :**
1. Allez sur la page **Diagnostic** et renseignez les données cliniques du patient
2. Le modèle Random Forest prédit la carence probable
3. Des recommandations alimentaires issues de la base **CIQUAL (ANSES)** sont proposées

---
""")

col1, col2, col3 = st.columns(3)

col1.metric("F1-Score (test)", "94.3%")
col2.metric("Patients (dataset)", "4 000")
col3.metric("Features cliniques", "21")

st.markdown("---")
st.caption("Projet universitaire · Machine Learning · Données Kaggle & CIQUAL (ANSES)")
st.caption("Lydia Moutchachou · Hazem Ibnmtar · Ahmed Bekakria · Serdar Varol")
