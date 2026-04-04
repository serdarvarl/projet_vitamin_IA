import streamlit as st

st.set_page_config(
    page_title="VitaIA — Prédiction des carences en vitamines",
    page_icon="💊",
    layout="wide",
)

st.markdown("""
<style>
    /* Header principal */
    .vita-hero {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .vita-hero h1 { font-size: 3rem; margin: 0; letter-spacing: -1px; }
    .vita-hero p  { font-size: 1.15rem; margin: 0.5rem 0 0; opacity: 0.88; }

    /* Cartes métriques personnalisées */
    .metric-card {
        background: white;
        border: 1px solid #e3e8f0;
        border-radius: 10px;
        padding: 1.2rem 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card .val { font-size: 2rem; font-weight: 700; color: #1a237e; }
    .metric-card .lbl { font-size: 0.82rem; color: #666; margin-top: 4px; }

    /* Encart "Comment ça marche" */
    .step-box {
        background: #f5f7ff;
        border-left: 4px solid #3949ab;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
        font-size: 0.95rem;
    }
    .step-box strong { color: #1a237e; }

    /* Team */
    .team-chip {
        display: inline-block;
        background: #e8eaf6;
        color: #1a237e;
        border-radius: 20px;
        padding: 4px 14px;
        margin: 4px;
        font-size: 0.88rem;
        font-weight: 500;
    }

    /* Cacher le menu Streamlit par défaut */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vita-hero">
    <h1>💊 VitaIA</h1>
    <p>Prédiction des carences en vitamines par apprentissage automatique</p>
</div>
""", unsafe_allow_html=True)

# ── Métriques ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("94,06 %", "F1-Score holdout (v5)"),
    ("94,00 %", "Exactitude holdout"),
    ("4 000",   "Patients (dataset)"),
    ("30",      "Variables cliniques"),
]
for col, (val, lbl) in zip([c1, c2, c3, c4], metrics):
    col.markdown(f"""
    <div class="metric-card">
        <div class="val">{val}</div>
        <div class="lbl">{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Présentation ──────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("### À propos du projet")
    st.markdown("""
    **VitaIA** est un système de classification automatique des carences en vitamines
    développé dans le cadre d'un projet universitaire L3 MIASHS.

    Le modèle **Random Forest** (340 arbres) prédit parmi 5 diagnostics possibles
    à partir du profil clinique, biologique et comportemental d'un patient :

    | Diagnostic | Description |
    |---|---|
    | **Healthy** | Aucune carence détectée |
    | **Anemia** | Carence en fer / vitamine B12 / folate |
    | **Rickets\_Osteomalacia** | Carence en vitamine D / calcium |
    | **Night\_Blindness** | Carence en vitamine A |
    | **Scurvy** | Carence sévère en vitamine C |
    """)

    st.markdown("### Comment utiliser l'application")
    for step in [
        ("1. Diagnostic", "Renseignez les données cliniques du patient dans le formulaire"),
        ("2. Prédiction", "Le modèle retourne le diagnostic probable avec un score de confiance"),
        ("3. Recommandations", "Des aliments correcteurs issus de la base CIQUAL (ANSES) sont proposés"),
        ("4. Résultats", "Consultez les performances détaillées du modèle et l'étude d'ablation"),
    ]:
        st.markdown(f"""
        <div class="step-box">
            <strong>{step[0]}</strong> — {step[1]}
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown("### Résultats clés")
    st.markdown("""
    L'étude d'**ablation à 4 dimensions** a révélé la hiérarchie des contributions :

    - 🔴 **Biomarqueurs sériques** : composante critique (−17,4 % F1 sans eux)
    - 🟢 **Variables lifestyle** : gain majeur (+5,3 % F1, contre-intuitif)
    - 🟡 **SMOTE** : effet marginal global, mais protège le rappel des classes rares
    - ⚪ **Normalisation** : neutre pour Random Forest

    Le protocole **holdout propre v5** (30 %, jamais vu lors du développement)
    corrige le biais de sélection : les 94,06 % de F1 constituent une estimation
    non biaisée des performances sur de nouvelles données.

    > Scurvy détecté à **100 %** malgré seulement 2,3 % du dataset.
    """)

    st.markdown("### Équipe")
    for name in ["Ibnmtar Hazem", "Moutchachou Lydia", "Varol Serdar", "Bekakria Ahmed"]:
        st.markdown(f'<span class="team-chip">👤 {name}</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-serdarvarl%2Fprojet__vitamin__IA-1a237e?logo=github)]"
        "(https://github.com/serdarvarl/projet_vitamin_IA)"
    )

st.markdown("---")
st.caption("Projet universitaire · L3 MIASHS · 2025–2026 · Données : Kaggle (Vitamin Deficiency Dataset) & CIQUAL/ANSES")
