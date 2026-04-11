"""
VitaIA — Page d'accueil (Accueil)
==================================
Point d'entrée de l'application multi-pages Streamlit VitaIA.

Affiche :
- Bannière principale avec logo SVG (base64)
- Cartes métriques clés du modèle Random Forest final (v5)
- Présentation du projet et guide d'utilisation
- Synthèse des résultats de l'étude d'ablation
- Membres de l'équipe avec liens GitHub

Sources / Références :
    - Streamlit documentation : https://docs.streamlit.io
    - Dataset Kaggle (Vitamin Deficiency) :
        https://www.kaggle.com/datasets/nudratabbas/vitamin-deficiency-disease-prediction-dataset
    - Base CIQUAL (ANSES) : https://ciqual.anses.fr
    - Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
    - Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
        Journal of Artificial Intelligence Research, 16, 321–357.

Auteurs : Ibnmtar Hazem, Moutchachou Lydia, Varol Serdar, Bekakria Ahmed
Formation : L3 MIASHS — Université Paul-Valéry Montpellier 3 — 2025/2026
"""

import streamlit as st
import base64, os

LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logos", "logo2_pilule_neurone.svg")

def _logo_b64():
    """Encode le logo SVG en base64 pour l'intégration HTML inline."""
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()

st.set_page_config(
    page_title="VitaIA — Prédiction des carences en vitamines",
    page_icon="💊",
    layout="wide",
)

st.markdown("""
<style>
    /* ── Typographie globale ── */
    html, body, [class*="css"] {
        font-family: "Segoe UI", Arial, sans-serif;
    }

    /* ── Bannière principale ── */
    .vita-header {
        background: linear-gradient(100deg, #0f2744 0%, #1a3a5c 60%, #1e4976 100%);
        color: #ffffff;
        padding: 2.8rem 2.5rem;
        margin-bottom: 2.5rem;
        border-bottom: 4px solid #2e6da4;
        box-shadow: 0 4px 18px rgba(15,39,68,0.18);
    }
    .vita-header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .vita-header .tagline {
        font-size: 1rem;
        margin: 0;
        color: #a8c4e0;
        letter-spacing: 0.3px;
    }
    .vita-header .badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        color: #e0eeff;
        font-size: 0.75rem;
        padding: 3px 12px;
        margin-top: 0.8rem;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }

    /* ── Cartes métriques ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #d8e2ec;
        border-top: 4px solid #2e6da4;
        box-shadow: 0 2px 12px rgba(26,58,92,0.08);
        padding: 1.3rem 1rem;
        text-align: center;
    }
    .metric-card .val {
        font-size: 2rem;
        font-weight: 700;
        color: #1a3a5c;
        line-height: 1.1;
    }
    .metric-card .lbl {
        font-size: 0.75rem;
        color: #5a6878;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Étapes ── */
    .step-row {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 0.9rem 1rem;
        border-bottom: 1px solid #e8ecf0;
        transition: background 0.15s;
    }
    .step-row:hover { background: #f4f8fc; }
    .step-row:last-child { border-bottom: none; }
    .step-num {
        background: #2e6da4;
        color: #fff;
        font-size: 0.80rem;
        font-weight: 800;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .step-text {
        font-size: 0.93rem;
        color: #2d3748;
        line-height: 1.55;
    }
    .step-text strong { color: #1a3a5c; font-weight: 700; }

    /* ── Section titres ── */
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #1a3a5c;
        border-bottom: 2px solid #2e6da4;
        padding-bottom: 6px;
        margin-bottom: 1rem;
    }

    /* ── Membres équipe ── */
    .team-member {
        display: inline-block;
        background: #ffffff;
        border: 1px solid #c8d8e8;
        color: #1a3a5c !important;
        padding: 6px 18px;
        margin: 4px;
        font-size: 0.86rem;
        font-weight: 600;
        text-decoration: none !important;
        box-shadow: 0 1px 4px rgba(26,58,92,0.08);
        transition: background 0.15s, border-color 0.15s, box-shadow 0.15s;
    }
    a.team-member:hover {
        background: #e8eef4;
        border-color: #2e6da4;
        box-shadow: 0 2px 8px rgba(26,58,92,0.14);
    }

    /* ── Cards de contenu ── */
    .content-card {
        background: #fff;
        border: 1px solid #d8e2ec;
        box-shadow: 0 2px 12px rgba(26,58,92,0.07);
        padding: 1.6rem 1.8rem;
        height: 100%;
    }

    /* ── Tableau Streamlit ── */
    table { border-collapse: collapse; width: 100%; font-size: 0.91rem; }
    th { background: #1a3a5c; color: #fff; padding: 8px 12px; text-align: left; font-weight: 600; }
    td { padding: 7px 12px; border-bottom: 1px solid #e8ecf0; color: #212529; }
    tr:nth-child(even) td { background: #f8fafc; }

    /* ── Logo sidebar agrandi ── */
    [data-testid="stSidebarHeader"] img {
        height: 90px !important;
        width: auto !important;
    }
    [data-testid="stSidebarHeader"] {
        padding: 1rem 1rem 0.5rem !important;
    }

    /* ── Masquer chrome Streamlit ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Bannière ──────────────────────────────────────────────────────────────────
_b64 = _logo_b64()
st.markdown(f"""
<div class="vita-header" style="display:flex;align-items:center;gap:2rem;">
    <img src="data:image/svg+xml;base64,{_b64}" height="90" style="flex-shrink:0;">
    <div>
        <h1>VitaIA</h1>
        <p class="tagline">Système de prédiction des carences en vitamines par apprentissage automatique</p>
        <span class="badge">Projet universitaire &nbsp;·&nbsp; L3 MIASHS &nbsp;·&nbsp; 2025–2026</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Logo sidebar (au-dessus de la navigation)
st.logo(LOGO_PATH, size="large")

# ── Métriques ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("94,06 %", "F1-Score pondéré"),
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

# ── Corps ─────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">À propos du projet</div>', unsafe_allow_html=True)
    st.markdown("""
    **VitaIA** est un système de classification automatique des carences en vitamines
    développé dans le cadre d'un projet universitaire L3 MIASHS.

    Le modèle **Random Forest** (340 arbres, pipeline normalisé + SMOTE) prédit parmi
    5 diagnostics à partir du profil clinique, biologique et comportemental d'un patient.
    """)

    st.markdown("""
    | Diagnostic | Description |
    |---|---|
    | **Healthy** | Aucune carence détectée |
    | **Anemia** | Carence en fer / vitamine B12 / folate |
    | **Rickets\_Osteomalacia** | Carence en vitamine D / calcium |
    | **Night\_Blindness** | Carence en vitamine A (rétinol) |
    | **Scurvy** | Carence sévère en vitamine C |
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Utilisation</div>', unsafe_allow_html=True)

    steps = [
        ("1", "<strong>Diagnostic</strong> — Renseignez les données cliniques du patient dans le formulaire"),
        ("2", "<strong>Prédiction</strong> — Le modèle retourne le diagnostic probable et le score de confiance"),
        ("3", "<strong>Recommandations</strong> — Des aliments correcteurs issus de la base CIQUAL (ANSES) sont proposés"),
        ("4", "<strong>Résultats</strong> — Consultez les performances détaillées du modèle et l'analyse d'ablation"),
    ]
    rows = "".join(
        f'<div class="step-row"><div class="step-num">{n}</div>'
        f'<div class="step-text">{txt}</div></div>'
        for n, txt in steps
    )
    st.markdown(f'<div style="border:1px solid #d8e2ec;box-shadow:0 1px 6px rgba(26,58,92,0.06)">{rows}</div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Résultats clés</div>', unsafe_allow_html=True)
    st.markdown("""
    L'analyse d'ablation à 4 dimensions a établi la hiérarchie des contributions :

    - **Biomarqueurs sériques** : composante critique (−17,4 % de F1 sans eux)
    - **Variables lifestyle** : gain significatif (+5,3 % de F1)
    - **SMOTE** : protection du rappel sur classes rares (Scorbut, Cécité nocturne)
    - **Normalisation** : effet neutre pour Random Forest

    Le protocole holdout v5 (30 % — données jamais vues lors du développement)
    garantit une estimation non biaisée des performances réelles.

    Le diagnostic Scorbut atteint **100 % de F1** malgré 2,3 % de représentation
    dans le dataset.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Équipe</div>', unsafe_allow_html=True)

    team = [
        ("Ibnmtar Hazem",     "https://github.com/IbnmtarHazem"),
        ("Moutchachou Lydia", "https://github.com/lydiamtch"),
        ("Varol Serdar",      "https://github.com/serdarvarl"),
        ("Bekakria Ahmed",    "https://github.com/ahmed-abc73"),
    ]
    chips = "".join(
        f'<a href="{url}" target="_blank" class="team-member">{name}</a>'
        for name, url in team
    )
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-serdarvarl%2Fprojet__vitamin__IA-1a3a5c?logo=github&style=flat-square)]"
        "(https://github.com/serdarvarl/projet_vitamin_IA)"
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Projet universitaire · L3 MIASHS · 2025–2026 · Données : Kaggle (Vitamin Deficiency Dataset) & CIQUAL/ANSES")
