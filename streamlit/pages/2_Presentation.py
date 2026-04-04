import os
import streamlit as st
import plotly.graph_objects as go

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOCS = os.path.join(REPO, "docs")

st.set_page_config(
    page_title="VitaIA — Résultats de l'étude",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .section-label {
        font-size: 0.78rem; font-weight: 700; color: #1a237e;
        text-transform: uppercase; letter-spacing: 1.2px;
        border-bottom: 2px solid #3949ab; padding-bottom: 3px;
        margin: 1.6rem 0 1rem;
    }
    .metric-card {
        background: white; border: 1px solid #e3e8f0;
        border-radius: 10px; padding: 1.1rem 0.9rem;
        text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .metric-card .val { font-size: 1.8rem; font-weight: 700; color: #1a237e; }
    .metric-card .lbl { font-size: 0.78rem; color: #666; margin-top: 3px; }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Résultats de l'étude")
st.markdown("Performances du modèle Random Forest final et étude d'ablation.")

# ── Métriques clés ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Performances finales — holdout v5 (n = 1 200, jamais vu)</div>',
            unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
for col, (val, lbl) in zip([m1, m2, m3, m4, m5], [
    ("94,06 %", "F1 pondéré"),
    ("94,00 %", "Exactitude"),
    ("1,00",    "F1 Scurvy (parfait)"),
    ("0,88",    "F1 Night_Blindness (difficile)"),
    ("340",     "Arbres (RF)"),
]):
    col.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                 unsafe_allow_html=True)

# ── Comparaison modèles ───────────────────────────────────────────────────────
st.markdown('<div class="section-label">Comparaison des algorithmes</div>', unsafe_allow_html=True)

col_chart, col_text = st.columns([2, 1], gap="large")
with col_chart:
    models    = ["RF final (v5\nholdout)", "RF v3\n(test partagé)", "SVM v3\n(test partagé)", "k-NN v3\n(test partagé)"]
    f1_vals   = [0.9406, 0.9431, 0.8926, 0.8208]
    colors    = ["#1a237e", "#5c6bc0", "#9fa8da", "#c5cae9"]
    fig = go.Figure(go.Bar(
        x=models, y=f1_vals,
        text=[f"{v:.1%}" for v in f1_vals],
        textposition="outside",
        marker_color=colors,
    ))
    fig.update_layout(
        yaxis=dict(range=[0.7, 1.0], tickformat=".0%", title="F1-Score"),
        title="F1-Score par modèle (⚠️ v3 : test partagé, biais de sélection possible)",
        height=360,
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_text:
    st.markdown("""
    **Pourquoi v5 < v3 ?**

    Les versions v1–v4 utilisaient le **même jeu de test** à chaque itération.
    Chaque choix de modèle était guidé par ce test → biais de sélection.

    La **v5** crée un nouveau holdout (`random_state=999`, 30 %) jamais utilisé
    pendant le développement.

    La baisse de **−2,6 points F1** reflète la **correction du biais**, pas une
    dégradation du modèle. Les 94,06 % sont l'estimation réaliste.
    """)
    st.image(os.path.join(DOCS, "comparaison_modeles.png"),
             caption="Comparaison CV vs test", use_container_width=True)

# ── Matrice de confusion holdout ─────────────────────────────────────────────
st.markdown('<div class="section-label">Matrice de confusion — holdout final (v5)</div>',
            unsafe_allow_html=True)
cm_col, perf_col = st.columns([1, 1], gap="large")
with cm_col:
    st.image(os.path.join(DOCS, "holdout_confusion_matrix.png"),
             caption="Matrice de confusion — n=1 200", use_container_width=True)
with perf_col:
    st.markdown("**Rapport de classification :**")
    df_perf = {
        "Classe": ["Anemia", "Healthy", "Night_Blindness", "Rickets_Osteomalacia", "Scurvy"],
        "Précision": [0.96, 0.98, 0.85, 0.87, 1.00],
        "Rappel":    [0.92, 0.95, 0.92, 0.94, 1.00],
        "F1":        [0.94, 0.97, 0.88, 0.91, 1.00],
        "Support":   [373, 453, 37, 309, 28],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(df_perf), hide_index=True, use_container_width=True)
    st.markdown("""
    - **Scurvy** : F1 = 1,00 — détection parfaite malgré seulement 28 cas
    - **Night_Blindness** : F1 = 0,88 — classe la plus difficile (37 cas, symptômes partagés)
    - **Healthy** : F1 = 0,97 — la mieux discriminée
    """)

# ── Étude d'ablation ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Étude d\'ablation (v4) — contribution de chaque composante</div>',
            unsafe_allow_html=True)

abl_left, abl_right = st.columns([1, 1], gap="large")
with abl_left:
    st.image(os.path.join(DOCS, "ablation_recap_final.png"),
             caption="Synthèse des 4 ablations", use_container_width=True)
with abl_right:
    st.markdown("""
    | Ablation | Δ F1 test | Interprétation |
    |---|---|---|
    | Sans biomarqueurs | **−17,4 %** 🔴 | Composante critique — irremplaçable |
    | + Variables lifestyle | **+5,3 %** 🟢 | Découverte majeure — contre-intuitif |
    | Sans SMOTE | +0,8 % | Marginal globalement |
    | Sans normalisation | +0,9 % | Neutre pour Random Forest |

    > **Enseignement clé :** les variables socio-comportementales (âge, IMC, exposition
    > solaire, région géographique...) apportent une information contextuelle que les
    > seuls dosages biologiques ne capturent pas — elles ont fait passer le modèle de
    > 91,4 % à 96,6 % F1 sur le test interne.
    """)

# ── Détail des ablations ──────────────────────────────────────────────────────
st.markdown('<div class="section-label">Détail des ablations</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["1 — SMOTE", "2 — Lifestyle", "3 — Biomarqueurs", "4 — Normalisation"]
)
for tab, img, cap in zip(
    [tab1, tab2, tab3, tab4],
    ["ablation1_smote.png", "ablation2_lifestyle.png", "ablation3_bio.png", "ablation4_scaler.png"],
    ["Impact de SMOTE", "Gain des variables lifestyle (+5,3 % F1)",
     "Chute sans biomarqueurs (−17,4 % F1)", "Impact de la normalisation"],
):
    with tab:
        st.image(os.path.join(DOCS, img), caption=cap, use_container_width=True)

# ── Importance des variables ──────────────────────────────────────────────────
st.markdown('<div class="section-label">Importance des variables</div>', unsafe_allow_html=True)

fi_col, roc_col = st.columns(2, gap="large")
with fi_col:
    st.image(os.path.join(DOCS, "feature_importance.png"),
             caption="Importance des variables — Random Forest", use_container_width=True)
with roc_col:
    st.image(os.path.join(DOCS, "roc_curves_v3.png"),
             caption="Courbes ROC — AUC > 0,95 pour toutes les classes", use_container_width=True)

# ── SMOTE ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Effet de SMOTE sur le déséquilibre de classes</div>',
            unsafe_allow_html=True)
smote_col, lc_col = st.columns(2, gap="large")
with smote_col:
    st.image(os.path.join(DOCS, "smote_comparison.png"),
             caption="Distribution avant / après SMOTE", use_container_width=True)
with lc_col:
    st.image(os.path.join(DOCS, "learning_curves_v3.png"),
             caption="Courbes d'apprentissage — pas de sur-apprentissage", use_container_width=True)

st.markdown("---")
st.caption(
    "Projet universitaire L3 MIASHS · Ibnmtar Hazem, Moutchachou Lydia, "
    "Varol Serdar, Bekakria Ahmed · 2025–2026"
)
