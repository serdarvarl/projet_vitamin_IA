"""
VitaIA — Page Résultats de l'étude
=====================================
Tableau de bord complet des performances du modèle Random Forest v5 final.

Affiche en temps réel (recalculé sur le holdout 30 %) :
    - Métriques clés : F1 pondéré, exactitude, F1 par classe difficile
    - Matrice de confusion (heatmap Plotly, effectifs + % rappel)
    - Précision / Rappel / F1 par classe (barres groupées)
    - Courbes ROC One-vs-Rest + AUC par classe
    - Importance des variables Random Forest (Gini, 4 groupes colorés)
    - Étude d'ablation à 4 dimensions (F1 CV vs F1 test + Δ)
    - Distribution des classes et effet de SMOTE sur l'entraînement
    - Comparaison des algorithmes RF / SVM / k-NN

Sources / Références :
    - Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
      DOI: 10.1023/A:1010933404324
    - Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
      JAIR, 16, 321–357. DOI: 10.1613/jair.953
    - Cortes, C. & Vapnik, V. (1995). Support-vector networks.
      Machine Learning, 20(3), 273–297.
    - Cover, T. & Hart, P. (1967). Nearest neighbor pattern classification.
      IEEE Trans. Information Theory, 13(1), 21–27.
    - Scikit-learn — sklearn.metrics (roc_curve, auc, confusion_matrix) :
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    - Plotly Python graphing library : https://plotly.com/python/
    - Streamlit cache_data : https://docs.streamlit.io/library/api-reference/performance/st.cache_data
    - Dataset Kaggle : https://www.kaggle.com/datasets/nudratabbas/vitamin-deficiency-disease-prediction-dataset

Auteurs : Ibnmtar Hazem, Moutchachou Lydia, Varol Serdar, Bekakria Ahmed
Formation : L3 MIASHS — Université Paul-Valéry Montpellier 3 — 2025/2026
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

REPO      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(REPO, "notebooks", "models_final")
LOGO_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "logos", "logo2_pilule_neurone.svg")
DATA_ODS  = os.path.join(REPO, "data_csv", "raw",
                         "vitamin_deficiency_disease_dataset_20260123.ods")

st.set_page_config(page_title="VitaIA — Résultats", page_icon="💊", layout="wide")

st.markdown("""
<style>
    /* Bandeau page */
    .page-header {
        background: #1a3a5c;
        color: #fff;
        padding: 1.6rem 2rem;
        margin-bottom: 2rem;
        border-bottom: 4px solid #2e6da4;
    }
    .page-header h2 { margin: 0 0 0.25rem; font-size: 1.7rem; font-weight: 700; letter-spacing: 0.3px; }
    .page-header p  { margin: 0; color: #c8d8e8; font-size: 0.92rem; }

    /* Séparateurs de section */
    .section-label {
        font-size: 0.72rem; font-weight: 700; color: #2e6da4;
        text-transform: uppercase; letter-spacing: 1.4px;
        border-bottom: 2px solid #2e6da4; padding-bottom: 5px;
        margin: 2rem 0 1.2rem;
    }

    /* Cartes métriques */
    .metric-card {
        background: #fff;
        border: 1px solid #d8e2ec;
        border-top: 4px solid #2e6da4;
        box-shadow: 0 2px 10px rgba(26,58,92,0.08);
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .metric-card .val {
        font-size: 1.9rem; font-weight: 700; color: #1a3a5c; line-height: 1.1;
    }
    .metric-card .lbl {
        font-size: 0.73rem; color: #5a6878; margin-top: 5px;
        text-transform: uppercase; letter-spacing: 0.5px;
    }

    [data-testid="stSidebarHeader"] img {
        height: 90px !important; width: auto !important;
    }
    [data-testid="stSidebarHeader"] { padding: 1rem 1rem 0.5rem !important; }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Couleurs constantes ───────────────────────────────────────────────────────
COLORS = {
    "Anemia":               "#c0392b",
    "Healthy":              "#27ae60",
    "Night_Blindness":      "#d68910",
    "Rickets_Osteomalacia": "#e67e22",
    "Scurvy":               "#7d3c98",
}
CLASSES = ["Anemia", "Healthy", "Night_Blindness", "Rickets_Osteomalacia", "Scurvy"]
PALETTE = [COLORS[c] for c in CLASSES]

# ── Chargement modèle + calcul métriques ──────────────────────────────────────
@st.cache_data(show_spinner="Calcul des métriques du modèle...")
def compute_all_metrics():
    """
    Charge le modèle final v5 et calcule toutes les métriques de performance
    sur le holdout 30 % (random_state=999, jamais utilisé pendant le développement).

    Protocole exact reproduit depuis le notebook modeles_IA_v5_holdout_final.ipynb :
        1. Chargement du dataset ODS (encodage catégoriel par .cat.codes)
        2. Séparation holdout : test_size=0.3, random_state=999, stratify=y
        3. Application du StandardScaler (ajusté sur X_train uniquement)
        4. Prédiction sur X_test (données jamais vues lors de l'entraînement)

    Métriques calculées :
        - Matrice de confusion (sklearn.metrics.confusion_matrix)
        - Rapport de classification par classe (precision, recall, f1)
        - Courbes ROC One-vs-Rest + AUC par classe (sklearn.metrics.roc_curve)
        - Importance des variables Random Forest (feature_importances_ via critère Gini)
        - Distribution des classes dans le dataset complet

    Returns:
        dict avec clés : 'cm', 'classes', 'report', 'roc', 'fi', 'feats', 'class_dist'

    Note:
        Le modèle v5 est un RandomForestClassifier direct (non un Pipeline imbalanced-learn),
        contrairement aux versions v1-v3. L'accès aux feature_importances_ utilise un
        try/except pour gérer les deux cas (Pipeline.steps[-1][1] vs modèle direct).

    Ref:
        - sklearn.metrics : https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        - sklearn.preprocessing.LabelBinarizer : pour ROC multiclasse One-vs-Rest
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import LabelBinarizer

    # Charger modèle v5
    model  = pickle.load(open(os.path.join(MODEL_DIR, "random_forest_final.pkl"), "rb"))
    le     = pickle.load(open(os.path.join(MODEL_DIR, "label_encoder_final.pkl"), "rb"))
    feats  = pickle.load(open(os.path.join(MODEL_DIR, "feature_cols_final.pkl"), "rb"))
    scaler_path = os.path.join(MODEL_DIR, "scaler_final.pkl")
    scaler = pickle.load(open(scaler_path, "rb")) if os.path.exists(scaler_path) else None

    # Charger dataset
    df = pd.read_excel(DATA_ODS, engine="odf")
    for col in df.select_dtypes(include=["object", "str"]).columns:
        if col != "disease_diagnosis":
            df[col] = df[col].astype("category").cat.codes

    X = df[feats]
    y = le.transform(df["disease_diagnosis"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=999, stratify=y
    )

    if scaler is not None:
        X_test = scaler.transform(X_test)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred,
                                    target_names=le.classes_, output_dict=True)

    # ROC par classe
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    roc_data = {}
    for i, cls in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_data[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                         "auc": float(auc(fpr, tpr))}

    # Feature importance
    try:
        rf = model.steps[-1][1]  # Pipeline
    except AttributeError:
        rf = model               # Modèle direct
    fi = rf.feature_importances_

    # Distribution classes avant SMOTE
    class_dist = df["disease_diagnosis"].value_counts().to_dict()

    return {
        "cm": cm.tolist(),
        "classes": le.classes_.tolist(),
        "report": report,
        "roc": roc_data,
        "fi": fi.tolist(),
        "feats": feats,
        "class_dist": class_dist,
    }

# ── Données d'ablation (chiffres exacts du notebook v4) ──────────────────────
ABLATION = {
    "config":  ["Référence\n(Bio+Sympt+SMOTE+Scaler)",
                "Sans SMOTE",
                "+ Lifestyle",
                "Sans Bio",
                "Sans Scaler"],
    "cv_f1":   [0.9232, 0.9230, 0.9651, 0.7481, 0.9244],
    "test_f1": [0.9139, 0.9222, 0.9665, 0.7403, 0.9224],
    "delta":   [0,      0.0083, 0.0526, -0.1736, 0.0085],
    "colors":  ["#2e6da4", "#5dade2", "#27ae60", "#c0392b", "#5dade2"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if os.path.exists(LOGO_PATH):
    st.logo(LOGO_PATH, size="large")

st.markdown("""
<div class="page-header">
    <h2>Résultats de l'étude</h2>
    <p>Performances du modèle Random Forest v5 — métriques calculées en temps réel à partir du holdout 30 %</p>
</div>
""", unsafe_allow_html=True)

# Charger les métriques
try:
    data = compute_all_metrics()
    report   = data["report"]
    classes  = data["classes"]
    cm       = np.array(data["cm"])
    roc_data = data["roc"]
    fi       = np.array(data["fi"])
    feats    = data["feats"]
    dist     = data["class_dist"]
except Exception as e:
    st.error(f"Erreur chargement modèle : {e}\nLancez d'abord le notebook pour générer les pkl.")
    st.stop()

# ── Métriques clés ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Performances globales — Random Forest v5 final (holdout 30%)</div>',
            unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
for col, (val, lbl) in zip([m1, m2, m3, m4, m5], [
    (f"{report['weighted avg']['f1-score']:.1%}", "F1 pondéré"),
    (f"{report['accuracy']:.1%}",                 "Exactitude"),
    (f"{report['Scurvy']['f1-score']:.2f}",        "F1 Scorbut"),
    (f"{report['Night_Blindness']['f1-score']:.2f}", "F1 Cécité nocturne"),
    ("340", "Arbres (Random Forest)"),
]):
    col.markdown(
        f'<div class="metric-card"><div class="val">{val}</div>'
        f'<div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True
    )

# ── Matrice de confusion + F1 par classe ─────────────────────────────────────
st.markdown('<div class="section-label">Matrice de confusion et performances par classe</div>',
            unsafe_allow_html=True)
col_cm, col_f1 = st.columns(2, gap="large")

with col_cm:
    # Normaliser pour afficher les pourcentages
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    labels_short = ["Anemia", "Healthy", "Night\nBlind.", "Rickets", "Scurvy"]
    text = [[f"{cm[i][j]}<br>({cm_pct[i][j]:.0f}%)" for j in range(5)] for i in range(5)]

    fig_cm = go.Figure(go.Heatmap(
        z=cm_pct,
        x=labels_short, y=labels_short,
        text=text, texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
    ))
    fig_cm.update_layout(
        title="Matrice de confusion (effectifs + % rappel)",
        xaxis_title="Prédit", yaxis_title="Réel",
        height=380,
        margin=dict(t=50, b=40, l=40, r=20),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_f1:
    metrics_df = pd.DataFrame({
        "Classe":    classes,
        "Précision": [report[c]["precision"] for c in classes],
        "Rappel":    [report[c]["recall"]    for c in classes],
        "F1":        [report[c]["f1-score"]  for c in classes],
    })
    fig_f1 = go.Figure()
    for metric, color in [("Précision","#2e6da4"), ("Rappel","#27ae60"), ("F1","#e67e22")]:
        fig_f1.add_trace(go.Bar(
            name=metric,
            x=metrics_df["Classe"],
            y=metrics_df[metric],
            text=[f"{v:.2f}" for v in metrics_df[metric]],
            textposition="outside",
            marker_color=color,
        ))
    fig_f1.update_layout(
        barmode="group",
        title="Précision / Rappel / F1 par classe",
        yaxis=dict(range=[0.6, 1.08], tickformat=".0%"),
        legend=dict(orientation="h", y=1.1),
        height=380,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_f1, use_container_width=True)

# ── Courbes ROC ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Courbes ROC (One-vs-Rest)</div>',
            unsafe_allow_html=True)
col_roc, col_auc = st.columns(2, gap="large")

with col_roc:
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                      line=dict(dash="dash", color="#aaa", width=1))
    for cls in classes:
        rd = roc_data[cls]
        fig_roc.add_trace(go.Scatter(
            x=rd["fpr"], y=rd["tpr"],
            name=f"{cls} (AUC={rd['auc']:.3f})",
            mode="lines",
            line=dict(color=COLORS[cls], width=2),
        ))
    fig_roc.update_layout(
        title="Courbes ROC par classe",
        xaxis_title="Taux faux positifs",
        yaxis_title="Taux vrais positifs",
        legend=dict(orientation="v", x=0.55, y=0.08, font=dict(size=10)),
        height=380,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with col_auc:
    aucs = [roc_data[c]["auc"] for c in classes]
    fig_auc = go.Figure(go.Bar(
        x=classes, y=aucs,
        text=[f"{v:.4f}" for v in aucs],
        textposition="outside",
        marker_color=PALETTE,
    ))
    fig_auc.update_layout(
        title="AUC par classe",
        yaxis=dict(range=[0.95, 1.005], tickformat=".3f"),
        height=380,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_auc, use_container_width=True)

# ── Importance des variables ──────────────────────────────────────────────────
st.markdown('<div class="section-label">Importance des variables (Random Forest)</div>',
            unsafe_allow_html=True)

fi_df = pd.DataFrame({"feature": feats, "importance": fi})
# Agréger les doublons éventuels
fi_df = fi_df.groupby("feature", as_index=False)["importance"].sum()
fi_df = fi_df.sort_values("importance", ascending=True)

# Groupes de features
LIFESTYLE = {"age","gender","bmi","smoking_status","alcohol_consumption",
             "exercise_level","diet_type","sun_exposure","latitude_region","income_level"}

def get_group(f):
    """
    Attribue un groupe sémantique à une feature pour la colorisation du graphique.

    Groupes définis selon la taxonomie du dataset (Kaggle, 4 groupes fonctionnels) :
        - Biomarqueur sérique : mesures sanguines directes (hémoglobine, vitamines sériques)
        - Symptôme clinique   : variables binaires has_* (symptômes déclarés)
        - Variable lifestyle  : facteurs socio-comportementaux (âge, IMC, exposition solaire...)
        - Apport nutritionnel : apports en % des AJR (vitamines, calcium, fer)

    Args:
        f (str): Nom de la feature (colonne du dataset).

    Returns:
        str: Nom du groupe ('Biomarqueur sérique', 'Symptôme clinique',
             'Variable lifestyle', ou 'Apport nutritionnel (%AJR)').
    """
    if f.startswith("serum") or f == "hemoglobin_g_dl": return "Biomarqueur sérique"
    if f.startswith("has_"):                            return "Symptôme clinique"
    if f in LIFESTYLE:                                  return "Variable lifestyle"
    return "Apport nutritionnel (%AJR)"

fi_df["groupe"] = fi_df["feature"].apply(get_group)
group_colors = {"Biomarqueur sérique":       "#c0392b",
                "Symptôme clinique":         "#2e6da4",
                "Apport nutritionnel (%AJR)":"#27ae60",
                "Variable lifestyle":        "#e67e22"}

fig_fi = go.Figure()
for groupe, color in group_colors.items():
    sub = fi_df[fi_df["groupe"] == groupe]
    fig_fi.add_trace(go.Bar(
        x=sub["importance"], y=sub["feature"],
        orientation="h", name=groupe,
        marker_color=color,
        text=[f"{v:.3f}" for v in sub["importance"]],
        textposition="outside",
    ))
fig_fi.update_layout(
    barmode="stack",
    title="Importance des variables par groupe",
    xaxis_title="Importance (Gini)",
    height=600,
    legend=dict(orientation="h", y=1.04),
    margin=dict(t=60, b=40, l=240),
    yaxis=dict(
        categoryorder="array",
        categoryarray=fi_df["feature"].tolist(),  # ordre global par importance
    ),
)
st.plotly_chart(fig_fi, use_container_width=True)

# ── Étude d'ablation ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Étude d\'ablation — contribution de chaque composante (v4)</div>',
            unsafe_allow_html=True)

col_abl, col_delta = st.columns([3, 2], gap="large")

with col_abl:
    fig_abl = go.Figure()
    fig_abl.add_trace(go.Bar(
        name="F1 validation croisée",
        x=ABLATION["config"], y=ABLATION["cv_f1"],
        marker_color="#2e6da4",
        text=[f"{v:.4f}" for v in ABLATION["cv_f1"]],
        textposition="outside",
    ))
    fig_abl.add_trace(go.Bar(
        name="F1 test",
        x=ABLATION["config"], y=ABLATION["test_f1"],
        marker_color="#27ae60",
        text=[f"{v:.4f}" for v in ABLATION["test_f1"]],
        textposition="outside",
    ))
    fig_abl.update_layout(
        barmode="group",
        title="F1 CV vs F1 test par configuration",
        yaxis=dict(range=[0.65, 1.02], tickformat=".2f"),
        legend=dict(orientation="h", y=1.1),
        height=380,
        margin=dict(t=60, b=60),
    )
    st.plotly_chart(fig_abl, use_container_width=True)

with col_delta:
    delta_colors = ["#aabbcc" if d == 0 else "#27ae60" if d > 0 else "#c0392b"
                    for d in ABLATION["delta"]]
    fig_delta = go.Figure(go.Bar(
        x=ABLATION["delta"][1:],
        y=ABLATION["config"][1:],
        orientation="h",
        marker_color=delta_colors[1:],
        text=[f"{d:+.4f}" for d in ABLATION["delta"][1:]],
        textposition="outside",
    ))
    fig_delta.add_vline(x=0, line_dash="dash", line_color="#aaa")
    fig_delta.update_layout(
        title="Variation F1 vs référence (Δ)",
        xaxis=dict(range=[-0.22, 0.08]),
        height=380,
        margin=dict(t=50, b=40, l=160),
    )
    st.plotly_chart(fig_delta, use_container_width=True)

st.markdown("""
> **Lecture :** Les biomarqueurs sont la composante critique (−17,4 % sans eux).
> Les variables lifestyle apportent le plus grand gain (+5,3 %).
> SMOTE et la normalisation ont un effet marginal sur Random Forest.
""")

# ── Distribution des classes + SMOTE ─────────────────────────────────────────
st.markdown('<div class="section-label">Distribution des classes et effet de SMOTE</div>',
            unsafe_allow_html=True)

col_dist, col_smote = st.columns(2, gap="large")

with col_dist:
    cls_names = list(dist.keys())
    cls_vals  = list(dist.values())
    fig_dist = go.Figure(go.Bar(
        x=cls_names, y=cls_vals,
        marker_color=PALETTE,
        text=cls_vals, textposition="outside",
    ))
    fig_dist.update_layout(
        title="Distribution des classes (dataset complet, n=4000)",
        yaxis_title="Effectif",
        height=340,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_smote:
    # Entraînement 70% → 2800 obs, après SMOTE équilibré à ~1509 par classe
    train_size   = int(4000 * 0.8)  # 3200 pour v3 (20% test)
    avant_smote  = {c: round(v * train_size / 4000) for c, v in dist.items()}
    max_class    = max(avant_smote.values())
    apres_smote  = {c: max_class for c in cls_names}

    fig_smote = go.Figure()
    fig_smote.add_trace(go.Bar(
        name="Avant SMOTE",
        x=list(avant_smote.keys()), y=list(avant_smote.values()),
        marker_color="#2e6da4", text=list(avant_smote.values()), textposition="outside",
    ))
    fig_smote.add_trace(go.Bar(
        name="Après SMOTE",
        x=list(apres_smote.keys()), y=list(apres_smote.values()),
        marker_color="#27ae60", text=list(apres_smote.values()), textposition="outside",
    ))
    fig_smote.update_layout(
        barmode="group",
        title="Effet de SMOTE sur le jeu d'entraînement",
        yaxis_title="Effectif",
        height=340,
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig_smote, use_container_width=True)

# ── Comparaison des modèles ───────────────────────────────────────────────────
st.markdown('<div class="section-label">Comparaison des algorithmes</div>',
            unsafe_allow_html=True)

models_name = ["RF v3\n(test partagé)", "SVM v3\n(test partagé)",
               "k-NN v3\n(test partagé)", "RF final v5\n(holdout propre)"]
f1_vals     = [0.9431, 0.8926, 0.8208, 0.9406]
acc_vals    = [0.9400, 0.8875, 0.8163, 0.9400]
col_cmp     = ["#2e6da4", "#95a5a6", "#95a5a6", "#27ae60"]

fig_cmp = go.Figure()
fig_cmp.add_trace(go.Bar(
    name="F1 pondéré",
    x=models_name, y=f1_vals,
    marker_color=col_cmp,
    text=[f"{v:.1%}" for v in f1_vals],
    textposition="outside",
))
fig_cmp.add_trace(go.Bar(
    name="Exactitude",
    x=models_name, y=acc_vals,
    marker_color=["rgba(46,109,164,0.3)", "rgba(149,165,166,0.3)",
                  "rgba(149,165,166,0.3)", "rgba(39,174,96,0.3)"],
    text=[f"{v:.1%}" for v in acc_vals],
    textposition="outside",
))
fig_cmp.update_layout(
    barmode="group",
    title="F1 pondéré et Exactitude par modèle",
    yaxis=dict(range=[0.70, 1.04], tickformat=".0%"),
    legend=dict(orientation="h", y=1.1),
    height=380,
    margin=dict(t=60, b=60),
)
st.plotly_chart(fig_cmp, use_container_width=True)

st.caption(
    "Projet universitaire L3 MIASHS · Ibnmtar Hazem, Moutchachou Lydia, "
    "Varol Serdar, Bekakria Ahmed · 2025–2026"
)
