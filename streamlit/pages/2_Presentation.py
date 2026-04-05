import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

REPO      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(REPO, "notebooks", "models_v3")
DATA_ODS  = os.path.join(REPO, "data_csv", "raw",
                         "vitamin_deficiency_disease_dataset_20260123.ods")

st.set_page_config(page_title="VitaIA — Résultats", page_icon="💊", layout="wide")

st.markdown("""
<style>
    .section-label {
        font-size: 0.75rem; font-weight: 700; color: #90caf9;
        text-transform: uppercase; letter-spacing: 1.4px;
        border-bottom: 2px solid #42a5f5; padding-bottom: 4px;
        margin: 1.8rem 0 1rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.05); border: 1px solid #42a5f5;
        border-radius: 10px; padding: 1.1rem 0.9rem; text-align: center;
    }
    .metric-card .val { font-size: 1.8rem; font-weight: 700; color: #90caf9; }
    .metric-card .lbl { font-size: 0.78rem; color: #b0bec5; margin-top: 3px; }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Couleurs constantes ───────────────────────────────────────────────────────
COLORS = {
    "Anemia":               "#ef9a9a",
    "Healthy":              "#a5d6a7",
    "Night_Blindness":      "#fff176",
    "Rickets_Osteomalacia": "#ffcc80",
    "Scurvy":               "#ce93d8",
}
CLASSES = ["Anemia", "Healthy", "Night_Blindness", "Rickets_Osteomalacia", "Scurvy"]
PALETTE = [COLORS[c] for c in CLASSES]

# ── Chargement modèle + calcul métriques ──────────────────────────────────────
@st.cache_data(show_spinner="Calcul des métriques du modèle...")
def compute_all_metrics():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import LabelBinarizer

    # Charger modèle
    model = pickle.load(open(os.path.join(MODEL_DIR, "best_model_v3.pkl"), "rb"))
    le    = pickle.load(open(os.path.join(MODEL_DIR, "label_encoder_v3.pkl"), "rb"))
    feats = pickle.load(open(os.path.join(MODEL_DIR, "feature_cols_v3.pkl"), "rb"))

    # Charger dataset
    df = pd.read_excel(DATA_ODS, engine="odf")
    for col in df.select_dtypes(include=["object", "str"]).columns:
        if col != "disease_diagnosis":
            df[col] = df[col].astype("category").cat.codes

    X = df[feats]
    y = le.transform(df["disease_diagnosis"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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

    # Feature importance (depuis le pipeline)
    rf = model.steps[-1][1]
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
    "colors":  ["#90caf9", "#80deea", "#a5d6a7", "#ef9a9a", "#80deea"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Résultats de l'étude")
st.markdown("Performances du modèle Random Forest — métriques calculées en temps réel.")

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
st.markdown('<div class="section-label">Performances globales — Random Forest v3 (test set 20%)</div>',
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
    for metric, color in [("Précision","#90caf9"), ("Rappel","#a5d6a7"), ("F1","#ffcc80")]:
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
                      line=dict(dash="dash", color="#555", width=1))
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
fi_df = fi_df.sort_values("importance", ascending=True)

# Groupes de features
def get_group(f):
    if f.startswith("serum") or f == "hemoglobin_g_dl": return "Biomarqueur sérique"
    if f.startswith("has_"):                            return "Symptôme clinique"
    return "Apport nutritionnel (%AJR)"

fi_df["groupe"] = fi_df["feature"].apply(get_group)
group_colors = {"Biomarqueur sérique": "#ef9a9a",
                "Symptôme clinique":   "#90caf9",
                "Apport nutritionnel (%AJR)": "#a5d6a7"}

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
    height=520,
    legend=dict(orientation="h", y=1.05),
    margin=dict(t=60, b=40, l=200),
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
        marker_color="#90caf9",
        text=[f"{v:.4f}" for v in ABLATION["cv_f1"]],
        textposition="outside",
    ))
    fig_abl.add_trace(go.Bar(
        name="F1 test",
        x=ABLATION["config"], y=ABLATION["test_f1"],
        marker_color="#a5d6a7",
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
    delta_colors = ["#ffcc80" if d == 0 else "#a5d6a7" if d > 0 else "#ef9a9a"
                    for d in ABLATION["delta"]]
    fig_delta = go.Figure(go.Bar(
        x=ABLATION["delta"][1:],
        y=ABLATION["config"][1:],
        orientation="h",
        marker_color=delta_colors[1:],
        text=[f"{d:+.4f}" for d in ABLATION["delta"][1:]],
        textposition="outside",
    ))
    fig_delta.add_vline(x=0, line_dash="dash", line_color="#555")
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
        marker_color="#90caf9", text=list(avant_smote.values()), textposition="outside",
    ))
    fig_smote.add_trace(go.Bar(
        name="Après SMOTE",
        x=list(apres_smote.keys()), y=list(apres_smote.values()),
        marker_color="#a5d6a7", text=list(apres_smote.values()), textposition="outside",
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
col_cmp     = ["#90caf9", "#b0bec5", "#b0bec5", "#a5d6a7"]

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
    marker_color=["rgba(144,202,249,0.4)", "rgba(176,190,197,0.4)",
                  "rgba(176,190,197,0.4)", "rgba(165,214,167,0.4)"],
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
