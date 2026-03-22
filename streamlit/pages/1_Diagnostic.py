import sys
import os

# Chemin vers la racine du repo
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import pickle
import pandas as pd
import streamlit as st
import plotly.express as px

# ── Chargement du modèle ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(REPO_ROOT, "notebooks/models_v3")

@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "best_model_v3.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder_v3.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_cols_v3.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    return model, le, feature_cols

model, le, feature_cols = load_model()

# ── Correspondances ───────────────────────────────────────────────────────────
DIAGNOSIS_FR = {
    "Anemia":               "Anémie (carence en Fer)",
    "Scurvy":               "Scorbut (carence en Vitamine C)",
    "Rickets_Osteomalacia": "Rachitisme / Ostéomalacie (carence en Vitamine D)",
    "Night_Blindness":      "Cécité nocturne (carence en Vitamine A)",
    "Healthy":              "Aucune carence détectée",
}

DIAGNOSIS_VITAMIN = {
    "Anemia":               "Fer",
    "Scurvy":               "Vitamine C",
    "Rickets_Osteomalacia": "Vitamine D",
    "Night_Blindness":      "Vitamine A",
    "Healthy":              None,
}

# ── Chargement CIQUAL ─────────────────────────────────────────────────────────
CIQUAL_PATH = os.path.join(REPO_ROOT, "bdd_final", "ciqual.csv")

VITAMIN_COL_CIQUAL = {
    "Fer":        "Fer (mg/100g)",
    "Vitamine C": "Vitamine C (mg/100g)",
    "Vitamine D": "Vitamine D (µg/100g)",
    "Vitamine A": "Vitamine A (µg/100g)",
}

@st.cache_data
def load_ciqual():
    if not os.path.exists(CIQUAL_PATH):
        return None
    df = pd.read_csv(CIQUAL_PATH, sep=";", encoding="latin-1")
    return df

def get_recommendations(vitamin, top_n=10):
    df = load_ciqual()
    if df is None:
        st.warning("Fichier CIQUAL introuvable. Placez ciqual.csv dans bdd_final/")
        return pd.DataFrame()

    col = VITAMIN_COL_CIQUAL.get(vitamin)
    if col not in df.columns:
        st.warning(f"Colonne '{col}' introuvable dans le fichier CIQUAL.")
        return pd.DataFrame()

    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", "."), errors="coerce"
    )
    df = df.dropna(subset=[col]).sort_values(col, ascending=False)
    return df[["alim_nom_fr", "alim_grp_nom_fr", col]].head(top_n).rename(
        columns={"alim_nom_fr": "Aliment", "alim_grp_nom_fr": "Groupe", col: f"Teneur ({col.split('(')[1]}"}
    ).reset_index(drop=True)

# ── Page ──────────────────────────────────────────────────────────────────────

# TODO (équipe) : ajoutez votre CSS ici
# st.markdown("<style> ... </style>", unsafe_allow_html=True)

st.title("🩺 Diagnostic IA")
st.markdown("Renseignez les données du patient pour obtenir une prédiction de carence.")
st.markdown("---")

# ── Formulaire ────────────────────────────────────────────────────────────────
with st.form("diagnostic"):

    st.subheader("Apports nutritionnels (% AJR)")
    c1, c2, c3, c4 = st.columns(4)
    vit_a   = c1.slider("Vitamine A",   0, 150, 65)
    vit_c   = c2.slider("Vitamine C",   0, 150, 70)
    vit_d   = c3.slider("Vitamine D",   0, 150, 30)
    vit_e   = c4.slider("Vitamine E",   0, 150, 60)

    c5, c6, c7, c8 = st.columns(4)
    vit_b12 = c5.slider("Vitamine B12", 0, 150, 55)
    folate  = c6.slider("Folate",       0, 150, 50)
    calcium = c7.slider("Calcium",      0, 150, 70)
    iron    = c8.slider("Fer",          0, 150, 60)

    st.markdown("---")
    st.subheader("Marqueurs biologiques")
    b1, b2, b3, b4 = st.columns(4)
    hemoglobin    = b1.number_input("Hémoglobine (g/dL)",        5.0, 20.0,   13.5, 0.1)
    serum_vit_d   = b2.number_input("Vit. D sérique (ng/mL)",    1.0, 100.0,  18.0, 0.5)
    serum_vit_b12 = b3.number_input("Vit. B12 sérique (pg/mL)", 50.0, 1000.0, 185.0, 5.0)
    serum_folate  = b4.number_input("Folate sérique (ng/mL)",    1.0, 30.0,    6.5, 0.5)

    st.markdown("---")
    st.subheader("Symptômes")
    s1, s2, s3 = st.columns(3)
    has_fatigue         = s1.checkbox("Fatigue chronique")
    has_bone_pain       = s1.checkbox("Douleurs osseuses")
    has_muscle_weakness = s1.checkbox("Faiblesse musculaire")
    has_numbness        = s2.checkbox("Engourdissements")
    has_memory_problems = s2.checkbox("Troubles de mémoire")
    has_pale_skin       = s2.checkbox("Peau pâle")
    has_night_blindness = s3.checkbox("Cécité nocturne")
    has_bleeding_gums   = s3.checkbox("Gencives saignantes")
    has_multiple        = s3.checkbox("Carences multiples suspectées")

    st.markdown("---")
    top_n    = st.slider("Nombre d'aliments recommandés", 3, 15, 10)
    submitted = st.form_submit_button("🔍 Lancer l'analyse", use_container_width=True)

# ── Résultats ─────────────────────────────────────────────────────────────────
if submitted:
    patient = {
        "vitamin_a_percent_rda":     vit_a,
        "vitamin_c_percent_rda":     vit_c,
        "vitamin_d_percent_rda":     vit_d,
        "vitamin_e_percent_rda":     vit_e,
        "vitamin_b12_percent_rda":   vit_b12,
        "folate_percent_rda":        folate,
        "calcium_percent_rda":       calcium,
        "iron_percent_rda":          iron,
        "hemoglobin_g_dl":           hemoglobin,
        "serum_vitamin_d_ng_ml":     serum_vit_d,
        "serum_vitamin_b12_pg_ml":   serum_vit_b12,
        "serum_folate_ng_ml":        serum_folate,
        "has_night_blindness":       int(has_night_blindness),
        "has_fatigue":               int(has_fatigue),
        "has_bleeding_gums":         int(has_bleeding_gums),
        "has_bone_pain":             int(has_bone_pain),
        "has_muscle_weakness":       int(has_muscle_weakness),
        "has_numbness_tingling":     int(has_numbness),
        "has_memory_problems":       int(has_memory_problems),
        "has_pale_skin":             int(has_pale_skin),
        "has_multiple_deficiencies": int(has_multiple),
    }

    df_input  = pd.DataFrame([patient])[feature_cols]
    pred_enc  = model.predict(df_input)[0]
    proba     = model.predict_proba(df_input)[0]
    diagnosis = le.inverse_transform([pred_enc])[0]
    confidence = round(float(max(proba)) * 100, 1)

    st.markdown("---")
    st.subheader("Résultat")

    res_col, chart_col = st.columns(2)

    with res_col:
        if diagnosis == "Healthy":
            st.success(f"✅ {DIAGNOSIS_FR[diagnosis]}")
        else:
            st.error(f"⚠️ {DIAGNOSIS_FR[diagnosis]}")

        st.metric("Confiance du modèle", f"{confidence}%")
        st.caption("⚠️ Cet outil est une aide à la décision. Consultez un médecin pour confirmation.")

        # Top 3 probabilités
        st.markdown("**Probabilités par classe :**")
        top3 = sorted(zip(le.classes_, proba), key=lambda x: -x[1])[:3]
        for cls, prob in top3:
            st.write(f"- {DIAGNOSIS_FR[cls]} : **{round(prob*100, 1)}%**")

    with chart_col:
        # Radar chart apports nutritionnels
        import plotly.graph_objects as go
        categories = ["Vit. A", "Vit. C", "Vit. D", "Vit. E", "Vit. B12", "Folate", "Calcium", "Fer"]
        values     = [vit_a, vit_c, vit_d, vit_e, vit_b12, folate, calcium, iron]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself", name="Patient"
        ))
        fig.add_trace(go.Scatterpolar(
            r=[80] * len(categories) + [80],
            theta=categories + [categories[0]],
            fill="toself", name="Référence 80%",
            line=dict(dash="dot")
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 150])),
            title="Profil nutritionnel (%AJR)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Recommandations CIQUAL ────────────────────────────────────────────────
    vitamin = DIAGNOSIS_VITAMIN.get(diagnosis)
    if vitamin:
        st.markdown("---")
        st.subheader(f"🥗 Recommandations alimentaires — {vitamin}")
        st.caption("Source : Base CIQUAL – ANSES")

        df_food = get_recommendations(vitamin, top_n)
        if not df_food.empty:
            col_teneur = df_food.columns[-1]

            tab1, tab2 = st.tabs(["Graphique", "Tableau"])
            with tab1:
                fig2 = px.bar(
                    df_food, x=col_teneur, y="Aliment",
                    color="Groupe", orientation="h",
                    title=f"Aliments les plus riches en {vitamin}"
                )
                fig2.update_layout(yaxis=dict(categoryorder="total ascending"), height=400)
                st.plotly_chart(fig2, use_container_width=True)
            with tab2:
                st.dataframe(df_food, use_container_width=True, hide_index=True)
