import sys
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

st.set_page_config(
    page_title="VitaIA — Diagnostic",
    page_icon="🔍",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #1a237e;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        border-bottom: 2px solid #3949ab;
        padding-bottom: 3px;
        margin: 1.4rem 0 0.8rem;
    }
    .result-box {
        border-radius: 12px;
        padding: 1.6rem 1.4rem;
        margin-bottom: 1.2rem;
    }
    .result-title { font-size: 1.7rem; font-weight: 700; }
    .result-msg   { font-size: 1rem; margin-top: 6px; opacity: 0.92; }
    .prob-row     { display: flex; align-items: center; margin: 4px 0; gap: 8px; font-size: 0.85rem; }
    .prob-label   { width: 210px; flex-shrink: 0; }
    .food-chip {
        display: inline-block;
        background: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 4px 11px;
        margin: 3px;
        font-size: 0.84rem;
    }
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Recommandations (fallback si pas de CIQUAL) ───────────────────────────────
RECO = {
    "Healthy": {
        "emoji": "✅", "color": "#2e7d32", "bg": "#f1f8e9",
        "titre": "Aucune carence détectée",
        "message": "Continuez avec une alimentation variée et équilibrée.",
        "aliments": ["Fruits et légumes variés", "Céréales complètes", "Légumineuses",
                     "Produits laitiers", "Protéines maigres"],
        "conseil": "Maintenir l'exposition solaire régulière et l'activité physique.",
    },
    "Anemia": {
        "emoji": "🔴", "color": "#b71c1c", "bg": "#ffebee",
        "titre": "Anémie (carence en fer / B12 / folate)",
        "message": "Carence probable en fer et/ou vitamine B12 ou folate.",
        "aliments": ["Foie de bœuf", "Lentilles et pois chiches", "Viande rouge",
                     "Épinards", "Huîtres", "Haricots rouges", "Tofu", "Graines de courge"],
        "conseil": "Associer fer végétal + vitamine C pour améliorer l'absorption. "
                   "Éviter thé/café lors des repas.",
    },
    "Rickets_Osteomalacia": {
        "emoji": "🦴", "color": "#e65100", "bg": "#fff3e0",
        "titre": "Rachitisme / Ostéomalacie (carence en vitamine D)",
        "message": "Carence probable en vitamine D et/ou calcium.",
        "aliments": ["Saumon, thon, maquereau", "Jaune d'œuf", "Champignons exposés au soleil",
                     "Lait et yaourts enrichis en vitamine D", "Sardines en boîte", "Fromages à pâte dure"],
        "conseil": "Augmenter l'exposition solaire (15–30 min/jour). Consulter un médecin "
                   "pour une supplémentation en vitamine D3.",
    },
    "Night_Blindness": {
        "emoji": "👁️", "color": "#f57f17", "bg": "#fffde7",
        "titre": "Cécité nocturne (carence en vitamine A)",
        "message": "Carence probable en vitamine A.",
        "aliments": ["Carottes", "Patate douce", "Foie de volaille",
                     "Épinards et brocolis", "Mangue et abricots", "Lait entier", "Œufs (jaune)"],
        "conseil": "La vitamine A est liposoluble : consommer avec un corps gras. "
                   "Éviter les suppléments en excès (toxicité possible).",
    },
    "Scurvy": {
        "emoji": "🍊", "color": "#6a1b9a", "bg": "#f3e5f5",
        "titre": "Scorbut (carence sévère en vitamine C)",
        "message": "Carence sévère probable en vitamine C.",
        "aliments": ["Kiwi (plus concentré)", "Poivron rouge et jaune",
                     "Orange et citron", "Fraises", "Brocoli", "Persil frais", "Cassis"],
        "conseil": "Consommer cru ou peu cuit (vitamine C fragile à la chaleur). "
                   "200–500 mg/jour corrige rapidement la carence.",
    },
}

# ── Chargement modèle ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    for folder, suffix, label in [
        (os.path.join(REPO, "notebooks", "models_final"), "final",
         "v5 final — Random Forest 30 features (holdout F1 = 94,06 %)"),
        (os.path.join(REPO, "notebooks", "models_v3"), "v3",
         "v3 — Random Forest 21 features (évaluation antérieure)"),
    ]:
        mp = os.path.join(folder, f"best_model_{suffix}.pkl")
        lp = os.path.join(folder, f"label_encoder_{suffix}.pkl")
        fp = os.path.join(folder, f"feature_cols_{suffix}.pkl")
        sp = os.path.join(folder, f"scaler_{suffix}.pkl")
        if os.path.exists(mp) and os.path.exists(lp) and os.path.exists(fp):
            model  = pickle.load(open(mp, "rb"))
            le     = pickle.load(open(lp, "rb"))
            feats  = pickle.load(open(fp, "rb"))
            scaler = pickle.load(open(sp, "rb")) if os.path.exists(sp) else None
            return model, le, feats, scaler, label
    return None, None, None, None, None

model, le, feature_cols, scaler, model_label = load_model()

# ── Titre ─────────────────────────────────────────────────────────────────────
st.title("🔍 Diagnostic patient")
st.markdown("Renseignez les données cliniques pour obtenir une prédiction de carence en vitamines.")

if model is None:
    st.error(
        "Aucun modèle trouvé. Lancez d'abord le notebook `modeles_IA_v5_holdout_final.ipynb` "
        "ou `modeles_IA_v3.ipynb` pour générer les fichiers `.pkl`."
    )
    st.stop()

st.info(f"**Modèle chargé :** {model_label}", icon="🤖")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# FORMULAIRE
# ═══════════════════════════════════════════════════════════════════════════════
with st.form("diagnostic_form"):

    # Apports nutritionnels
    st.markdown('<div class="section-label">📊 Apports nutritionnels (% des AJR)</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    vit_a   = c1.number_input("Vitamine A (%)",   0.0, 300.0, 80.0, 1.0)
    vit_c   = c2.number_input("Vitamine C (%)",   0.0, 300.0, 80.0, 1.0)
    vit_d   = c3.number_input("Vitamine D (%)",   0.0, 300.0, 60.0, 1.0)
    vit_e   = c4.number_input("Vitamine E (%)",   0.0, 300.0, 80.0, 1.0)
    c5, c6, c7, c8 = st.columns(4)
    vit_b12 = c5.number_input("Vitamine B12 (%)", 0.0, 300.0, 80.0, 1.0)
    folate  = c6.number_input("Folate (%)",        0.0, 300.0, 80.0, 1.0)
    calcium = c7.number_input("Calcium (%)",       0.0, 300.0, 80.0, 1.0)
    iron    = c8.number_input("Fer (%)",           0.0, 300.0, 80.0, 1.0)

    # Biomarqueurs
    st.markdown('<div class="section-label">🧪 Biomarqueurs sériques</div>',
                unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    hemo       = b1.number_input("Hémoglobine (g/dL)",        4.0,  20.0,  13.5, 0.1,
                                  help="Normale : 12–17 g/dL")
    ser_vit_d  = b2.number_input("Vitamine D sérique (ng/mL)", 5.0, 120.0,  30.0, 0.5,
                                  help="Carence < 12 ng/mL")
    ser_b12    = b3.number_input("Vitamine B12 (pg/mL)",      50.0,1200.0, 350.0, 5.0,
                                  help="Carence < 200 pg/mL")
    ser_folate = b4.number_input("Folate sérique (ng/mL)",     1.0,  40.0,  12.0, 0.5,
                                  help="Carence < 3 ng/mL")

    # Symptômes
    st.markdown('<div class="section-label">🩺 Symptômes cliniques</div>',
                unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    night_blind = s1.checkbox("Cécité nocturne")
    fatigue     = s1.checkbox("Fatigue chronique")
    bleeding    = s2.checkbox("Gingivorragies")
    bone_pain   = s2.checkbox("Douleurs osseuses")
    muscle_weak = s3.checkbox("Faiblesse musculaire")
    numbness    = s3.checkbox("Engourdissements")
    memory      = s4.checkbox("Troubles de mémoire")
    pale_skin   = s4.checkbox("Pâleur cutanée")
    multi_def   = st.checkbox("Carences multiples suspectées")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button(
        "🔮 Analyser le profil patient", use_container_width=True, type="primary"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PRÉDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if submitted:
    input_map = {
        "vitamin_a_percent_rda":     vit_a,
        "vitamin_c_percent_rda":     vit_c,
        "vitamin_d_percent_rda":     vit_d,
        "vitamin_e_percent_rda":     vit_e,
        "vitamin_b12_percent_rda":   vit_b12,
        "folate_percent_rda":        folate,
        "calcium_percent_rda":       calcium,
        "iron_percent_rda":          iron,
        "hemoglobin_g_dl":           hemo,
        "serum_vitamin_d_ng_ml":     ser_vit_d,
        "serum_vitamin_b12_pg_ml":   ser_b12,
        "serum_folate_ng_ml":        ser_folate,
        "has_night_blindness":       int(night_blind),
        "has_fatigue":               int(fatigue),
        "has_bleeding_gums":         int(bleeding),
        "has_bone_pain":             int(bone_pain),
        "has_muscle_weakness":       int(muscle_weak),
        "has_numbness_tingling":     int(numbness),
        "has_memory_problems":       int(memory),
        "has_pale_skin":             int(pale_skin),
        "has_multiple_deficiencies": int(multi_def),
    }

    df_input = pd.DataFrame([{f: input_map.get(f, 0.0) for f in feature_cols}])
    X = df_input.values
    if scaler is not None:
        X = scaler.transform(X)

    pred_enc   = model.predict(X)[0]
    pred_label = le.inverse_transform([pred_enc])[0]
    probas     = model.predict_proba(X)[0]
    confidence = float(probas.max()) * 100

    reco = RECO[pred_label]

    st.markdown("---")

    # ── Résultat principal ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-box" style="background:{reco['bg']};border:2px solid {reco['color']}">
        <div class="result-title" style="color:{reco['color']}">{reco['emoji']} {reco['titre']}</div>
        <div class="result-msg">{reco['message']}</div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    # ── Colonne gauche : confiance + barres proba ─────────────────────────────
    with left:
        st.metric("Confiance du modèle", f"{confidence:.1f} %")
        if confidence >= 85:
            st.success("Prédiction fiable")
        elif confidence >= 65:
            st.warning("Prédiction modérée — confirmation clinique recommandée")
        else:
            st.error("Confiance faible — résultat indicatif seulement")

        st.markdown("**Distribution des probabilités :**")
        classes = le.inverse_transform(range(len(probas)))
        for cls, prob in sorted(zip(classes, probas), key=lambda x: -x[1]):
            pct = prob * 100
            bar_color = reco["color"] if cls == pred_label else "#bdbdbd"
            st.markdown(
                f'<div class="prob-row">'
                f'<span class="prob-label">{cls.replace("_"," ")}</span>'
                f'<span style="background:{bar_color};height:12px;'
                f'width:{pct*2.2:.0f}px;display:inline-block;border-radius:3px"></span>'
                f'<span style="color:#555">{pct:.1f} %</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Colonne droite : radar nutritionnel ───────────────────────────────────
    with right:
        cats   = ["Vit. A", "Vit. C", "Vit. D", "Vit. E", "Vit. B12", "Folate", "Calcium", "Fer"]
        values = [vit_a, vit_c, vit_d, vit_e, vit_b12, folate, calcium, iron]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]], theta=cats + [cats[0]],
            fill="toself", name="Patient",
            line=dict(color=reco["color"], width=2),
            fillcolor=reco["color"].replace(")", ",0.15)").replace("rgb", "rgba")
                       if reco["color"].startswith("rgb") else reco["color"] + "33",
        ))
        fig.add_trace(go.Scatterpolar(
            r=[80] * len(cats) + [80], theta=cats + [cats[0]],
            fill="toself", name="Référence 80 %",
            line=dict(color="#9e9e9e", dash="dot", width=1),
            fillcolor="rgba(200,200,200,0.1)",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 200], tickfont=dict(size=9))),
            title=dict(text="Profil nutritionnel (% AJR)", font=dict(size=13)),
            legend=dict(orientation="h", y=-0.15),
            height=350,
            margin=dict(t=60, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Recommandations alimentaires ─────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🥗 Recommandations alimentaires")
    col_food, col_tip = st.columns([1, 1], gap="large")
    with col_food:
        st.markdown("**Aliments à privilégier :**")
        for food in reco["aliments"]:
            st.markdown(f'<span class="food-chip">🍽 {food}</span>', unsafe_allow_html=True)
    with col_tip:
        st.info(f"**Conseil :** {reco['conseil']}")

    st.markdown("---")
    st.caption(
        "⚠️ Cet outil est un aide à la décision à usage pédagogique. "
        "Il ne remplace pas un avis médical professionnel."
    )
