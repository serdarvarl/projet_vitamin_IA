"""
VitaIA — Page Diagnostic patient
==================================
Formulaire de saisie des données cliniques du patient et prédiction de carence
par le modèle Random Forest v5 (30 variables, holdout 30 %, F1 = 94,06 %).

Fonctionnalités :
    - Chargement du modèle sérialisé (pkl) avec fallback v3 si v5 absent
    - Formulaire structuré en 3 sections : apports nutritionnels (% AJR),
      biomarqueurs sériques, symptômes cliniques
    - Prédiction avec probabilités par classe (distribution barres)
    - Radar chart nutritionnel (Plotly) comparé à la référence 80 % AJR
    - Recommandations alimentaires depuis la base CIQUAL / ANSES

Sources / Références :
    - Scikit-learn — RandomForestClassifier :
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    - Streamlit forms : https://docs.streamlit.io/library/api-reference/control-flow/st.form
    - Base CIQUAL (ANSES) — Table de composition nutritionnelle des aliments :
        https://ciqual.anses.fr/#/cms/telechargement/node/20
    - Plotly Scatterpolar (radar) :
        https://plotly.com/python/radar-chart/
    - Stack Overflow — lecture fichiers ODS avec pandas/odfpy :
        https://stackoverflow.com/questions/17834995/how-to-convert-opendocument-spreadsheets-to-a-csv-file-in-python

Auteurs : Ibnmtar Hazem, Moutchachou Lydia, Varol Serdar, Bekakria Ahmed
Formation : L3 MIASHS — Université Paul-Valéry Montpellier 3 — 2025/2026
"""

import sys
import os
import pickle
import base64

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CIQUAL_ODS = os.path.join(REPO, "data_csv", "raw", "Table_Ciqual_V2.ods")
LOGO_PATH  = os.path.join(os.path.dirname(__file__), "..", "assets", "logos", "logo2_pilule_neurone.svg")

# ── Chargement CIQUAL ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement de la base CIQUAL...")
def load_ciqual():
    """
    Charge et prétraite la table CIQUAL depuis le fichier ODS.

    La table CIQUAL (ANSES) contient la composition nutritionnelle de plusieurs
    milliers d'aliments français. Les colonnes numériques sont détectées et
    converties automatiquement (gestion des virgules décimales françaises).

    Returns:
        pd.DataFrame | None: DataFrame CIQUAL ou None si fichier absent ou erreur.

    Ref: ANSES (2020). Table Ciqual 2020. https://ciqual.anses.fr
    """
    if not os.path.exists(CIQUAL_ODS):
        return None
    try:
        df = pd.read_excel(CIQUAL_ODS, engine="odf")
    except Exception as e:
        st.error(f"Impossible de lire Table_Ciqual_V2.ods : {e}\n"
                 "Installez odfpy : pip install odfpy")
        return None
    # Convertir toutes les colonnes potentiellement numériques
    for col in df.columns:
        if df[col].dtype == object:
            try:
                converted = pd.to_numeric(
                    df[col].astype(str).str.replace(",", ".").str.strip(),
                    errors="coerce"
                )
                # Ne remplacer que si au moins 30% des valeurs sont numériques
                if converted.notna().mean() > 0.3:
                    df[col] = converted
            except Exception:
                pass
    return df

# Mapping maladie → colonnes riche_ et colonne teneur pour tri
DISEASE_CIQUAL = {
    "Anemia": {
        "col_mots": ["Fer (", "Fer("],
        "label_filtre": "les plus riches en Fer (sources naturelles)",
        "groupes_exclus": ["produits céréaliers", "boissons"],
    },
    "Rickets_Osteomalacia": {
        "col_mots": ["VitamineD", "Vitamine D"],
        "label_filtre": "les plus riches en Vitamine D",
        "groupes_exclus": ["boissons"],
    },
    "Night_Blindness": {
        "col_mots": ["tinol", "Retinol", "R\xe9tinol", "vitamine A", "Vitamine A"],
        "label_filtre": "les plus riches en Vitamine A / Rétinol",
        "groupes_exclus": ["boissons"],
    },
    "Scurvy": {
        "col_mots": ["VitamineC", "Vitamine C"],
        "label_filtre": "les plus riches en Vitamine C",
        "groupes_exclus": ["boissons"],
    },
}

# Mots-clés à exclure (aliments enrichis artificiellement ou peu pratiques)
EXCLUSIONS = [
    "enrichi", "enrichie", "enrichies", "enrichis",
    "compl", "suppl", "fortifi",
    "desh", "s\xe9ch\xe9", "s\xe9ch\xe9e",
    "algue", "chlorelle", "spiruline", "lithothamne",
]

def _find_col(df, mots):
    """
    Trouve la première colonne dont le nom contient un des mots-clés donnés.

    Args:
        df (pd.DataFrame): DataFrame CIQUAL.
        mots (list[str]): Liste de sous-chaînes à rechercher dans les noms de colonnes.

    Returns:
        str | None: Nom de la première colonne correspondante, ou None.
    """
    for mot in mots:
        matches = [c for c in df.columns if mot in c]
        if matches:
            return matches[0]
    return None

def get_ciqual_aliments(disease, df_ciqual, top_n=12):
    """
    Retourne les aliments CIQUAL les plus riches en nutriment(s) ciblés pour une carence.

    Exclut les aliments enrichis artificiellement (spiruline, compléments, etc.)
    ainsi que certains groupes alimentaires (boissons, céréales) jugés moins pertinents
    comme sources naturelles. Trie par teneur décroissante et déduplique.

    Args:
        disease (str): Nom de la maladie/carence ('Anemia', 'Rickets_Osteomalacia',
                       'Night_Blindness', 'Scurvy').
        df_ciqual (pd.DataFrame): Table CIQUAL chargée par load_ciqual().
        top_n (int): Nombre maximum d'aliments à retourner (défaut : 12).

    Returns:
        pd.DataFrame | None: Tableau avec colonnes Aliment, Groupe alimentaire,
                             Teneur (unité/100g), ou None si aucun résultat.

    Ref: Base CIQUAL ANSES — https://ciqual.anses.fr
    """
    if df_ciqual is None or disease not in DISEASE_CIQUAL:
        return None
    info = DISEASE_CIQUAL[disease]

    try:
        # Trouver la colonne de tri
        col_tri = _find_col(df_ciqual, info["col_mots"])
        if col_tri is None:
            return None

        df = df_ciqual.copy()

        # S'assurer que la colonne est numérique
        df[col_tri] = pd.to_numeric(
            df[col_tri].astype(str).str.replace(",", ".").str.strip(),
            errors="coerce"
        )

        # Exclure les aliments enrichis ou peu pratiques (comparaison simple sans accents)
        nom_lower = df["alim_nom_fr"].astype(str).str.lower()
        excl_mask = pd.Series([False] * len(df), index=df.index)
        for kw in EXCLUSIONS:
            excl_mask = excl_mask | nom_lower.str.contains(kw, na=False, regex=False)
        df = df[~excl_mask]

        # Exclure certains groupes alimentaires
        groupes_excl = info.get("groupes_exclus", [])
        if groupes_excl and "alim_grp_nom_fr" in df.columns:
            grp_col = df["alim_grp_nom_fr"].astype(str).str.lower()
            grp_mask = pd.Series([False] * len(df), index=df.index)
            for g in groupes_excl:
                grp_mask = grp_mask | grp_col.str.contains(g, na=False, regex=False)
            df = df[~grp_mask]

        # Garder seulement les aliments avec une vraie teneur > 0
        df = df[pd.to_numeric(df[col_tri], errors="coerce") > 0].dropna(
            subset=[col_tri, "alim_nom_fr"]
        )

        if df.empty:
            return None

        # Trier par teneur décroissante
        result = (df[["alim_nom_fr", "alim_grp_nom_fr", col_tri]]
                  .sort_values(col_tri, ascending=False)
                  .drop_duplicates("alim_nom_fr")
                  .head(top_n)
                  .reset_index(drop=True))
        result.index = result.index + 1

        # Renommer les colonnes proprement
        unite = col_tri.split("(")[-1].rstrip(")").strip() if "(" in col_tri else "?"
        result.columns = ["Aliment", "Groupe alimentaire", f"Teneur ({unite}/100g)"]
        return result

    except Exception as e:
        st.warning(f"Erreur recommandations CIQUAL : {e}")
        return None

st.set_page_config(
    page_title="VitaIA — Diagnostic",
    page_icon="💊",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
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

    /* Conteneur formulaire */
    .form-card {
        background: #fff;
        border: 1px solid #d8e2ec;
        box-shadow: 0 2px 12px rgba(26,58,92,0.07);
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
    }

    /* Titres de section dans le formulaire */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #2e6da4;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        border-bottom: 2px solid #2e6da4;
        padding-bottom: 5px;
        margin: 0 0 1.1rem;
    }

    /* Résultat diagnostic */
    .result-box {
        border-left: 5px solid;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.5rem;
        background: #f8fafc;
        box-shadow: 0 2px 10px rgba(26,58,92,0.07);
    }
    .result-title { font-size: 1.35rem; font-weight: 700; margin-bottom: 6px; }
    .result-msg   { font-size: 0.95rem; color: #424242; }

    /* Barres de probabilité */
    .prob-row   { display: flex; align-items: center; margin: 6px 0; gap: 10px; font-size: 0.85rem; color: #212529; }
    .prob-label { width: 200px; flex-shrink: 0; font-weight: 500; }

    /* Chips aliments */
    .food-chip {
        display: inline-block;
        background: #eef2f7;
        color: #1a3a5c;
        border: 1px solid #c8d8e8;
        padding: 5px 13px;
        margin: 4px;
        font-size: 0.84rem;
        font-weight: 500;
    }

    /* Logo sidebar agrandi */
    [data-testid="stSidebarHeader"] img {
        height: 90px !important;
        width: auto !important;
    }
    [data-testid="stSidebarHeader"] { padding: 1rem 1rem 0.5rem !important; }

    /* Masquer chrome Streamlit */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Recommandations (fallback si pas de CIQUAL) ───────────────────────────────
RECO = {
    "Healthy": {
        "color": "#1b7a4a", "border": "#2e9e60",
        "titre": "Aucune carence détectée",
        "message": "Continuez avec une alimentation variée et équilibrée.",
        "aliments": ["Fruits et légumes variés", "Céréales complètes", "Légumineuses",
                     "Produits laitiers", "Protéines maigres"],
        "conseil": "Maintenir l'exposition solaire régulière et l'activité physique.",
    },
    "Anemia": {
        "color": "#b71c1c", "border": "#c62828",
        "titre": "Anémie (carence en fer / B12 / folate)",
        "message": "Carence probable en fer et/ou vitamine B12 ou folate.",
        "aliments": ["Foie de boeuf", "Lentilles et pois chiches", "Viande rouge",
                     "Épinards", "Huîtres", "Haricots rouges", "Tofu", "Graines de courge"],
        "conseil": "Associer fer végétal + vitamine C pour améliorer l'absorption. "
                   "Éviter thé/café lors des repas.",
    },
    "Rickets_Osteomalacia": {
        "color": "#b45309", "border": "#d97706",
        "titre": "Rachitisme / Ostéomalacie (carence en vitamine D)",
        "message": "Carence probable en vitamine D et/ou calcium.",
        "aliments": ["Saumon, thon, maquereau", "Jaune d'oeuf", "Champignons exposés au soleil",
                     "Lait et yaourts enrichis en vitamine D", "Sardines en boîte", "Fromages à pâte dure"],
        "conseil": "Augmenter l'exposition solaire (15-30 min/jour). Consulter un médecin "
                   "pour une supplémentation en vitamine D3.",
    },
    "Night_Blindness": {
        "color": "#92400e", "border": "#b45309",
        "titre": "Cécité nocturne (carence en vitamine A)",
        "message": "Carence probable en vitamine A.",
        "aliments": ["Carottes", "Patate douce", "Foie de volaille",
                     "Épinards et brocolis", "Mangue et abricots", "Lait entier", "Oeufs (jaune)"],
        "conseil": "La vitamine A est liposoluble : consommer avec un corps gras. "
                   "Éviter les suppléments en excès (toxicité possible).",
    },
    "Scurvy": {
        "color": "#5b21b6", "border": "#7c3aed",
        "titre": "Scorbut (carence sévère en vitamine C)",
        "message": "Carence sévère probable en vitamine C.",
        "aliments": ["Kiwi (plus concentré)", "Poivron rouge et jaune",
                     "Orange et citron", "Fraises", "Brocoli", "Persil frais", "Cassis"],
        "conseil": "Consommer cru ou peu cuit (vitamine C fragile à la chaleur). "
                   "200-500 mg/jour corrige rapidement la carence.",
    },
}

# ── Chargement modèle ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Charge le modèle Random Forest sérialisé avec ses artefacts associés.

    Tente d'abord le modèle final v5 (random_state=999, holdout 30 %, 30 features).
    Si absent, bascule automatiquement sur le modèle v3 (21 features, test 20 %).

    Le modèle v5 est un RandomForestClassifier direct (non Pipeline) : le scaler
    StandardScaler est sérialisé séparément dans scaler_final.pkl et doit être
    appliqué manuellement avant toute prédiction.

    Returns:
        tuple: (model, label_encoder, feature_cols, scaler, model_label)
               Tous None si aucun fichier pkl n'est trouvé.

    Ref:
        - scikit-learn RandomForestClassifier :
          https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        - Python pickle : https://docs.python.org/3/library/pickle.html
        - Streamlit cache_resource : https://docs.streamlit.io/library/api-reference/performance/st.cache_resource
    """
    for folder, suffix, label in [
        (os.path.join(REPO, "notebooks", "models_final"), "final",
         "v5 final — Random Forest 30 features (holdout F1 = 94,06 %)"),
        (os.path.join(REPO, "notebooks", "models_v3"), "v3",
         "v3 — Random Forest 21 features"),
    ]:
        mp = os.path.join(folder, f"random_forest_{suffix}.pkl") if suffix == "final" else os.path.join(folder, f"best_model_{suffix}.pkl")
        lp = os.path.join(folder, f"label_encoder_{suffix}.pkl")
        fp = os.path.join(folder, f"feature_cols_{suffix}.pkl")
        sp = os.path.join(folder, f"scaler_{suffix}.pkl")
        if os.path.exists(mp) and os.path.exists(lp) and os.path.exists(fp):
            try:
                model  = pickle.load(open(mp, "rb"))
                le     = pickle.load(open(lp, "rb"))
                feats  = pickle.load(open(fp, "rb"))
                scaler = pickle.load(open(sp, "rb")) if os.path.exists(sp) else None
                return model, le, feats, scaler, label
            except Exception as e:
                st.warning(f"Erreur chargement modèle ({suffix}) : {e}")
                continue
    return None, None, None, None, None

model, le, feature_cols, scaler, model_label = load_model()

# ── Sidebar logo ──────────────────────────────────────────────────────────────
if os.path.exists(LOGO_PATH):
    st.logo(LOGO_PATH, size="large")

# ── Bandeau ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h2>Diagnostic patient</h2>
    <p>Renseignez les données cliniques pour obtenir une prédiction de carence en vitamines — modèle Random Forest v5</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(
        "**Modèle introuvable.** Lancez d'abord un notebook pour générer les fichiers `.pkl` :\n\n"
        "```\nnotebooks/modeles_IA_v5_holdout_final.ipynb\n```\n\n"
        "Consultez le fichier `streamlit/LANCER_APP.md` pour les instructions complètes."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# FORMULAIRE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="form-card">', unsafe_allow_html=True)
with st.form("diagnostic_form"):

    # Apports nutritionnels
    st.markdown('<div class="section-label">Apports nutritionnels (% des AJR)</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    vit_a   = c1.number_input("Vitamine A (%)",   0.0, 300.0, 100.0, 1.0)
    vit_c   = c2.number_input("Vitamine C (%)",   0.0, 300.0, 100.0, 1.0)
    vit_d   = c3.number_input("Vitamine D (%)",   0.0, 300.0, 100.0, 1.0)
    vit_e   = c4.number_input("Vitamine E (%)",   0.0, 300.0, 100.0, 1.0)
    c5, c6, c7, c8 = st.columns(4)
    vit_b12 = c5.number_input("Vitamine B12 (%)", 0.0, 300.0, 100.0, 1.0)
    folate  = c6.number_input("Folate (%)",        0.0, 300.0, 100.0, 1.0)
    calcium = c7.number_input("Calcium (%)",       0.0, 300.0, 100.0, 1.0)
    iron    = c8.number_input("Fer (%)",           0.0, 300.0, 100.0, 1.0)

    # Biomarqueurs
    st.markdown('<div class="section-label">Biomarqueurs sériques</div>',
                unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    hemo       = b1.number_input("Hémoglobine (g/dL)",          4.0,  20.0,  13.5, 0.1,
                                  help="Normale : 12–17 g/dL (H) / 12–15 g/dL (F)")
    ser_vit_d  = b2.number_input("Vitamine D sérique (ng/mL)", 5.0, 120.0,  30.0, 0.5,
                                  help="Optimal ≥ 30 · Insuffisance 12–30 · Carence < 12")
    ser_b12    = b3.number_input("Vitamine B12 (pg/mL)",       50.0,1200.0, 400.0, 5.0,
                                  help="Normale 200–900 pg/mL · Carence < 200")
    ser_folate = b4.number_input("Folate sérique (ng/mL)",      1.0,  40.0,  10.0, 0.5,
                                  help="Normale > 3 ng/mL · Optimal 5–20")

    # Symptômes
    st.markdown('<div class="section-label">Symptômes cliniques</div>',
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
        "Analyser le profil patient", use_container_width=True, type="primary"
    )
st.markdown('</div>', unsafe_allow_html=True)

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
    <div class="result-box" style="border:2px solid {reco['border']}">
        <div class="result-title" style="color:{reco['color']}">{reco['titre']}</div>
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
            bar_color = reco["border"] if cls == pred_label else "#c8d0d8"
            st.markdown(
                f'<div class="prob-row">'
                f'<span class="prob-label">{cls.replace("_"," ")}</span>'
                f'<span style="background:{bar_color};height:12px;'
                f'width:{pct*2.2:.0f}px;display:inline-block;border-radius:3px"></span>'
                f'<span style="color:#444;font-size:0.82rem">{pct:.1f} %</span>'
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
            line=dict(color=reco["border"], width=2),
            fillcolor="rgba(46,109,164,0.12)",
        ))
        fig.add_trace(go.Scatterpolar(
            r=[80] * len(cats) + [80], theta=cats + [cats[0]],
            fill="toself", name="Référence 80 %",
            line=dict(color="#9e9e9e", dash="dot", width=1),
            fillcolor="rgba(200,200,200,0.1)",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 200], tickfont=dict(size=9))),
            title=dict(text="Profil nutritionnel (% AJR)", font=dict(size=13, color="#1a3a5c")),
            legend=dict(orientation="h", y=-0.15),
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=350,
            margin=dict(t=60, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Recommandations alimentaires CIQUAL ──────────────────────────────────
    st.markdown("---")
    st.markdown("### Recommandations alimentaires")

    df_ciqual = load_ciqual()

    if pred_label != "Healthy":
        df_aliments = get_ciqual_aliments(pred_label, df_ciqual)
        info_ciqual = DISEASE_CIQUAL.get(pred_label, {})

        col_food, col_tip = st.columns([3, 2], gap="large")
        with col_food:
            if df_aliments is not None and not df_aliments.empty:
                st.markdown(
                    f"**Top aliments {info_ciqual.get('label_filtre','')} "
                    f"— Base CIQUAL / ANSES ({len(df_aliments)} résultats) :**"
                )
                st.dataframe(
                    df_aliments[["Aliment", "Groupe alimentaire"]],
                    use_container_width=True,
                    hide_index=False,
                )
            else:
                st.markdown("**Aliments recommandés :**")
                for food in reco["aliments"]:
                    st.markdown(f'<span class="food-chip">{food}</span>',
                                unsafe_allow_html=True)
        with col_tip:
            st.info(f"**Conseil :** {reco['conseil']}")
    else:
        st.markdown("**Aucune carence détectée** — maintenez une alimentation équilibrée.")
        st.info(f"**Conseil :** {reco['conseil']}")

    st.markdown("---")
    st.caption(
        "⚠️ Cet outil est un aide à la décision à usage pédagogique. "
        "Il ne remplace pas un avis médical professionnel."
    )
