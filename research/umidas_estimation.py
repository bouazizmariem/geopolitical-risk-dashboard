import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings('ignore')

os.makedirs("research/outputs", exist_ok=True)

# ============================================================
# ÉTAPE 1 : COLLECTE DES DONNÉES
# ============================================================
print("=" * 60)
print("ÉTAPE 1 — Collecte des données historiques")
print("=" * 60)

# ── VIX journalier ──
print("\n[1/3] Téléchargement VIX journalier...")
vix = yf.download("^VIX", start="1990-01-01",
                  end="2025-01-01", progress=False)["Close"]
vix.index = pd.to_datetime(vix.index)
if isinstance(vix.columns, pd.MultiIndex):
    vix = vix.iloc[:, 0]
vix = vix.squeeze()
vix.name = "vix"
print(f"   OK — {len(vix)} observations ({vix.index[0].date()} → {vix.index[-1].date()})")

# ── GPR mensuel ──
print("\n[2/3] Téléchargement GPR (Caldara & Iacoviello)...")
try:
    gpr_url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    gpr_raw = pd.read_excel(gpr_url, engine="xlrd")
    date_col = gpr_raw.columns[0]
    dates_parsed = []
    for v in gpr_raw[date_col]:
        try:
            s = str(v).strip().upper().replace("M", "-")
            dates_parsed.append(pd.to_datetime(s, errors="coerce"))
        except:
            dates_parsed.append(pd.NaT)
    gpr_raw.index = pd.DatetimeIndex(dates_parsed)
    gpr_raw = gpr_raw[gpr_raw.index.notna()]
    gpr_raw.index = gpr_raw.index.map(lambda x: x.replace(day=1))
    gpr = pd.to_numeric(gpr_raw["GPR"], errors="coerce").dropna()
    gpr = gpr[gpr.index >= "1990-01-01"]
    gpr.name = "gpr"
    print(f"   OK — {len(gpr)} obs (min={gpr.min():.1f}, max={gpr.max():.1f})")
except Exception as e:
    print(f"   Erreur GPR : {e} → données simulées")
    dates = pd.date_range("1990-01-01", "2024-12-01", freq="MS")
    np.random.seed(42)
    gpr = pd.Series(100 + np.cumsum(np.random.randn(len(dates)) * 5),
                    index=dates, name="gpr").clip(50, 500)

# ── EPU mensuel ──
print("\n[3/3] Téléchargement EPU (Baker, Bloom & Davis)...")
try:
    epu_url = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx"
    epu_raw = pd.read_excel(epu_url, engine="openpyxl")
    for col in ["Year", "Month"]:
        epu_raw[col] = pd.to_numeric(epu_raw[col], errors="coerce")
    epu_raw = epu_raw.dropna(subset=["Year", "Month"])
    epu_raw = epu_raw[
        (epu_raw["Year"] >= 1985) & (epu_raw["Year"] <= 2025) &
        (epu_raw["Month"] >= 1)   & (epu_raw["Month"] <= 12)
    ]
    epu_raw["Year"]  = epu_raw["Year"].astype(int)
    epu_raw["Month"] = epu_raw["Month"].astype(int)
    epu_raw["date"]  = pd.to_datetime(
        {"year": epu_raw["Year"], "month": epu_raw["Month"], "day": 1}
    )
    epu_raw = epu_raw.set_index("date").sort_index()
    epu_col = [c for c in epu_raw.columns
               if any(k in str(c).lower()
                      for k in ["news", "uncertainty", "epu"])][0]
    epu = pd.to_numeric(epu_raw[epu_col], errors="coerce").dropna()
    epu = epu[epu.index >= "1990-01-01"]
    epu.name = "epu"
    print(f"   OK — {len(epu)} obs (min={epu.min():.1f}, max={epu.max():.1f})")
except Exception as e:
    print(f"   Erreur EPU : {e} → données simulées")
    dates = pd.date_range("1990-01-01", "2024-12-01", freq="MS")
    np.random.seed(99)
    epu = pd.Series(120 + np.cumsum(np.random.randn(len(dates)) * 8),
                    index=dates, name="epu").clip(50, 600)

# ── Actifs surveillés ──
print("\n[+] Téléchargement des actifs surveillés...")
ACTIFS = {
    "gold":    "GC=F",
    "oil":     "CL=F",
    "bonds":   "^TNX",
    "dollar":  "DX-Y.NYB",
    "defense": "ITA",
    "silver":  "SI=F",
    "btc":     "BTC-USD",
}

prices = {}
for nom, ticker in ACTIFS.items():
    try:
        data = yf.download(ticker, start="1990-01-01",
                           end="2025-01-01", progress=False)["Close"]
        data.index = pd.to_datetime(data.index)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.iloc[:, 0]
        data = data.squeeze()
        data.name = nom
        prices[nom] = data
        print(f"   {nom:10} ({ticker:12}) : {len(data)} obs")
    except Exception as e:
        print(f"   {nom:10} ERREUR : {e}")

print("\n✓ Étape 1 terminée")

# ============================================================
# ÉTAPE 2 : PRÉTRAITEMENT ET ALIGNEMENT
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 2 — Prétraitement et alignement des fréquences")
print("=" * 60)

# ── Normalisation des indices mensuels ──
print("\n[1/4] Normalisation Min-Max des indices mensuels...")

scaler_gpr = MinMaxScaler()
scaler_epu = MinMaxScaler()

gpr_norm = pd.Series(
    scaler_gpr.fit_transform(gpr.values.reshape(-1, 1)).flatten(),
    index=gpr.index, name="gpr_norm"
)
epu_norm = pd.Series(
    scaler_epu.fit_transform(epu.values.reshape(-1, 1)).flatten(),
    index=epu.index, name="epu_norm"
)
print(f"   GPR normalisé : [{gpr_norm.min():.3f}, {gpr_norm.max():.3f}]")
print(f"   EPU normalisé : [{epu_norm.min():.3f}, {epu_norm.max():.3f}]")

# ── Agrégation VIX mensuel + normalisation ──
print("\n[2/4] Agrégation et normalisation VIX mensuel...")

vix_monthly = pd.DataFrame({
    "vix_mean": vix.resample("MS").mean(),
    "vix_max":  vix.resample("MS").max(),
    "vix_std":  vix.resample("MS").std(),
}).dropna()

scaler_vix = MinMaxScaler()
vix_monthly_norm = pd.DataFrame(
    scaler_vix.fit_transform(vix_monthly),
    index=vix_monthly.index,
    columns=["vix_mean_norm", "vix_max_norm", "vix_std_norm"]
)
print(f"   VIX mean normalisé : [{vix_monthly_norm['vix_mean_norm'].min():.3f}, {vix_monthly_norm['vix_mean_norm'].max():.3f}]")

# ── Variations mensuelles de chaque actif ──
print("\n[3/4] Calcul des variations mensuelles de chaque actif...")

returns = {}
for nom, series in prices.items():
    try:
        monthly = series.resample("MS").last().dropna()
        ret = np.log(monthly / monthly.shift(1)) * 100
        ret = ret.dropna()
        ret.name = f"{nom}_return"
        returns[nom] = ret
        print(f"   Δ{nom:10} : {len(ret)} obs | "
              f"moy={ret.mean():.2f}% | std={ret.std():.2f}%")
    except Exception as e:
        print(f"   Δ{nom:10} ERREUR : {e}")

# ── Dataset mensuel de base ──
print("\n[4/4] Assemblage du dataset mensuel de base...")

df_base = pd.DataFrame({
    "gpr_norm":      gpr_norm,
    "epu_norm":      epu_norm,
    "vix_mean_norm": vix_monthly_norm["vix_mean_norm"],
    "vix_max_norm":  vix_monthly_norm["vix_max_norm"],
})

for nom, ret in returns.items():
    df_base[f"{nom}_return"] = ret

df_base = df_base.dropna(subset=["gpr_norm", "epu_norm", "vix_mean_norm"])
df_base = df_base[
    (df_base.index >= "1990-01-01") &
    (df_base.index <= "2024-12-31")
]

print(f"\n   Observations  : {len(df_base)}")
print(f"   Période       : {df_base.index[0].date()} → {df_base.index[-1].date()}")
print(f"   Colonnes      : {list(df_base.columns)}")

df_base.to_csv("research/outputs/dataset_monthly_base.csv")
print("\n✓ Étape 2 terminée")

# ============================================================
# ÉTAPE 3 : CONSTRUCTION DU DATASET U-MIDAS
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 3 — Construction du dataset U-MIDAS")
print("=" * 60)

print("\nConstruction des 20 colonnes journalières du VIX...")

# Normaliser le VIX journalier
scaler_vix_daily = MinMaxScaler()
vix_norm_daily = pd.Series(
    scaler_vix_daily.fit_transform(vix.values.reshape(-1, 1)).flatten(),
    index=vix.index, name="vix_norm"
)

umidas_rows = []
MAX_JOURS = 22  # jours ouvrables max par mois

for date in df_base.index:
    row = {"date": date}

    # Indices mensuels
    row["gpr_norm"] = df_base.loc[date, "gpr_norm"]
    row["epu_norm"] = df_base.loc[date, "epu_norm"]

    # 20 valeurs journalières du VIX pour ce mois
    debut = date
    fin   = date + pd.offsets.MonthEnd(0)
    vix_mois = vix_norm_daily[
        (vix_norm_daily.index >= debut) &
        (vix_norm_daily.index <= fin)
    ].values

    # Remplir jusqu'à MAX_JOURS colonnes
    for j in range(MAX_JOURS):
        if j < len(vix_mois):
            row[f"vix_j{j+1:02d}"] = vix_mois[j]
        else:
            row[f"vix_j{j+1:02d}"] = np.nan

    # Variables cibles
    for nom in returns.keys():
        col = f"{nom}_return"
        if col in df_base.columns and not pd.isna(df_base.loc[date, col]):
            row[col] = df_base.loc[date, col]

    umidas_rows.append(row)

df_umidas = pd.DataFrame(umidas_rows).set_index("date")

# Remplir les NaN dans les colonnes VIX par la moyenne du mois
vix_cols = [c for c in df_umidas.columns if c.startswith("vix_j")]
df_umidas[vix_cols] = df_umidas[vix_cols].apply(
    lambda row: row.fillna(row.mean()), axis=1
)

# Supprimer les lignes avec trop de NaN
df_umidas = df_umidas.dropna(subset=["gpr_norm", "epu_norm"] + vix_cols[:5])

print(f"\n   Dataset U-MIDAS :")
print(f"   Observations   : {len(df_umidas)}")
print(f"   Colonnes total : {len(df_umidas.columns)}")
print(f"   Cols indices   : gpr_norm, epu_norm + {len(vix_cols)} cols VIX journalières")
print(f"   Cols cibles    : {[c for c in df_umidas.columns if 'return' in c]}")

df_umidas.to_csv("research/outputs/dataset_umidas.csv")
print("\n✓ Étape 3 terminée")

# ============================================================
# ÉTAPE 4 : ESTIMATION U-MIDAS PAR ACTIF (NIVEAU 2)
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 4 — Estimation U-MIDAS par actif (Niveau 2)")
print("=" * 60)

# Colonnes explicatives X
X_COLS = ["gpr_norm", "epu_norm"] + vix_cols

# Définition des deux groupes d'actifs
GROUPE_GEO = ["gold", "oil", "defense", "silver"]
GROUPE_FIN = ["bonds", "dollar", "btc"]

# Alphas Ridge à tester
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Stockage des résultats
resultats = {}

def estimer_umidas(df, actif, x_cols, alphas):
    """Estime la régression U-MIDAS Ridge pour un actif donné."""
    col_y = f"{actif}_return"
    if col_y not in df.columns:
        return None

    df_clean = df[x_cols + [col_y]].dropna()
    if len(df_clean) < 50:
        print(f"   {actif:12} : pas assez de données ({len(df_clean)} obs)")
        return None

    X = df_clean[x_cols].values
    y = df_clean[col_y].values

    # Validation croisée temporelle (TimeSeriesSplit)
    tscv = TimeSeriesSplit(n_splits=5)

    # Ridge avec cross-validation
    model = RidgeCV(
        alphas=alphas,
        cv=tscv,
        fit_intercept=True,
        scoring="r2"
    )
    model.fit(X, y)

    # Prédictions et métriques
    y_pred = model.predict(X)
    r2     = r2_score(y, y_pred)
    rmse   = np.sqrt(mean_squared_error(y, y_pred))

    # Extraction des poids bruts
    coefs = model.coef_

    # Poids GPR = coef[0]
    # Poids EPU = coef[1]
    # Poids VIX = somme des coefs journaliers (coef[2:])
    w_gpr = coefs[0]
    w_epu = coefs[1]
    w_vix = coefs[2:].sum()

    # Forcer positivité (remplacer négatifs par 0)
    w_gpr = max(w_gpr, 0)
    w_epu = max(w_epu, 0)
    w_vix = max(w_vix, 0)

    # Normaliser pour que la somme = 1
    total = w_gpr + w_epu + w_vix
    if total > 0:
        w_gpr /= total
        w_epu /= total
        w_vix /= total
    else:
        w_gpr, w_epu, w_vix = 0.35, 0.20, 0.45

    return {
        "actif":       actif,
        "n_obs":       len(df_clean),
        "alpha":       model.alpha_,
        "r2":          round(r2, 4),
        "rmse":        round(rmse, 4),
        "w_gpr":       round(w_gpr, 4),
        "w_epu":       round(w_epu, 4),
        "w_vix":       round(w_vix, 4),
        "dominant":    max(["GPR", "EPU", "VIX"],
                          key=lambda k: {"GPR": w_gpr, "EPU": w_epu, "VIX": w_vix}[k])
    }

# ── Estimation pour chaque actif ──
print(f"\n{'Actif':12} {'N':>5} {'R²':>8} {'RMSE':>8} "
      f"{'w_GPR':>8} {'w_VIX':>8} {'w_EPU':>8} {'Dominant':>10}")
print("-" * 75)

for actif in list(GROUPE_GEO) + list(GROUPE_FIN):
    res = estimer_umidas(df_umidas, actif, X_COLS, ALPHAS)
    if res:
        resultats[actif] = res
        print(f"{actif:12} {res['n_obs']:>5} {res['r2']:>8.4f} "
              f"{res['rmse']:>8.4f} {res['w_gpr']:>8.4f} "
              f"{res['w_vix']:>8.4f} {res['w_epu']:>8.4f} "
              f"{res['dominant']:>10}")

# ============================================================
# ÉTAPE 5 : CALCUL DES POIDS FINAUX PAR GROUPE
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 5 — Poids finaux par groupe d'actifs")
print("=" * 60)

def poids_groupe(groupe, resultats):
    """Calcule les poids moyens d'un groupe d'actifs."""
    actifs_ok = [a for a in groupe if a in resultats]
    if not actifs_ok:
        return None

    # Pondérer par R² (les actifs mieux prédits ont plus d'influence)
    r2_vals = np.array([max(resultats[a]["r2"], 0.01) for a in actifs_ok])
    poids_r2 = r2_vals / r2_vals.sum()

    w_gpr = sum(poids_r2[i] * resultats[a]["w_gpr"] for i, a in enumerate(actifs_ok))
    w_epu = sum(poids_r2[i] * resultats[a]["w_epu"] for i, a in enumerate(actifs_ok))
    w_vix = sum(poids_r2[i] * resultats[a]["w_vix"] for i, a in enumerate(actifs_ok))

    return {
        "w_gpr": round(w_gpr, 4),
        "w_epu": round(w_epu, 4),
        "w_vix": round(w_vix, 4),
        "actifs_inclus": actifs_ok,
    }

poids_geo = poids_groupe(GROUPE_GEO, resultats)
poids_fin = poids_groupe(GROUPE_FIN, resultats)

print("\n── Score géopolitique (actifs : or, pétrole, défense, argent) ──")
if poids_geo:
    print(f"   w_GPR = {poids_geo['w_gpr']:.4f}  ({poids_geo['w_gpr']*100:.1f}%)")
    print(f"   w_VIX = {poids_geo['w_vix']:.4f}  ({poids_geo['w_vix']*100:.1f}%)")
    print(f"   w_EPU = {poids_geo['w_epu']:.4f}  ({poids_geo['w_epu']*100:.1f}%)")
    print(f"   Formule : Score_geo = {poids_geo['w_gpr']:.2f}×GPR + {poids_geo['w_vix']:.2f}×VIX + {poids_geo['w_epu']:.2f}×EPU")

print("\n── Score financier (actifs : obligations, dollar, bitcoin) ──")
if poids_fin:
    print(f"   w_GPR = {poids_fin['w_gpr']:.4f}  ({poids_fin['w_gpr']*100:.1f}%)")
    print(f"   w_VIX = {poids_fin['w_vix']:.4f}  ({poids_fin['w_vix']*100:.1f}%)")
    print(f"   w_EPU = {poids_fin['w_epu']:.4f}  ({poids_fin['w_epu']*100:.1f}%)")
    print(f"   Formule : Score_fin = {poids_fin['w_vix']:.2f}×VIX + {poids_fin['w_epu']:.2f}×EPU + {poids_fin['w_gpr']:.2f}×GPR")

# ============================================================
# ÉTAPE 6 : VALIDATION ET COHÉRENCE
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 6 — Validation et tests de cohérence")
print("=" * 60)

print("\n── Test 1 : GPR dominant dans le score géopolitique ──")
if poids_geo:
    test1 = poids_geo["w_gpr"] > poids_geo["w_epu"]
    print(f"   GPR ({poids_geo['w_gpr']:.3f}) > EPU ({poids_geo['w_epu']:.3f}) : "
          f"{'✓ VALIDÉ' if test1 else '✗ ÉCHOUÉ'}")
    print(f"   Attendu par Caldara & Iacoviello (2022) : GPR dominant")

print("\n── Test 2 : VIX dominant dans le score financier ──")
if poids_fin:
    test2 = poids_fin["w_vix"] > poids_fin["w_gpr"]
    print(f"   VIX ({poids_fin['w_vix']:.3f}) > GPR ({poids_fin['w_gpr']:.3f}) : "
          f"{'✓ VALIDÉ' if test2 else '✗ ÉCHOUÉ'}")
    print(f"   Attendu par Whaley (2000) : VIX dominant")

print("\n── Test 3 : Comparaison avec poids heuristiques initiaux ──")
print("   Poids heuristiques initiaux : GPR=0.35, VIX=0.25, EPU=0.20")
if poids_geo:
    ecart_gpr = abs(poids_geo["w_gpr"] - 0.35)
    ecart_vix = abs(poids_geo["w_vix"] - 0.25)
    ecart_epu = abs(poids_geo["w_epu"] - 0.20)
    print(f"   Écart GPR : {ecart_gpr:.3f}  {'✓ Proche' if ecart_gpr < 0.10 else '→ Différent'}")
    print(f"   Écart VIX : {ecart_vix:.3f}  {'✓ Proche' if ecart_vix < 0.10 else '→ Différent'}")
    print(f"   Écart EPU : {ecart_epu:.3f}  {'✓ Proche' if ecart_epu < 0.10 else '→ Différent'}")

print("\n── Métriques de performance par actif ──")
print(f"\n{'Actif':12} {'R²':>8} {'Qualité':>12}")
print("-" * 35)
for actif, res in resultats.items():
    if res["r2"] >= 0.20:
        qualite = "✓ Bon"
    elif res["r2"] >= 0.10:
        qualite = "~ Acceptable"
    else:
        qualite = "✗ Faible"
    print(f"{actif:12} {res['r2']:>8.4f} {qualite:>12}")

# ============================================================
# ÉTAPE 7 : SAUVEGARDE DES POIDS FINAUX
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 7 — Sauvegarde des poids pour le pipeline")
print("=" * 60)

# Poids par actif individuel
weights_par_actif = {
    actif: {
        "w_gpr": res["w_gpr"],
        "w_vix": res["w_vix"],
        "w_epu": res["w_epu"],
        "r2":    res["r2"],
        "rmse":  res["rmse"],
        "dominant": res["dominant"],
    }
    for actif, res in resultats.items()
}

# Poids par groupe
weights_groupes = {
    "geo": poids_geo,
    "fin": poids_fin,
    "alpha_nlp": 0.20,
    "methode": "U-MIDAS Ridge (Foroni, Marcellino & Schumacher, 2015)",
    "periode_calibration": "1990-01-01 to 2024-12-31",
}

# Sauvegarde JSON
with open("research/outputs/weights_par_actif.json", "w") as f:
    json.dump(weights_par_actif, f, indent=2)

with open("research/outputs/weights_groupes.json", "w") as f:
    json.dump(weights_groupes, f, indent=2)

# Rapport CSV
rapport = pd.DataFrame(resultats).T
rapport.to_csv("research/outputs/rapport_calibration.csv")

print("\n   Fichiers sauvegardés :")
print("   research/outputs/weights_par_actif.json")
print("   research/outputs/weights_groupes.json")
print("   research/outputs/rapport_calibration.csv")

# ── Affichage du code à intégrer dans recommender.py ──
print("\n" + "=" * 60)
print("CODE À INTÉGRER DANS processors/recommender.py")
print("=" * 60)

if poids_geo and poids_fin:
    print(f"""
# Poids estimés par U-MIDAS — Foroni, Marcellino & Schumacher (2015)
# Calibration sur données historiques 1990-2024
# Méthode : Ridge Regression avec TimeSeriesSplit CV

WEIGHTS_GEO = {{
    'gpr': {poids_geo['w_gpr']},
    'nlp': 0.30,   # fixé par jugement expert (Loughran & McDonald 2011)
    'epu': {poids_geo['w_epu']},
    'vix': {poids_geo['w_vix']},
}}

WEIGHTS_FIN = {{
    'vix': {poids_fin['w_vix']},
    'epu': {poids_fin['w_epu']},
    'gpr': {poids_fin['w_gpr']},
}}

ALPHA_NLP = 0.20  # Bloom (2009) — Econometrica

def compute_geo_score(gpr_norm, nlp_norm, epu_norm, vix_norm):
    base = (WEIGHTS_GEO['gpr'] * gpr_norm
          + WEIGHTS_GEO['epu'] * epu_norm
          + WEIGHTS_GEO['vix'] * vix_norm)
    return float(np.clip(base * (1 + ALPHA_NLP * nlp_norm), 0, 1))

def compute_fin_score(vix_norm, epu_norm, gpr_norm, nlp_norm):
    base = (WEIGHTS_FIN['vix'] * vix_norm
          + WEIGHTS_FIN['epu'] * epu_norm
          + WEIGHTS_FIN['gpr'] * gpr_norm)
    return float(np.clip(base * (1 + ALPHA_NLP * nlp_norm), 0, 1))
""")

print("\n" + "=" * 60)
print("✓ CALIBRATION U-MIDAS NIVEAU 2 TERMINÉE")
print("=" * 60)