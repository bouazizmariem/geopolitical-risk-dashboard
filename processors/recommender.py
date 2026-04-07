import numpy as np
from datetime import datetime

# ── Poids estimés empiriquement ──
WEIGHTS_GEO = {
    'gpr': 0.3067,  # U-MIDAS estimé
    'vix': 0.4080,  # U-MIDAS estimé
    'epu': 0.2852,  # U-MIDAS estimé
    'nlp': 0.20,    # fixé — Loughran & McDonald (2011)
}

WEIGHTS_FIN = {
    'gpr': 0.4549,  # U-MIDAS estimé — dominant sur horizon mensuel
    'epu': 0.4491,  # U-MIDAS estimé
    'vix': 0.0960,  # U-MIDAS estimé — faible sur horizon mensuel
}

ALPHA_NLP = 0.20  # modulateur NLP — Bloom (2009)

# ── Seuils de classification ──
SEUILS = {
    "faible":  0.25,
    "modere":  0.50,
    "eleve":   0.75,
}

# ── Allocations par niveau et par score ──
ALLOCATIONS = {
    "geo": {
        "faible":  {"or": 0.20, "petrole": 0.10, "defense": 0.05, "argent": 0.05},
        "modere":  {"or": 0.35, "petrole": 0.25, "defense": 0.10, "argent": 0.10},
        "eleve":   {"or": 0.50, "petrole": 0.25, "defense": 0.15, "argent": 0.10},
        "crise":   {"or": 0.65, "petrole": 0.20, "defense": 0.10, "argent": 0.05},
    },
    "fin": {
        "faible":  {"obligations": 0.30, "dollar": 0.10, "cash": 0.00},
        "modere":  {"obligations": 0.35, "dollar": 0.15, "cash": 0.10},
        "eleve":   {"obligations": 0.25, "dollar": 0.20, "cash": 0.30},
        "crise":   {"obligations": 0.15, "dollar": 0.15, "cash": 0.30},
    },
}

def _niveau(score):
    """Convertit un score [0,1] en niveau de risque."""
    if score < SEUILS["faible"]:
        return "faible"
    elif score < SEUILS["modere"]:
        return "modere"
    elif score < SEUILS["eleve"]:
        return "eleve"
    else:
        return "crise"

def _label_niveau(niveau):
    labels = {
        "faible": "Faible",
        "modere": "Modéré",
        "eleve":  "Élevé",
        "crise":  "Crise majeure",
    }
    return labels.get(niveau, niveau)

def compute_geo_score(gpr_norm, nlp_norm, epu_norm, vix_norm):
    """
    Score géopolitique — pilote les actifs géopolitiques
    (or, pétrole, défense, argent).
    Poids estimés par U-MIDAS avec l'or comme variable cible.
    """
    base = (
        WEIGHTS_GEO['gpr'] * gpr_norm +
        WEIGHTS_GEO['vix'] * vix_norm +
        WEIGHTS_GEO['epu'] * epu_norm
    )
    score = base * (1 + ALPHA_NLP * nlp_norm)
    return float(np.clip(score, 0, 1))

def compute_fin_score(vix_norm, epu_norm, gpr_norm, nlp_norm):
    """
    Score financier — pilote les actifs financiers
    (obligations, dollar, cash).
    Poids estimés par U-MIDAS avec les obligations comme variable cible.
    Note : GPR et EPU dominent sur horizon mensuel (Bekaert & Hoerova 2014).
    """
    base = (
        WEIGHTS_FIN['gpr'] * gpr_norm +
        WEIGHTS_FIN['epu'] * epu_norm +
        WEIGHTS_FIN['vix'] * vix_norm
    )
    score = base * (1 + ALPHA_NLP * nlp_norm)
    return float(np.clip(score, 0, 1))

def compute_confidence_score(geo_score, fin_score, nlp_signal):
    """
    Score de confiance (0-100%) basé sur la cohérence
    entre les deux scores et le signal NLP.
    """
    # Divergence entre les deux scores
    divergence = abs(geo_score - fin_score)

    # Cohérence NLP : NLP dans le même sens que les scores
    score_moyen = (geo_score + fin_score) / 2
    if score_moyen > 0.5 and nlp_signal > 0:
        coherence_nlp = 1.0
    elif score_moyen < 0.5 and nlp_signal < 0:
        coherence_nlp = 1.0
    elif score_moyen > 0.5 and nlp_signal < 0:
        coherence_nlp = 0.4
    elif score_moyen < 0.5 and nlp_signal > 0:
        coherence_nlp = 0.4
    else:
        coherence_nlp = 0.7

    # Score de confiance
    confiance = (1 - divergence) * coherence_nlp
    return round(float(np.clip(confiance, 0, 1)) * 100, 1)

def generate_recommendation(gpr_norm, vix_norm, epu_norm, nlp_signal):
    """
    Génère la recommandation complète d'allocation.

    Args:
        gpr_norm   : GPR normalisé [0,1]
        vix_norm   : VIX normalisé [0,1]
        epu_norm   : EPU normalisé [0,1]
        nlp_signal : score NLP [-1,+1]

    Returns:
        dict avec scores, niveaux, allocation et justification
    """
    # Calculer les deux scores
    geo = compute_geo_score(gpr_norm, nlp_signal, epu_norm, vix_norm)
    fin = compute_fin_score(vix_norm, epu_norm, gpr_norm, nlp_signal)

    # Niveaux de risque
    niveau_geo = _niveau(geo)
    niveau_fin = _niveau(fin)

    # Score global (moyenne des deux)
    score_global = round((geo + fin) / 2, 4)
    niveau_global = _niveau(score_global)

    # Confiance
    confiance = compute_confidence_score(geo, fin, nlp_signal)

    # Allocations
    alloc_geo = ALLOCATIONS["geo"][niveau_geo]
    alloc_fin = ALLOCATIONS["fin"][niveau_fin]

    # Allocation combinée
    allocation = {**alloc_geo, **alloc_fin}

    # Cas de divergence
    divergence = abs(geo - fin)
    alerte_divergence = divergence > 0.25

    # Justification
    justification = _generer_justification(
        geo, fin, niveau_geo, niveau_fin,
        gpr_norm, vix_norm, epu_norm, nlp_signal,
        alerte_divergence
    )

    return {
        "timestamp":          datetime.utcnow().isoformat(),
        "score_geo":          round(geo, 4),
        "score_fin":          round(fin, 4),
        "score_global":       score_global,
        "niveau_geo":         _label_niveau(niveau_geo),
        "niveau_fin":         _label_niveau(niveau_fin),
        "niveau_global":      _label_niveau(niveau_global),
        "confiance":          confiance,
        "alerte_divergence":  alerte_divergence,
        "allocation":         allocation,
        "justification":      justification,
        "inputs": {
            "gpr_norm":   round(gpr_norm, 4),
            "vix_norm":   round(vix_norm, 4),
            "epu_norm":   round(epu_norm, 4),
            "nlp_signal": round(nlp_signal, 4),
        }
    }

def _generer_justification(geo, fin, niv_geo, niv_fin,
                            gpr, vix, epu, nlp, divergence):
    """Génère une justification textuelle de la recommandation."""
    parties = []

    # Score géopolitique
    parties.append(
        f"Score géopolitique : {geo:.2f} ({_label_niveau(niv_geo)}) — "
        f"piloté par GPR={gpr:.2f}, VIX={vix:.2f}, EPU={epu:.2f}."
    )

    # Score financier
    parties.append(
        f"Score financier : {fin:.2f} ({_label_niveau(niv_fin)}) — "
        f"piloté par GPR={gpr:.2f}, EPU={epu:.2f} sur horizon mensuel."
    )

    # Signal NLP
    if nlp > 0.3:
        parties.append(
            f"Signal NLP positif ({nlp:.2f}) : "
            f"les actualités géopolitiques récentes signalent une escalade."
        )
    elif nlp < -0.3:
        parties.append(
            f"Signal NLP négatif ({nlp:.2f}) : "
            f"les actualités géopolitiques récentes signalent une désescalade."
        )
    else:
        parties.append(
            f"Signal NLP neutre ({nlp:.2f}) : "
            f"pas de signal médiatique fort détecté."
        )

    # Alerte divergence
    if divergence:
        if geo > fin:
            parties.append(
                "ALERTE : divergence entre les scores — crise géopolitique sans "
                "panique financière totale. Privilégier or et pétrole, "
                "alléger les obligations."
            )
        else:
            parties.append(
                "ALERTE : divergence entre les scores — panique financière sans "
                "crise géopolitique majeure. Privilégier obligations et cash, "
                "réduire l'exposition aux matières premières."
            )

    return " ".join(parties)