# processors/nlp.py
# Analyse du sentiment des actualités géopolitiques
# VADER enrichi avec lexique financier-géopolitique spécialisé
# Référence : Loughran & McDonald (2011), Hutto & Gilbert (2014)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Lexique financier-géopolitique custom ──
# Scores entre -1 (très négatif) et +1 (très positif)
# Référence : Loughran & McDonald (2011) — Journal of Finance
LEXIQUE_CUSTOM = {
    # Tier 1 — Escalade militaire directe (très négatif)
    "airstrike":       -0.9,
    "airstrikes":      -0.9,
    "bombardment":     -0.9,
    "invasion":        -0.9,
    "invaded":         -0.9,
    "missile":         -0.8,
    "missiles":        -0.8,
    "nuclear":         -0.8,
    "bomb":            -0.8,
    "bombed":          -0.9,
    "bombing":         -0.9,
    "killed":          -0.8,
    "casualties":      -0.8,
    "troops":          -0.6,
    "military strike": -0.9,
    "war":             -0.9,
    "warfare":         -0.9,

    # Tier 2 — Tensions diplomatiques (négatif modéré)
    "sanctions":       -0.7,
    "sanctioned":      -0.6,
    "blockade":        -0.7,
    "embargo":         -0.6,
    "escalation":      -0.7,
    "escalating":      -0.7,
    "escalated":       -0.7,
    "hawkish":         -0.6,
    "risk-off":        -0.6,
    "flight-to-safety": -0.5,
    "safe-haven":       0.2,
    "terror":          -0.8,
    "terrorist":       -0.8,
    "coup":            -0.7,
    "siege":           -0.7,
    "hostilities":     -0.7,
    "confrontation":   -0.5,
    "standoff":        -0.5,
    "threat":          -0.5,
    "threatened":      -0.5,

    # Tier 3 — Préoccupations géopolitiques (légèrement négatif)
    "geopolitical":    -0.3,
    "tension":         -0.4,
    "tensions":        -0.4,
    "conflict":        -0.6,
    "crisis":          -0.6,
    "instability":     -0.5,
    "uncertainty":     -0.4,
    "volatile":        -0.4,
    "volatility":      -0.3,
    "concern":         -0.2,
    "warning":         -0.3,
    "pressure":        -0.2,

    # Désescalade (positif)
    "ceasefire":       0.8,
    "peace deal":      0.9,
    "peace talks":     0.6,
    "de-escalation":   0.8,
    "de-escalating":   0.8,
    "truce":           0.7,
    "armistice":       0.8,
    "diplomatic":      0.3,
    "negotiation":     0.4,
    "negotiations":    0.4,
    "agreement":       0.5,
    "stability":       0.5,
    "stabilize":       0.5,
    "calm":            0.3,
    "resolved":        0.6,
    "resolution":      0.5,
}


def analyze_sentiment(news_data):
    """
    Analyse le sentiment des actualités géopolitiques collectées.

    Args:
        news_data : dict retourné par collect_news()

    Returns:
        dict avec nlp_score, nlp_signal, label,
        articles_analyzed, details
    """
    logger.info("Analyse NLP démarrée")

    articles = news_data.get("articles", [])

    if not articles:
        logger.warning("Aucun article à analyser")
        return _score_neutre()

    # Initialiser VADER avec lexique custom
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(LEXIQUE_CUSTOM)

    scores_ponderes = []
    details         = []
    now_iso         = datetime.utcnow().isoformat()

    for article in articles:
        # Texte à analyser : titre + résumé
        titre   = article.get("title",   "")
        summary = article.get("summary", "")
        texte   = titre + ". " + summary

        # Score VADER
        vader_scores = analyzer.polarity_scores(texte)
        compound     = vader_scores["compound"]  # [-1, +1]

        # Pondération par tier et fraîcheur
        poids_tier     = _poids_tier(article.get("tier", 3))
        poids_fraicheur = _poids_fraicheur(
            article.get("published_at"), now_iso
        )
        poids_total = poids_tier * poids_fraicheur

        scores_ponderes.append((compound, poids_total))

        details.append({
            "title":    titre[:80],
            "compound": round(compound, 4),
            "tier":     article.get("tier"),
            "poids":    round(poids_total, 4),
            "pos":      vader_scores["pos"],
            "neg":      vader_scores["neg"],
            "neu":      vader_scores["neu"],
        })

    # Score NLP global pondéré
    if not scores_ponderes:
        return _score_neutre()

    scores  = np.array([s for s, _ in scores_ponderes])
    poids   = np.array([p for _, p in scores_ponderes])
    poids   = poids / poids.sum()  # normaliser les poids

    nlp_score = float(np.average(scores, weights=poids))

    # Conversion en signal [0, 1] pour le score composite
    # nlp_score ∈ [-1, +1] → nlp_signal ∈ [0, 1]
    # -1 (désescalade) → 0.0
    #  0 (neutre)      → 0.5
    # +1 (escalade)    → 1.0
    # MAIS dans notre formule multiplicative :
    # nlp_signal ∈ [-1, +1] directement
    # (négatif = actualités alarmistes = risque plus élevé)
    # On inverse : actualités négatives → signal positif
    nlp_signal = -nlp_score  # inverser : négatif VADER = escalade

    # Label
    label = _label_sentiment(nlp_signal)

    # Statistiques par tier
    tier_stats = _stats_par_tier(details)

    result = {
        "nlp_score":         round(nlp_score, 4),
        "nlp_signal":        round(nlp_signal, 4),
        "label":             label,
        "articles_analyzed": len(articles),
        "tier_stats":        tier_stats,
        "top_articles":      _top_articles(details),
        "computed_at":       now_iso,
    }

    logger.info(
        f"NLP terminé — score={nlp_score:.4f}, "
        f"signal={nlp_signal:.4f} ({label}), "
        f"{len(articles)} articles analysés"
    )

    return result


def _poids_tier(tier):
    """Pondération selon le tier géopolitique de l'article."""
    poids = {1: 1.0, 2: 0.6, 3: 0.3}
    return poids.get(tier, 0.3)


def _poids_fraicheur(published_at, now_iso):
    """
    Pondération selon la fraîcheur de l'article.
    Décroissance exponentielle sur 48h.
    Articles récents (< 6h) : poids 1.0
    Articles anciens (48h)  : poids ~0.15
    """
    if not published_at:
        return 0.5

    try:
        from datetime import datetime
        pub  = datetime.fromisoformat(published_at.replace("Z", ""))
        now  = datetime.fromisoformat(now_iso)
        age_heures = (now - pub).total_seconds() / 3600
        age_heures = max(0, age_heures)

        # Décroissance exponentielle : e^(-age/20)
        poids = np.exp(-age_heures / 20)
        return float(np.clip(poids, 0.05, 1.0))

    except Exception:
        return 0.5


def _label_sentiment(nlp_signal):
    """Interprète le signal NLP en label lisible."""
    if nlp_signal > 0.6:
        return "escalade forte"
    elif nlp_signal > 0.3:
        return "escalade modérée"
    elif nlp_signal > 0.1:
        return "légère tension"
    elif nlp_signal > -0.1:
        return "neutre"
    elif nlp_signal > -0.3:
        return "légère détente"
    else:
        return "désescalade"


def _stats_par_tier(details):
    """Calcule les statistiques de sentiment par tier."""
    stats = {}
    for tier in [1, 2, 3]:
        articles_tier = [d for d in details if d["tier"] == tier]
        if articles_tier:
            scores_tier = [d["compound"] for d in articles_tier]
            stats[f"tier_{tier}"] = {
                "count":       len(articles_tier),
                "score_moyen": round(np.mean(scores_tier), 4),
                "score_min":   round(min(scores_tier), 4),
                "score_max":   round(max(scores_tier), 4),
            }
    return stats


def _top_articles(details):
    """Retourne les 5 articles avec le score le plus négatif."""
    sorted_details = sorted(
        details, key=lambda x: x["compound"]
    )
    return sorted_details[:5]


def _score_neutre():
    """Retourne un score NLP neutre en cas d'absence de données."""
    return {
        "nlp_score":         0.0,
        "nlp_signal":        0.0,
        "label":             "neutre",
        "articles_analyzed": 0,
        "tier_stats":        {},
        "top_articles":      [],
        "computed_at":       datetime.utcnow().isoformat(),
    }