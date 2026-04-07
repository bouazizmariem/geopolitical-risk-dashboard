# main.py
# Orchestrateur du pipeline Geopolitical Risk Dashboard
# Exécuté automatiquement toutes les 12h via GitHub Actions

import logging
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Charger les variables d'environnement
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Imports des modules du pipeline
from collectors.gpr    import collect_gpr
from collectors.assets import collect_assets
from collectors.news   import collect_news
from processors.nlp    import analyze_sentiment
from processors.recommender import generate_recommendation
from storage.mongo_client   import save_snapshot, test_connection


def run_pipeline():
    """
    Exécute le pipeline complet de collecte, analyse
    et stockage du snapshot géopolitique.
    """
    start_time = datetime.utcnow()
    logger.info("=" * 60)
    logger.info("PIPELINE DÉMARRÉ")
    logger.info(f"Timestamp : {start_time.isoformat()}")
    logger.info("=" * 60)

    # ── Étape 0 : Test connexion MongoDB ──
    logger.info("\n[0/5] Test connexion MongoDB Atlas...")
    if not test_connection():
        logger.error("Connexion MongoDB échouée — pipeline arrêté")
        sys.exit(1)
    logger.info("Connexion MongoDB : OK")

    # ── Étape 1 : Collecte GPR ──
    logger.info("\n[1/5] Collecte GPR...")
    try:
        gpr_data = collect_gpr()
        logger.info(
            f"GPR collecté — combined={gpr_data['gpr_combined']}, "
            f"norm={gpr_data['gpr_norm']}"
        )
    except Exception as e:
        logger.error(f"Erreur collecte GPR : {e}")
        gpr_data = {"gpr_norm": 0.5, "gpr_combined": 200.0, "erreur": str(e)}

    # ── Étape 2 : Collecte actifs ──
    logger.info("\n[2/5] Collecte actifs financiers...")
    try:
        asset_data = collect_assets()
        nb_actifs  = len(asset_data.get("prices", {}))
        vix_norm   = asset_data.get("vix_norm", 0.3)
        logger.info(
            f"Actifs collectés — {nb_actifs} actifs, "
            f"VIX_norm={vix_norm}, "
            f"VIX_level={asset_data.get('summary', {}).get('vix_level')}"
        )
    except Exception as e:
        logger.error(f"Erreur collecte actifs : {e}")
        asset_data = {"prices": {}, "vix_norm": 0.3, "erreur": str(e)}
        vix_norm   = 0.3

    # ── Étape 3 : Collecte actualités ──
    logger.info("\n[3/5] Collecte actualités géopolitiques...")
    try:
        news_data   = collect_news()
        nb_articles = news_data.get("stats", {}).get("total_geo_articles", 0)
        logger.info(f"Actualités collectées — {nb_articles} articles géopolitiques")
    except Exception as e:
        logger.error(f"Erreur collecte news : {e}")
        news_data = {"articles": [], "stats": {}, "erreur": str(e)}

    # ── Étape 4 : Analyse NLP ──
    logger.info("\n[4/5] Analyse NLP du sentiment...")
    try:
        sentiment_data = analyze_sentiment(news_data)
        logger.info(
            f"NLP terminé — signal={sentiment_data['nlp_signal']}, "
            f"label={sentiment_data['label']}, "
            f"articles={sentiment_data['articles_analyzed']}"
        )
    except Exception as e:
        logger.error(f"Erreur analyse NLP : {e}")
        sentiment_data = {
            "nlp_score": 0.0, "nlp_signal": 0.0,
            "label": "neutre", "erreur": str(e)
        }

    # ── Étape 5 : Génération recommandation ──
    logger.info("\n[5/5] Génération de la recommandation...")
    try:
        gpr_norm   = gpr_data.get("gpr_norm", 0.5)
        epu_norm   = 0.3  # EPU sera collecté dans une prochaine version
        nlp_signal = sentiment_data.get("nlp_signal", 0.0)

        recommendation = generate_recommendation(
            gpr_norm=gpr_norm,
            vix_norm=vix_norm,
            epu_norm=epu_norm,
            nlp_signal=nlp_signal,
        )
        logger.info(
            f"Recommandation — niveau={recommendation['niveau_global']}, "
            f"score={recommendation['score_global']}, "
            f"confiance={recommendation['confiance']}%"
        )
    except Exception as e:
        logger.error(f"Erreur recommandation : {e}")
        recommendation = {"erreur": str(e)}

    # ── Assemblage du snapshot ──
    snapshot = {
        "collected_at":   start_time.isoformat(),
        "gpr":            gpr_data,
        "assets":         asset_data,
        "news": {
            "stats":        news_data.get("stats", {}),
            "top_articles": news_data.get("top_articles", [])[:5],
        },
        "sentiment":      sentiment_data,
        "recommendation": recommendation,
    }

    # ── Sauvegarde MongoDB ──
    logger.info("\n[+] Sauvegarde dans MongoDB Atlas...")
    try:
        doc_id = save_snapshot(snapshot)
        if doc_id:
            logger.info(f"Snapshot sauvegardé — _id={doc_id}")
        else:
            logger.error("Échec sauvegarde MongoDB")
    except Exception as e:
        logger.error(f"Erreur sauvegarde : {e}")

    # ── Résumé final ──
    duration = (datetime.utcnow() - start_time).total_seconds()
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE TERMINÉ")
    logger.info(f"Durée         : {duration:.1f} secondes")
    logger.info(f"GPR norm      : {gpr_data.get('gpr_norm', 'N/A')}")
    logger.info(f"VIX norm      : {vix_norm}")
    logger.info(f"NLP signal    : {sentiment_data.get('nlp_signal', 'N/A')}")
    logger.info(f"Niveau risque : {recommendation.get('niveau_global', 'N/A')}")
    logger.info(f"Confiance     : {recommendation.get('confiance', 'N/A')}%")
    logger.info("=" * 60)

    return snapshot


if __name__ == "__main__":
    run_pipeline()