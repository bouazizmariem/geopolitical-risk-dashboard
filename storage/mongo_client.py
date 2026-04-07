# storage/mongo_client.py
# Client MongoDB Atlas — insertion et requêtes des snapshots
# Référence : pymongo documentation

import os
import certifi
import logging
from datetime import datetime, timedelta
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)

# ── Configuration ──
MONGO_URI     = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "geopolitical_risk_db")

# Collections MongoDB
COL_SNAPSHOTS = "snapshots"
COL_ASSETS    = "assets_history"
COL_NEWS      = "news_history"


def get_client():
    """Crée et retourne un client MongoDB Atlas."""
    if not MONGO_URI:
        raise ValueError(
            "MONGO_URI non défini. "
            "Vérifiez vos variables d'environnement ou GitHub Secrets."
        )
    client = MongoClient(
        MONGO_URI,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=10000,
    )
    return client


def test_connection():
    """Teste la connexion MongoDB Atlas."""
    try:
        client = get_client()
        client.admin.command("ping")
        logger.info("Connexion MongoDB Atlas : OK")
        client.close()
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Connexion MongoDB Atlas échouée : {e}")
        return False
    except Exception as e:
        logger.error(f"Erreur inattendue MongoDB : {e}")
        return False


def save_snapshot(snapshot):
    """
    Insère un snapshot complet dans MongoDB Atlas.

    Args:
        snapshot : dict contenant gpr, assets, news,
                   sentiment, recommendation

    Returns:
        str : _id du document inséré, ou None si erreur
    """
    try:
        client = get_client()
        db     = client[MONGO_DB_NAME]
        col    = db[COL_SNAPSHOTS]

        # Ajouter horodatage si absent
        if "collected_at" not in snapshot:
            snapshot["collected_at"] = datetime.utcnow().isoformat()

        # Ajouter métadonnées
        snapshot["_meta"] = {
            "version":    "1.0",
            "pipeline":   "geopolitical-risk-dashboard",
            "inserted_at": datetime.utcnow().isoformat(),
        }

        result = col.insert_one(snapshot)
        doc_id = str(result.inserted_id)

        logger.info(f"Snapshot inséré — _id={doc_id}")
        client.close()
        return doc_id

    except Exception as e:
        logger.error(f"Erreur insertion snapshot : {e}")
        return None


def get_latest_snapshot():
    """
    Récupère le snapshot le plus récent.

    Returns:
        dict ou None
    """
    try:
        client = get_client()
        db     = client[MONGO_DB_NAME]
        col    = db[COL_SNAPSHOTS]

        doc = col.find_one(
            {},
            sort=[("collected_at", DESCENDING)]
        )

        client.close()

        if doc:
            doc["_id"] = str(doc["_id"])
            logger.info(
                f"Dernier snapshot : {doc.get('collected_at')}"
            )
        return doc

    except Exception as e:
        logger.error(f"Erreur récupération snapshot : {e}")
        return None


def get_snapshots_history(days=30):
    """
    Récupère l'historique des snapshots sur N jours.

    Args:
        days : nombre de jours d'historique

    Returns:
        list de dicts
    """
    try:
        client   = get_client()
        db       = client[MONGO_DB_NAME]
        col      = db[COL_SNAPSHOTS]
        cutoff   = (
            datetime.utcnow() - timedelta(days=days)
        ).isoformat()

        cursor = col.find(
            {"collected_at": {"$gte": cutoff}},
            sort=[("collected_at", DESCENDING)]
        )

        docs = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            docs.append(doc)

        client.close()
        logger.info(
            f"Historique récupéré : {len(docs)} snapshots "
            f"sur {days} jours"
        )
        return docs

    except Exception as e:
        logger.error(f"Erreur récupération historique : {e}")
        return []


def get_gpr_history(days=90):
    """
    Récupère l'historique du score GPR pour les graphiques.

    Returns:
        list de dicts {date, gpr_norm, gpr_combined}
    """
    try:
        client = get_client()
        db     = client[MONGO_DB_NAME]
        col    = db[COL_SNAPSHOTS]
        cutoff = (
            datetime.utcnow() - timedelta(days=days)
        ).isoformat()

        cursor = col.find(
            {"collected_at": {"$gte": cutoff}},
            {
                "collected_at":        1,
                "gpr.gpr_norm":        1,
                "gpr.gpr_combined":    1,
                "recommendation.score_global": 1,
            },
            sort=[("collected_at", DESCENDING)]
        )

        history = []
        for doc in cursor:
            history.append({
                "date":         doc.get("collected_at"),
                "gpr_norm":     doc.get("gpr", {}).get("gpr_norm"),
                "gpr_combined": doc.get("gpr", {}).get("gpr_combined"),
                "score_global": doc.get("recommendation", {})
                                   .get("score_global"),
            })

        client.close()
        return history

    except Exception as e:
        logger.error(f"Erreur historique GPR : {e}")
        return []


def get_assets_history(asset_name, days=30):
    """
    Récupère l'historique des prix d'un actif.

    Args:
        asset_name : nom de l'actif (ex: 'gold', 'oil')
        days       : nombre de jours

    Returns:
        list de dicts {date, price, change_pct}
    """
    try:
        client = get_client()
        db     = client[MONGO_DB_NAME]
        col    = db[COL_SNAPSHOTS]
        cutoff = (
            datetime.utcnow() - timedelta(days=days)
        ).isoformat()

        field  = f"assets.prices.{asset_name}"
        cursor = col.find(
            {"collected_at": {"$gte": cutoff}},
            {
                "collected_at": 1,
                field:          1,
            },
            sort=[("collected_at", DESCENDING)]
        )

        history = []
        for doc in cursor:
            asset_data = (
                doc.get("assets", {})
                   .get("prices", {})
                   .get(asset_name, {})
            )
            if asset_data:
                history.append({
                    "date":       doc.get("collected_at"),
                    "price":      asset_data.get("price"),
                    "change_pct": asset_data.get("change_pct"),
                })

        client.close()
        return history

    except Exception as e:
        logger.error(f"Erreur historique {asset_name} : {e}")
        return []


def count_snapshots():
    """Retourne le nombre total de snapshots dans la base."""
    try:
        client = get_client()
        db     = client[MONGO_DB_NAME]
        col    = db[COL_SNAPSHOTS]
        count  = col.count_documents({})
        client.close()
        return count
    except Exception as e:
        logger.error(f"Erreur count : {e}")
        return 0