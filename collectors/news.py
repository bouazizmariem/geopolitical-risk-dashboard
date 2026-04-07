# collectors/news.py
# Collecte des actualités géopolitiques via flux RSS
# et filtrage par mots-clés géopolitiques

import feedparser
import pandas as pd
from datetime import datetime, timedelta
import logging
import hashlib

logger = logging.getLogger(__name__)

# Flux RSS surveillés
RSS_FEEDS = [
    {
        "url":    "https://feeds.reuters.com/reuters/worldNews",
        "source": "Reuters World",
        "langue": "en",
    },
    {
        "url":    "https://feeds.bbci.co.uk/news/world/rss.xml",
        "source": "BBC World",
        "langue": "en",
    },
    {
        "url":    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "source": "NYT World",
        "langue": "en",
    },
    {
        "url":    "https://feeds.reuters.com/reuters/businessNews",
        "source": "Reuters Business",
        "langue": "en",
    },
]

# Mots-clés géopolitiques — tier 1 (très fort signal)
KEYWORDS_T1 = [
    "war", "invasion", "airstrike", "missile", "nuclear",
    "bomb", "troops", "military strike", "attack", "killed",
    "ceasefire", "armistice",
]

# Mots-clés géopolitiques — tier 2 (signal modéré)
KEYWORDS_T2 = [
    "conflict", "sanctions", "blockade", "escalation",
    "tension", "crisis", "threat", "military", "troops",
    "NATO", "geopolitical", "coup", "protest", "siege",
    "embargo", "terror", "hostilities",
]

# Mots-clés géopolitiques — tier 3 (signal faible)
KEYWORDS_T3 = [
    "dispute", "warning", "concern", "security",
    "defense", "alliance", "diplomat", "negotiate",
    "pressure", "standoff", "confrontation",
]


def collect_news():
    """
    Collecte les actualités géopolitiques des 48 dernières heures
    depuis les flux RSS configurés.

    Returns:
        dict avec articles (liste), stats, et top_articles
    """
    logger.info("Collecte des actualités démarrée")

    now        = datetime.utcnow()
    cutoff_48h = now - timedelta(hours=48)
    cutoff_24h = now - timedelta(hours=24)

    all_articles = []
    seen_hashes  = set()

    for feed_config in RSS_FEEDS:
        articles = _fetch_feed(
            feed_config, cutoff_48h, seen_hashes
        )
        all_articles.extend(articles)
        logger.info(
            f"{feed_config['source']:20} : "
            f"{len(articles)} articles géopolitiques"
        )

    # Trier par date décroissante
    all_articles.sort(
        key=lambda x: x.get("published_at") or "",
        reverse=True
    )

    # Statistiques
    articles_24h = [
        a for a in all_articles
        if a.get("published_at") and
        a["published_at"] >= cutoff_24h.isoformat()
    ]

    stats = {
        "total_geo_articles": len(all_articles),
        "articles_last_24h":  len(articles_24h),
        "articles_last_48h":  len(all_articles),
        "t1_count": sum(1 for a in all_articles if a["tier"] == 1),
        "t2_count": sum(1 for a in all_articles if a["tier"] == 2),
        "t3_count": sum(1 for a in all_articles if a["tier"] == 3),
        "sources":  list({a["source"] for a in all_articles}),
    }

    result = {
        "articles":     all_articles,
        "top_articles": all_articles[:10],
        "stats":        stats,
        "collected_at": now.isoformat(),
    }

    logger.info(
        f"Actualités collectées : "
        f"{len(all_articles)} articles géopolitiques "
        f"({stats['t1_count']} T1 / "
        f"{stats['t2_count']} T2 / "
        f"{stats['t3_count']} T3)"
    )

    return result


def _fetch_feed(feed_config, cutoff, seen_hashes):
    """Collecte et filtre les articles d'un flux RSS."""
    articles = []

    try:
        feed = feedparser.parse(feed_config["url"])

        for entry in feed.entries:
            # Parser la date
            pub_date = _parse_date(entry)

            # Ignorer les articles trop anciens
            if pub_date and pub_date < cutoff:
                continue

            title   = getattr(entry, "title",   "")
            summary = getattr(entry, "summary", "")
            link    = getattr(entry, "link",    "")
            text    = (title + " " + summary).lower()

            # Déduplication par hash du titre
            title_hash = hashlib.md5(
                title.encode("utf-8")
            ).hexdigest()
            if title_hash in seen_hashes:
                continue

            # Classifier par tier
            tier, keywords_found = _classify_article(text)
            if tier == 0:
                continue

            seen_hashes.add(title_hash)

            articles.append({
                "title":        title,
                "summary":      summary[:300] if summary else "",
                "link":         link,
                "source":       feed_config["source"],
                "published_at": pub_date.isoformat() if pub_date else None,
                "tier":         tier,
                "keywords":     keywords_found[:5],
                "hash":         title_hash,
            })

    except Exception as e:
        logger.warning(
            f"Erreur flux {feed_config['source']} : {e}"
        )

    return articles


def _parse_date(entry):
    """Parse la date de publication d'un article RSS."""
    try:
        if hasattr(entry, "published_parsed") \
                and entry.published_parsed:
            return datetime(*entry.published_parsed[:6])
    except:
        pass
    try:
        if hasattr(entry, "updated_parsed") \
                and entry.updated_parsed:
            return datetime(*entry.updated_parsed[:6])
    except:
        pass
    return None


def _classify_article(text):
    """
    Classe un article selon son niveau géopolitique.

    Returns:
        (tier, keywords_found)
        tier=0 : non géopolitique
        tier=1 : signal fort
        tier=2 : signal modéré
        tier=3 : signal faible
    """
    t1_found = [kw for kw in KEYWORDS_T1 if kw in text]
    if t1_found:
        return 1, t1_found

    t2_found = [kw for kw in KEYWORDS_T2 if kw in text]
    if t2_found:
        return 2, t2_found

    t3_found = [kw for kw in KEYWORDS_T3 if kw in text]
    if t3_found:
        return 3, t3_found

    return 0, []