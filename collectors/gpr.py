# collectors/gpr.py
import pandas as pd
import numpy as np
import feedparser
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"

GPR_KEYWORDS = [
    "war", "conflict", "military", "attack", "strike",
    "invasion", "missile", "bomb", "troops", "weapon",
    "sanctions", "nuclear", "terror", "coup", "crisis",
    "geopolitical", "NATO", "escalation", "ceasefire",
    "hostilities", "siege", "blockade", "airstrike",
]

RSS_FEEDS_PROXY = [
    "https://feeds.reuters.com/reuters/worldNews",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
]

def collect_gpr():
    logger.info("Collecte GPR démarrée")
    gpr_official = _fetch_gpr_official()
    gpr_proxy    = _compute_gpr_proxy()

    if gpr_official and gpr_proxy:
        gpr_combined = 0.70 * gpr_official["value"] + \
                       0.30 * gpr_proxy["scaled_value"]
    elif gpr_official:
        gpr_combined = gpr_official["value"]
    else:
        gpr_combined = 100.0

    GPR_MIN, GPR_MAX = 39.0, 512.0
    gpr_norm = float(np.clip(
        (gpr_combined - GPR_MIN) / (GPR_MAX - GPR_MIN), 0, 1
    ))

    result = {
        "gpr_official": gpr_official,
        "gpr_proxy":    gpr_proxy,
        "gpr_combined": round(gpr_combined, 2),
        "gpr_norm":     round(gpr_norm, 4),
        "collected_at": datetime.utcnow().isoformat(),
    }

    logger.info(f"GPR collecté — combined={gpr_combined:.1f}, norm={gpr_norm:.4f}")
    return result


def _fetch_gpr_official():
    try:
        gpr_raw  = pd.read_excel(GPR_URL, engine="xlrd")
        date_col = gpr_raw.columns[0]

        dates = []
        for v in gpr_raw[date_col]:
            try:
                s = str(v).strip().upper().replace("M", "-")
                dates.append(pd.to_datetime(s, errors="coerce"))
            except:
                dates.append(pd.NaT)

        gpr_raw.index = pd.DatetimeIndex(dates)
        gpr_raw = gpr_raw[gpr_raw.index.notna()]
        gpr_raw.index = gpr_raw.index.map(lambda x: x.replace(day=1))

        gpr_series = pd.to_numeric(
            gpr_raw["GPR"], errors="coerce"
        ).dropna()

        last_date  = gpr_series.index[-1]
        last_value = float(gpr_series.iloc[-1])
        prev_value = float(gpr_series.iloc[-2]) \
            if len(gpr_series) >= 2 else last_value
        change_pct = ((last_value - prev_value) / prev_value) * 100

        logger.info(f"GPR officiel : {last_value:.1f} ({last_date.strftime('%Y-%m')})")

        return {
            "value":      round(last_value, 2),
            "month":      last_date.strftime("%Y-%m"),
            "change_pct": round(change_pct, 2),
            "prev_value": round(prev_value, 2),
            "source":     "Caldara & Iacoviello (Fed)",
        }

    except Exception as e:
        logger.error(f"Erreur GPR officiel : {e}")
        return None


def _compute_gpr_proxy():
    now        = datetime.utcnow()
    cutoff_48h = now - timedelta(hours=48)

    total_articles = 0
    geo_articles   = 0
    articles_detail = []

    for feed_url in RSS_FEEDS_PROXY:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                pub_date = None
                if hasattr(entry, "published_parsed") \
                        and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])

                if pub_date and pub_date < cutoff_48h:
                    continue

                total_articles += 1
                title   = getattr(entry, "title",   "").lower()
                summary = getattr(entry, "summary", "").lower()
                text    = title + " " + summary

                kw_found = [kw for kw in GPR_KEYWORDS if kw.lower() in text]
                if kw_found:
                    geo_articles += 1
                    articles_detail.append({
                        "title":    getattr(entry, "title", ""),
                        "keywords": kw_found[:3],
                        "date":     pub_date.isoformat() if pub_date else None,
                    })

        except Exception as e:
            logger.warning(f"Erreur flux RSS {feed_url} : {e}")

    if total_articles == 0:
        logger.warning("Aucun article RSS collecté pour GPR proxy")
        return None

    geo_ratio = geo_articles / total_articles
    GPR_MIN, GPR_MAX = 39.0, 512.0
    scaled = GPR_MIN + geo_ratio * (GPR_MAX - GPR_MIN)

    logger.info(
        f"GPR proxy : {geo_articles}/{total_articles} articles "
        f"→ ratio={geo_ratio:.3f} → scaled={scaled:.1f}"
    )

    return {
        "geo_articles":   geo_articles,
        "total_articles": total_articles,
        "geo_ratio":      round(geo_ratio, 4),
        "scaled_value":   round(scaled, 2),
        "window_hours":   48,
        "top_articles":   articles_detail[:5],
        "computed_at":    now.isoformat(),
    }