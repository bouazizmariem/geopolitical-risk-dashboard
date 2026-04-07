# collectors/assets.py
# Collecte des prix en temps réel des actifs surveillés
# via yfinance + calcul du VIX normalisé

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Actifs surveillés avec leurs métadonnées
ACTIFS = {
    "gold":    {"ticker": "GC=F",     "nom": "Or",              "groupe": "geo"},
    "oil":     {"ticker": "CL=F",     "nom": "Pétrole brut",    "groupe": "geo"},
    "silver":  {"ticker": "SI=F",     "nom": "Argent métal",    "groupe": "geo"},
    "defense": {"ticker": "ITA",      "nom": "ETF Défense",     "groupe": "geo"},
    "bonds":   {"ticker": "^TNX",     "nom": "Obligations 10Y", "groupe": "fin"},
    "dollar":  {"ticker": "DX-Y.NYB", "nom": "Dollar Index",    "groupe": "fin"},
    "btc":     {"ticker": "BTC-USD",  "nom": "Bitcoin",         "groupe": "fin"},
    "vix":     {"ticker": "^VIX",     "nom": "VIX",             "groupe": "indicateur"},
}

# Échelle historique VIX pour normalisation
VIX_MIN, VIX_MAX = 9.0, 90.0


def collect_assets():
    """
    Collecte les prix en temps réel de tous les actifs surveillés.

    Returns:
        dict avec prices (détail par actif), vix_norm,
        et summary (résumé par groupe)
    """
    logger.info("Collecte des actifs démarrée")

    prices  = {}
    erreurs = []

    for nom, meta in ACTIFS.items():
        try:
            data = _fetch_asset(nom, meta)
            if data:
                prices[nom] = data
                logger.info(
                    f"{nom:10} : {data['price']:.2f} "
                    f"({data['change_pct']:+.2f}%)"
                )
        except Exception as e:
            logger.error(f"Erreur {nom} : {e}")
            erreurs.append(nom)

    # Extraire et normaliser le VIX
    vix_norm = None
    if "vix" in prices:
        vix_val  = prices["vix"]["price"]
        vix_norm = float(np.clip(
            (vix_val - VIX_MIN) / (VIX_MAX - VIX_MIN), 0, 1
        ))
        prices["vix"]["vix_norm"] = round(vix_norm, 4)

    # Résumé par groupe
    summary = _build_summary(prices)

    result = {
        "prices":       prices,
        "vix_norm":     round(vix_norm, 4) if vix_norm else None,
        "summary":      summary,
        "erreurs":      erreurs,
        "collected_at": datetime.utcnow().isoformat(),
    }

    logger.info(
        f"Actifs collectés : {len(prices)}/{len(ACTIFS)} "
        f"({'OK' if not erreurs else 'erreurs: ' + str(erreurs)})"
    )
    return result


def _fetch_asset(nom, meta):
    """Télécharge les données d'un actif via yfinance."""
    ticker = meta["ticker"]

    # Télécharger les 5 derniers jours pour avoir
    # assez de données même si marché fermé
    data = yf.download(
        ticker,
        period="5d",
        interval="1d",
        progress=False,
        auto_adjust=True
    )

    if data.empty:
        logger.warning(f"{nom} ({ticker}) : aucune donnée")
        return None

    # Aplatir les colonnes si MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Prix de clôture le plus récent
    close   = data["Close"].dropna()
    if close.empty:
        return None

    price_now  = float(close.iloc[-1])
    price_prev = float(close.iloc[-2]) if len(close) >= 2 else price_now

    # Variation journalière
    change_pct = ((price_now - price_prev) / price_prev) * 100 \
        if price_prev != 0 else 0.0

    # Volume (si disponible)
    volume = None
    if "Volume" in data.columns:
        vol_series = data["Volume"].dropna()
        if not vol_series.empty:
            volume = int(vol_series.iloc[-1])

    # Variation sur 1 mois (si assez de données)
    change_1m = None
    try:
        data_1m = yf.download(
            ticker, period="35d", interval="1d",
            progress=False, auto_adjust=True
        )
        if isinstance(data_1m.columns, pd.MultiIndex):
            data_1m.columns = data_1m.columns.get_level_values(0)
        close_1m = data_1m["Close"].dropna()
        if len(close_1m) >= 20:
            price_1m   = float(close_1m.iloc[-21])
            change_1m  = ((price_now - price_1m) / price_1m) * 100
    except:
        pass

    return {
        "ticker":      ticker,
        "nom":         meta["nom"],
        "groupe":      meta["groupe"],
        "price":       round(price_now, 4),
        "price_prev":  round(price_prev, 4),
        "change_pct":  round(change_pct, 4),
        "change_1m":   round(change_1m, 2) if change_1m else None,
        "volume":      volume,
        "currency":    _get_currency(nom),
        "last_update": datetime.utcnow().isoformat(),
    }


def _get_currency(nom):
    currencies = {
        "gold":    "USD/oz",
        "oil":     "USD/baril",
        "silver":  "USD/oz",
        "defense": "USD",
        "bonds":   "%",
        "dollar":  "index",
        "btc":     "USD",
        "vix":     "index",
    }
    return currencies.get(nom, "USD")


def _build_summary(prices):
    """Construit un résumé des performances par groupe."""
    geo_assets = [
        nom for nom, p in prices.items()
        if p.get("groupe") == "geo"
    ]
    fin_assets = [
        nom for nom, p in prices.items()
        if p.get("groupe") == "fin"
    ]

    def avg_change(asset_list):
        changes = [
            prices[a]["change_pct"]
            for a in asset_list
            if a in prices and prices[a]["change_pct"] is not None
        ]
        return round(np.mean(changes), 4) if changes else None

    return {
        "geo_avg_change": avg_change(geo_assets),
        "fin_avg_change": avg_change(fin_assets),
        "geo_assets":     geo_assets,
        "fin_assets":     fin_assets,
        "vix_level":      _interpret_vix(prices),
    }


def _interpret_vix(prices):
    """Interprète le niveau du VIX."""
    if "vix" not in prices:
        return "inconnu"
    vix = prices["vix"]["price"]
    if vix < 15:
        return "calme"
    elif vix < 20:
        return "normal"
    elif vix < 30:
        return "stress"
    elif vix < 40:
        return "panique"
    else:
        return "crise extrême"