import logging
from collectors.gpr import collect_gpr
from collectors.assets import collect_assets
from collectors.news import collect_news
from processors.nlp import analyze_sentiment
from processors.recommender import generate_recommendation
from storage.mongo_client import save_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_pipeline():
    logging.info("=== Pipeline démarré ===")

    gpr_data        = collect_gpr()
    asset_data      = collect_assets()
    news_data       = collect_news()

    sentiment_score = analyze_sentiment(news_data)
    recommendation  = generate_recommendation(gpr_data, sentiment_score, asset_data)

    snapshot = {
        "gpr":            gpr_data,
        "assets":         asset_data,
        "news":           news_data,
        "sentiment":      sentiment_score,
        "recommendation": recommendation,
    }

    save_snapshot(snapshot)
    logging.info("=== Pipeline terminé ===")

if __name__ == "__main__":
    run_pipeline()