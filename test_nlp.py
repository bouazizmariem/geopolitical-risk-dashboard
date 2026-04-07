import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collectors.news import collect_news
from processors.nlp import analyze_sentiment
import json

print("Collecte des actualités...")
news = collect_news()
print(f"Articles collectés : {news['stats']['total_geo_articles']}")

print("\nAnalyse NLP...")
result = analyze_sentiment(news)

print(f"\nScore NLP    : {result['nlp_score']}")
print(f"Signal NLP   : {result['nlp_signal']}")
print(f"Label        : {result['label']}")
print(f"Articles     : {result['articles_analyzed']}")
print(f"\nStats par tier :")
print(json.dumps(result['tier_stats'], indent=2))
print(f"\nTop 3 articles les plus alarmistes :")
for a in result['top_articles'][:3]:
    print(f"  [{a['tier']}] compound={a['compound']:+.3f} — {a['title']}")