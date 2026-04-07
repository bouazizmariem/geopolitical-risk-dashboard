from collectors.news import collect_news
import json

result = collect_news()

print("Stats:")
print(json.dumps(result['stats'], indent=2))

print("\nTop 3 articles:")
for a in result['top_articles'][:3]:
    print(f"  [Tier {a['tier']}] {a['title']}")
    print(f"  Source   : {a['source']}")
    print(f"  Mots-clés: {a['keywords']}")
    print()