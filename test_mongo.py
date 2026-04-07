import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from storage.mongo_client import test_connection, save_snapshot, get_latest_snapshot, count_snapshots

# Test 1 — Connexion
print("=" * 50)
print("TEST 1 — Connexion MongoDB Atlas")
print("=" * 50)
ok = test_connection()
print(f"Connexion : {'OK' if ok else 'ECHEC'}")

if not ok:
    print("Vérifiez votre MONGO_URI dans le fichier .env")
    sys.exit(1)

# Test 2 — Insertion d'un snapshot de test
print("\n" + "=" * 50)
print("TEST 2 — Insertion snapshot de test")
print("=" * 50)

snapshot_test = {
    "collected_at": "2026-04-07T14:00:00",
    "gpr": {
        "gpr_combined": 245.3,
        "gpr_norm":     0.432,
    },
    "assets": {
        "prices": {
            "gold": {"price": 4655.0, "change_pct": -0.04},
            "oil":  {"price": 116.55, "change_pct": 3.68},
        },
        "vix_norm": 0.218,
    },
    "sentiment": {
        "nlp_score":  -0.524,
        "nlp_signal":  0.524,
        "label":      "escalade modérée",
    },
    "recommendation": {
        "score_geo":    0.57,
        "score_fin":    0.61,
        "score_global": 0.59,
        "niveau_global": "Élevé",
        "confiance":    95.9,
        "allocation": {
            "or":          0.50,
            "petrole":     0.25,
            "obligations": 0.25,
            "cash":        0.30,
        },
    },
    "_test": True,
}

doc_id = save_snapshot(snapshot_test)
print(f"Document inséré : {doc_id}")

# Test 3 — Récupération du dernier snapshot
print("\n" + "=" * 50)
print("TEST 3 — Récupération dernier snapshot")
print("=" * 50)

latest = get_latest_snapshot()
if latest:
    print(f"Date        : {latest.get('collected_at')}")
    print(f"Score global: {latest.get('recommendation', {}).get('score_global')}")
    print(f"Niveau      : {latest.get('recommendation', {}).get('niveau_global')}")

# Test 4 — Comptage
print("\n" + "=" * 50)
print("TEST 4 — Comptage total")
print("=" * 50)
total = count_snapshots()
print(f"Total snapshots dans MongoDB : {total}")

print("\n✓ Tous les tests MongoDB passés")