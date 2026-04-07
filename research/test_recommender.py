import sys
sys.path.append(".")
from processors.recommender import generate_recommendation

# Test avec des valeurs simulant une crise modérée
result = generate_recommendation(
    gpr_norm=0.65,
    vix_norm=0.45,
    epu_norm=0.50,
    nlp_signal=0.40
)

print("=" * 55)
print("TEST DU MOTEUR DE RECOMMANDATION")
print("=" * 55)
print(f"Score géopolitique : {result['score_geo']}")
print(f"Score financier    : {result['score_fin']}")
print(f"Score global       : {result['score_global']}")
print(f"Niveau global      : {result['niveau_global']}")
print(f"Confiance          : {result['confiance']}%")
print(f"Divergence         : {result['alerte_divergence']}")
print(f"\nAllocation :")
for actif, poids in result['allocation'].items():
    print(f"   {actif:15} : {poids*100:.0f}%")
print(f"\nJustification :")
print(f"   {result['justification']}")
