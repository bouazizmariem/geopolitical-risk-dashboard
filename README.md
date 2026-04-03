# Geopolitical Risk-Aware Investment Dashboard

Tableau de bord intelligent d'aide à la décision d'investissement
basé sur l'analyse du risque géopolitique en temps réel.

## Stack
- Python 3.11
- MongoDB Atlas
- GitHub Actions (Self-Hosted Runner)
- Streamlit + Plotly

## Structure
- `collectors/` : collecte GPR, prix actifs, actualités
- `processors/` : NLP sentiment + moteur de recommandation
- `storage/`    : client MongoDB Atlas
- `dashboard/`  : interface Streamlit

## Lancement local
```bash
pip install -r requirements.txt
python main.py
streamlit run dashboard/app.py
```