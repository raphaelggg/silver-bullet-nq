# Silver Bullet NQ

Backtest mécanique de la stratégie ICT Silver Bullet sur le Nasdaq-100 E-mini (NQ),
barres 1 minute Databento (`GLBX.MDP3`, contrat continu `NQ.v.0`, roll au volume).

```bash
pip install -r requirements.txt
export DATABENTO_API_KEY=...
python run_backtest.py            # télécharge les données si absentes (~2,6 $ pour 2 ans), puis backtest
python -m pytest -q tests
```

Résultats écrits dans `results/` : `report.md`, `equity_r.png`, `trades_<variante>.csv`.

## Règles (configuration de base)

- Fenêtre 10:00–11:00 heure de New York (`london` 03–04 h et `pm` 14–15 h disponibles).
- Premier FVG 1 minute dont les 3 bougies sont dans la fenêtre, écart ≥ 1 pt.
- Ordre limite au bord du FVG, stop 1 tick au-delà de la bougie 1, objectif 2R, risque ≥ 5 pts.
- Exécution avant la fin de la fenêtre, sinon annulation ; annulation aussi si l'objectif est
  touché avant l'exécution. Une position max par fenêtre ; sortie forcée à 16:00.
- Coûts : 4,50 $ A/R + 1 tick de slippage sur sorties stop/temps.
- Intra-barre conservateur : stop prioritaire si stop et objectif dans la même barre.

## Filtres optionnels

- `direction` : `"both"` (défaut), `"long"` ou `"short"`.
- `bias` (biais journalier, calculé sans regarder le futur) :
  - `prev_day` : sens de la bougie RTH 09:30–16:00 de la veille ;
  - `sma20` : clôture RTH de la veille au-dessus / en dessous de sa moyenne 20 jours,
    sur une série recollée aux rolls de contrat ;
  - `midnight` : prix juste avant la fenêtre au-dessus / en dessous de l'ouverture de minuit.

  Un FVG à contre-biais est ignoré et on attend le suivant dans la fenêtre.

Tous les paramètres sont dans `silver_bullet/strategy.py` (`Params`) ; les variantes testées
sont dans `run_backtest.py` (`VARIANTS`).
