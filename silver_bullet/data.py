"""Chargement des barres NQ 1 minute (Databento GLBX.MDP3, contrat continu NQ.v.0)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_FILE = DATA_DIR / "nq_ohlcv1m_raw.dbn.zst"
NY = "America/New_York"


def download(start: str, end: str, path: Path = RAW_FILE) -> Path:
    """Télécharge les barres 1m NQ (roll au volume) depuis Databento. Payant : vérifier le coût avant."""
    import databento as db

    client = db.Historical()  # lit DATABENTO_API_KEY
    store = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["NQ.v.0"],
        stype_in="continuous",
        schema="ohlcv-1m",
        start=start,
        end=end,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    store.to_file(path)
    return path


def load_bars(path: Path = RAW_FILE) -> pd.DataFrame:
    """Barres 1m indexées en heure de New York, avec l'identifiant du contrat sous-jacent.

    Databento horodate chaque barre à son ouverture (ts_event) ; on garde cette convention.
    """
    import databento as db

    df = db.DBNStore.from_file(path).to_df()
    df = df[["open", "high", "low", "close", "volume", "instrument_id"]].copy()
    df.index = df.index.tz_convert(NY)
    df.index.name = "ts"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df
