# 628-Capstone

## Run locally

```bash
pip install -r requirements.txt
streamlit run capstone.py
```

## Deployment notes
- Entry point: `capstone.py`
- Dependencies: see `requirements.txt` (must be uploaded with the app)
- Data: fetches CSSE GitHub time series; automatically falls back to bundled `data/*.csv` if remote fetch fails. No disk writes on startup.
- Metrics: recovered series is optional and flagged as incomplete if available.
