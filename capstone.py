# Data Cleaning
# 1. Fetch CSSE global COVID-19 time-series CSVs (confirmed/deaths/recovered) into pandas DataFrames.

from pathlib import Path
from typing import TYPE_CHECKING
import pandas as pd
import os

# Optional UI deps; still hint Pylance
if TYPE_CHECKING:
    import streamlit as st  # type: ignore
    import plotly.express as px  # type: ignore
else:
    try:
        import streamlit as st
        import plotly.express as px
    except ImportError:  # Streamlit/plotly not required for data pipeline usage
        st = None
        px = None

CSSE_BASE = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
)

CSSE_FILES = {
    "confirmed": CSSE_BASE + "time_series_covid19_confirmed_global.csv",
    "deaths": CSSE_BASE + "time_series_covid19_deaths_global.csv",
    "recovered": CSSE_BASE + "time_series_covid19_recovered_global.csv",
}

# Optional country name aliases to standardize downstream UI/joins.
COUNTRY_ALIASES = {
    "US": "United States",
    "Korea, South": "South Korea",
    "Taiwan*": "Taiwan",
    "Congo (Kinshasa)": "DR Congo",
    "Congo (Brazzaville)": "Republic of the Congo",
    "Burma": "Myanmar",
    "Holy See": "Vatican City",
    "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Ivory Coast",
}


def fetch_csse_timeseries():
    # Step 1: download the three global time-series CSVs (confirmed, deaths, recovered)
    return {name: pd.read_csv(url) for name, url in CSSE_FILES.items()}


def download_csse_to_disk(out_dir: str = "data"):
    # Step 1a: download and save CSVs locally
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    saved_paths = {}
    for name, url in CSSE_FILES.items():
        df = pd.read_csv(url)
        file_path = dest / f"{name}.csv"
        df.to_csv(file_path, index=False)
        saved_paths[name] = file_path
    return saved_paths


def load_timeseries(prefer_local: bool = True, data_dir: str = "data"):
    """
    Load time-series DataFrames with resilience:
    - try remote first (streamlit cloud) or local first (CLI)
    - fall back to local CSVs shipped in the repo if a remote fetch fails
    Returns (dataframes, errors)
    """
    local_paths = {name: Path(data_dir) / f"{name}.csv" for name in CSSE_FILES}
    data: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    for name, url in CSSE_FILES.items():
        def read_local():
            return pd.read_csv(local_paths[name])

        primary_first = "local" if prefer_local else "remote"

        for attempt in (primary_first, "remote" if primary_first == "local" else "local"):
            try:
                if attempt == "local":
                    data[name] = read_local()
                else:
                    data[name] = pd.read_csv(url)
                break  # success
            except Exception as exc:  # noqa: BLE001
                errors[name] = f"{attempt} load failed: {exc}"
        else:
            # Loop exhausted without break -> both failed
            continue

    return data, errors


def load_and_clean_timeseries(source):
    # Step 2: read one time-series table and drop Lat/Long, keep region + dates
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    # Keep identifiers plus all date columns; Lat/Long are removed.
    return df.drop(columns=["Lat", "Long"], errors="ignore")


# Step 3: harmonize country naming using a small alias dictionary.
def normalize_country_names(df: pd.DataFrame, aliases: dict | None = None) -> pd.DataFrame:
    # Step 3: replace known aliases in Country/Region; no-op if aliases is empty
    if not aliases:
        return df
    out = df.copy()
    out["Country/Region"] = out["Country/Region"].replace(aliases)
    return out


def aggregate_by_country(df: pd.DataFrame) -> pd.DataFrame:
    # Step 4: group by Country/Region and sum all date columns (national totals)
    date_cols = [c for c in df.columns if c not in ("Province/State", "Country/Region")]
    grouped = (
        df.groupby("Country/Region", as_index=False)[date_cols]
        .sum(min_count=1)  # keep NaN if all missing
    )
    return grouped


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    # Step 5: melt wide date columns to long format: Country/Region, date, cumulative
    id_vars = [c for c in ("Province/State", "Country/Region") if c in df.columns]
    date_cols = [c for c in df.columns if c not in id_vars]
    long_df = df.melt(id_vars=id_vars, value_vars=date_cols, var_name="date", value_name="cumulative")
    long_df["date"] = pd.to_datetime(long_df["date"])
    return long_df


def add_metric_column(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    # Step 6: tag long-format rows with metric type (confirmed/deaths/recovered)
    out = df.copy()
    out["metric"] = metric_name
    return out


def compute_daily_from_cumulative(df_long: pd.DataFrame) -> pd.DataFrame:
    # Step 7: compute daily increments per country from cumulative series
    if "cumulative" not in df_long.columns or "Country/Region" not in df_long.columns:
        raise ValueError("Expected columns: Country/Region and cumulative")
    out = df_long.sort_values(["Country/Region", "date"]).copy()
    out["daily"] = (
        out.groupby(["Country/Region", "metric"], dropna=False)["cumulative"]
        .diff()
        .clip(lower=0)
    )
    return out


def add_rolling_mean(df_long: pd.DataFrame, col: str = "daily", window: int = 7) -> pd.DataFrame:
    # Step 8: optional display smoothing (rolling mean) while keeping raw values
    """
    Compute rolling mean for a given value column (default daily) for display only.
    Keeps original values for download/analysis.
    """
    out = df_long.sort_values(["Country/Region", "date"]).copy()
    out[f"{col}_{window}d_avg"] = (
        out.groupby(["Country/Region", "metric"], dropna=False)[col]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    return out


def flag_recovered_incomplete(df_long: pd.DataFrame) -> pd.DataFrame:
    # Step 9: flag recovered series as incomplete for front-end warning
    """
    Flag recovered series as potentially incomplete for front-end warning.
    Adds boolean column 'recovered_incomplete' (True for metric == recovered).
    """
    out = df_long.copy()
    out["recovered_incomplete"] = out["metric"] == "recovered"
    return out


def concat_metrics(long_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Step 10: concat confirmed/deaths/recovered long tables into one
    required_cols = {"Country/Region", "date", "metric", "cumulative", "daily"}
    missing = [name for name, df in long_dfs.items() if not required_cols.issubset(df.columns)]
    if missing:
        raise ValueError(f"Missing required columns in: {missing}")
    return pd.concat(long_dfs.values(), ignore_index=True)


def prepare_for_downstream(
    df_long: pd.DataFrame, cache_path: str | None = None
) -> tuple[pd.DataFrame, dict]:
    # Step 11: sort/index, optional cache, and produce simple metadata for UI defaults
    sorted_df = df_long.sort_values(["Country/Region", "metric", "date"]).reset_index(drop=True)
    meta = {
        "countries": sorted_df["Country/Region"].dropna().drop_duplicates().sort_values().tolist(),
        "date_min": sorted_df["date"].min(),
        "date_max": sorted_df["date"].max(),
    }
    if cache_path:
        cache_suffix = cache_path.split(".")[-1].lower()
        if cache_suffix in ("parquet", "pq"):
            sorted_df.to_parquet(cache_path, index=False)
        elif cache_suffix in ("pkl", "pickle"):
            sorted_df.to_pickle(cache_path)
        else:
            sorted_df.to_csv(cache_path, index=False)
    return sorted_df, meta


def build_dataset():
    # Run full pipeline and return sorted long table plus metadata
    # Streamlit Cloud 不依赖本地 CSV，优先从远程拉取
    prefer_local = os.environ.get("STREAMLIT_SERVER_RUNNING") is None
    raw_tables, load_errors = load_timeseries(prefer_local=prefer_local)

    if not raw_tables:
        raise RuntimeError("No COVID-19 tables could be loaded from remote or local sources.")

    cleaned = {}
    for name, df in raw_tables.items():
        try:
            cleaned[name] = normalize_country_names(load_and_clean_timeseries(df), COUNTRY_ALIASES)
        except Exception as exc:  # noqa: BLE001
            load_errors[name] = load_errors.get(name, "") + f"; clean failed: {exc}"

    if not cleaned:
        raise RuntimeError("Failed to clean any COVID-19 tables.")

    aggregated = {name: aggregate_by_country(df) for name, df in cleaned.items()}

    # Wide -> long + metric tag + daily + rolling mean + recovered flag
    long_tables = {}
    for name, agg_df in aggregated.items():
        try:
            long_df = wide_to_long(agg_df)
            long_df = add_metric_column(long_df, name)
            long_df = compute_daily_from_cumulative(long_df)
            long_tables[name] = long_df
        except Exception as exc:  # noqa: BLE001
            load_errors[name] = load_errors.get(name, "") + f"; long/compute failed: {exc}"

    if not long_tables:
        raise RuntimeError("No metrics available after processing.")

    merged = concat_metrics(long_tables)
    merged = add_rolling_mean(merged, col="daily", window=7)
    merged = flag_recovered_incomplete(merged)
    sorted_df, meta = prepare_for_downstream(merged)
    meta["available_metrics"] = sorted(long_tables.keys())
    meta["warnings"] = [f"{k}: {v}" for k, v in load_errors.items() if v]
    return sorted_df, meta


def run_streamlit_app():
    if st is None:
        raise ImportError("streamlit is not installed in this environment.")

    @st.cache_data(show_spinner=False)
    def cached_dataset():
        return build_dataset()

    df, meta = cached_dataset()

    if meta.get("warnings"):
        for msg in meta["warnings"]:
            st.warning(f"Data load warning – {msg}")

    st.title("Global COVID-19 Time Series (CSSE)")
    st.caption("Data source: Johns Hopkins CSSE time_series* (auto-fetched).")

    # Controls
    countries = st.multiselect(
        "Countries", meta["countries"], default=meta["countries"][:5]
    )
    available_metrics = meta.get("available_metrics", ["confirmed", "deaths", "recovered"])
    default_metrics = [m for m in ("confirmed", "deaths", "recovered") if m in available_metrics]
    metrics = st.multiselect(
        "Metrics", available_metrics, default=default_metrics
    )
    date_min = meta["date_min"].to_pydatetime() if hasattr(meta["date_min"], "to_pydatetime") else meta["date_min"]
    date_max = meta["date_max"].to_pydatetime() if hasattr(meta["date_max"], "to_pydatetime") else meta["date_max"]
    date_range = st.slider(
        "Date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    use_rolling = st.checkbox("Show 7-day average", value=True)
    recovered_available = "recovered" in available_metrics
    show_recovered = recovered_available and st.checkbox(
        "Include recovered (data incomplete)", value=True
    )

    if not recovered_available:
        st.info("Recovered series not available (source often incomplete).")

    filtered = df[
        (df["Country/Region"].isin(countries))
        & (df["metric"].isin(metrics if show_recovered else [m for m in metrics if m != "recovered"]))
        & (df["date"].between(date_range[0], date_range[1]))
    ].copy()

    value_col = "daily_7d_avg" if use_rolling else "daily"
    y_label = "Daily (7d avg)" if use_rolling else "Daily"

    if filtered.empty:
        st.info("No data for current selection.")
    elif px:
        for metric in sorted(filtered["metric"].unique()):
            sub = filtered[filtered["metric"] == metric]
            fig = px.line(
                sub,
                x="date",
                y=value_col,
                color="Country/Region",
                hover_data=["cumulative"],
                labels={"date": "Date", value_col: y_label},
                title=f"{metric.title()}",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("plotly is not installed; install plotly to see charts.")

    # Data table preview (all selected metrics)
    st.subheader("Data preview")
    preview_cols = ["Country/Region", "metric", "date", "cumulative", "daily", "daily_7d_avg"]
    st.dataframe(filtered[preview_cols].head(200))


def run_cli_demo():
    saved = download_csse_to_disk()
    for key, path in saved.items():
        print(f"{key} saved to {path}")

    # Load and clean all three CSVs, then report their shapes.
    cleaned = {
        name: normalize_country_names(load_and_clean_timeseries(path), COUNTRY_ALIASES)
        for name, path in saved.items()
    }
    for name, df in cleaned.items():
        print(f"{name} shape after cleaning: {df.shape}")

    # Peek at the first few standardized country names for sanity.
    for name, df in cleaned.items():
        sample_countries = df["Country/Region"].drop_duplicates().sort_values().head(10).tolist()
        print(f"{name} sample countries: {sample_countries}")

    # Aggregate to country-level totals.
    aggregated = {name: aggregate_by_country(df) for name, df in cleaned.items()}
    for name, df in aggregated.items():
        print(f"{name} shape after country aggregation: {df.shape}")


if __name__ == "__main__":
    # If streamlit is available, launch the app; otherwise fall back to CLI demo.
    if st is not None:
        run_streamlit_app()
    else:
        run_cli_demo()
