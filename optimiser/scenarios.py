import logging

import pytz
import numpy as np
import pandas as pd
from scipy.stats import norm

from config import settings

logger = logging.getLogger(__name__)

N_SCENARIOS = 20

# ---------------------------------------------------------------------------
# Historical correlation estimation (computed once, cached)
# ---------------------------------------------------------------------------



def _estimate_correlation_matrix(cutoff_date: str = "2025-01-01", use_spread: bool = False) -> np.ndarray:
    """
    Estimate Gaussian copula correlation matrix from historical price data up to cutoff_date.

    Spearman rank correlation is computed then converted via rho_g = 2*sin(pi*rho_s/6),
    which is the exact mapping for a Gaussian copula. cutoff_date prevents leakage of
    future dependence structure into the backtest.
    """


    logger.info("Estimating historical price correlation matrix...")

    try:
        # Load historical actuals
        da_file = settings.data_dir / "processed" / "day_ahead_processed.parquet"
        bm_file = settings.data_dir / "processed" / "bmrs_processed.parquet"
        dc_file = settings.data_dir / "processed" / "dc_auction_processed.parquet"

        if not da_file.exists() or not bm_file.exists() or not dc_file.exists():
            logger.warning(
                "Historical data incomplete for correlation estimation. "
                "Using default correlation matrix."
            )
            return _default_correlation_matrix(use_spread=use_spread)

        hist_da = pd.read_parquet(da_file)
        hist_da["datetime"] = pd.to_datetime(hist_da["datetime"], utc=True)
        hist_da = hist_da.set_index("datetime").sort_index()["value"]

        hist_bm = pd.read_parquet(bm_file)
        hist_bm["startTime"] = pd.to_datetime(hist_bm["startTime"], utc=True)
        hist_bm = hist_bm.set_index("startTime").sort_index()["systemSellPrice"]

        hist_dc = pd.read_parquet(dc_file)
        hist_dc["delivery_start"] = pd.to_datetime(hist_dc["delivery_start"], utc=True)
        hist_dc = hist_dc.set_index("delivery_start").sort_index()

        hist_dcl = hist_dc[hist_dc["service"] == "DCL"]["clearing_price"]
        hist_dch = hist_dc[hist_dc["service"] == "DCH"]["clearing_price"]

        # Resample DC to 30-min (forward-fill EFA blocks)
        hist_dcl = hist_dcl.resample("30min").ffill()
        hist_dch = hist_dch.resample("30min").ffill()


        hist_da  = hist_da[hist_da.index  < pd.Timestamp(cutoff_date, tz="UTC")]
        hist_bm  = hist_bm[hist_bm.index  < pd.Timestamp(cutoff_date, tz="UTC")]
        hist_dcl = hist_dcl[hist_dcl.index < pd.Timestamp(cutoff_date, tz="UTC")]
        hist_dch = hist_dch[hist_dch.index < pd.Timestamp(cutoff_date, tz="UTC")]


        # Align timestamps and compute Spearman correlation
        if use_spread:
            aligned = pd.DataFrame({
                "da":     hist_da,
                "spread": hist_bm.reindex(hist_da.index, method="ffill") - hist_da,
                "dcl":    hist_dcl.reindex(hist_da.index, method="ffill"),
                "dch":    hist_dch.reindex(hist_da.index, method="ffill"),
            }).dropna()
        else:
            aligned = pd.DataFrame({
                "da":  hist_da,
                "bm":  hist_bm.reindex(hist_da.index, method="ffill"),
                "dcl": hist_dcl.reindex(hist_da.index, method="ffill"),
                "dch": hist_dch.reindex(hist_da.index, method="ffill"),
            }).dropna()


        if len(aligned) < 1000:
            logger.warning(
                f"Only {len(aligned)} aligned observations for correlation. "
                "Using default correlation matrix."
            )
            return _default_correlation_matrix(use_spread=use_spread)

        # Spearman (rank) correlation
        corr_matrix = aligned.corr(method="spearman").values
        corr_matrix = 2 * np.sin(np.pi / 6 * corr_matrix)  # convert to Gaussian copula correlation

        # Ensure positive definite (numerical stability)
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        if eigenvalues.min() < 1e-8:
            logger.warning("Correlation matrix near-singular (min eigenvalue=%.2e). Regularizing.", eigenvalues.min())
            corr_matrix += np.eye(4) * (abs(eigenvalues.min()) + 0.01)
            corr_matrix /= np.sqrt(np.outer(np.diag(corr_matrix), np.diag(corr_matrix)))

        logger.info(
            "Correlation matrix estimated from %d observations (use_spread=%s):\n%s",
            len(aligned), use_spread, corr_matrix,
        )


        return corr_matrix

    except Exception as e:
        logger.error("Error estimating correlation: %s. Using defaults.", e)
        return _default_correlation_matrix(use_spread=use_spread)


def _default_correlation_matrix(use_spread: bool = False) -> np.ndarray:
    if use_spread:
        # Correlation for [DA, spread, DC Low, DC High]
        # spread = BM - DA; its correlation with DA is much lower than BM's
        return np.array([
            [1.00, 0.30, 0.60, 0.55],  # DA
            [0.30, 1.00, 0.35, 0.30],  # spread (BM - DA)
            [0.60, 0.35, 1.00, 0.70],  # DC Low
            [0.55, 0.30, 0.70, 1.00],  # DC High
        ])
    return np.array([
        [1.00, 0.85, 0.60, 0.55],
        [0.85, 1.00, 0.50, 0.45],
        [0.60, 0.50, 1.00, 0.70],
        [0.55, 0.45, 0.70, 1.00],
    ])



def load_latest_forecast(model_id: str) -> pd.DataFrame:
    forecast_dir = settings.data_dir / "forecasts"

    oos_paths = sorted(forecast_dir.glob(f"{model_id}_oos_*.parquet"))
    if oos_paths:
        return pd.read_parquet(oos_paths[-1])

    oof_paths = sorted(forecast_dir.glob(f"{model_id}_oof_*.parquet"))
    if oof_paths:
        logger.warning(
            "No OOS forecast found for %s — falling back to OOF. "
            "OOF forecasts should only be used for the historical backtest period.",
            model_id,
        )
        return pd.read_parquet(oof_paths[-1])

    raise FileNotFoundError(f"No forecast found for {model_id} in {forecast_dir}")

def _efa_block_groups(window_start: pd.Timestamp, horizon: int) -> list[tuple[int, int]]:
    """
    Return (start, end) index pairs for actual EFA blocks within the window.
    EFA blocks are 8 half-hour periods starting at 23:00, 03:00, 07:00,
    11:00, 15:00, 19:00 local GB time. Partial blocks at window edges are included.
    """
    
    utc_offset_h = int(
        window_start.tz_convert(pytz.timezone("Europe/London"))
        .utcoffset().total_seconds() / 3600
    )
    # First EFA boundary after midnight UTC: 03:00 local = (3 - utc_offset_h) UTC
    offset = (3 - utc_offset_h) * 2  # periods: GMT->6, BST->4

    groups = []
    if offset > 0:
        groups.append((0, offset))          # partial first block
    i = offset
    while i + 8 <= horizon:
        groups.append((i, i + 8))           # complete EFA blocks
        i += 8
    if i < horizon:
        groups.append((i, horizon))         # partial last block
    return groups


def sample_scenarios_multimarket(
    da_fc: pd.DataFrame,
    bm_fc: pd.DataFrame,
    dc_low_fc: pd.DataFrame,
    dc_high_fc: pd.DataFrame,
    n: int = N_SCENARIOS,
    seed: int | None = None,
    corr_matrix: np.ndarray | None = None,
    spread_fc: pd.DataFrame | None = None,
) -> dict:
    """
    Generate correlated price scenarios across DA, BM, DC Low, DC High using a Gaussian copula.

    Scenarios represent the central 80% predictive interval only — output is bounded to
    [q10, q90] per market. DC draws are held constant within each 4-hour EFA block.

    Returns dict with keys "da", "bm", "dc_low", "dc_high", each shape (n_scenarios, n_periods).
    """
    rng = np.random.default_rng(seed)
    horizon = len(da_fc)

    # Validate all forecasts have same length
    if not (len(bm_fc) == len(dc_low_fc) == len(dc_high_fc) == horizon):
        raise ValueError(
            f"Forecast length mismatch: DA={horizon}, BM={len(bm_fc)}, "
            f"DC_Low={len(dc_low_fc)}, DC_High={len(dc_high_fc)}"
        )

    # Get correlation matrix
    if corr_matrix is None:
        corr_matrix = _estimate_correlation_matrix(use_spread=spread_fc is not None)

    # Cholesky decomposition for generating correlated normals
    L = np.linalg.cholesky(corr_matrix)

    # Pre-allocate output arrays
    scenarios = {
        "da": np.zeros((n, horizon)),
        "bm": np.zeros((n, horizon)),
        "dc_low": np.zeros((n, horizon)),
        "dc_high": np.zeros((n, horizon)),
    }

    # Generate all correlated normals at once (n, horizon, 4)
    # Applying L to the last axis is equivalent to (L @ Z[s,t,:]) for every (s, t)
    Z = rng.standard_normal((n, horizon, 4))
    X = Z @ L.T                  # (n, horizon, 4)
    U = norm.cdf(X)              # (n, horizon, 4) — uniform marginals

    # DC prices clear at EFA block level (4 hours = 8 half-hour periods).
    # Take the first draw of each block and tile it, preserving U(0,1) spread
    # and the Cholesky cross-market correlation structure.

    # EFA block alignment: the window starts at midnight UTC, which falls INSIDE
    # EFA block 1 (23:00–03:00 local GB time). The first real block boundary is:
    #   GMT: 03:00 UTC = period 6 from midnight
    #   BST: 02:00 UTC = period 4 from midnight
    # We compute block groups from the actual window start rather than assuming
    # a fixed offset. Revenue settlement is unaffected — it uses forward-filled
    # actual prices per period.


    if horizon % 8 != 0:
        raise ValueError(
            f"horizon={horizon} must be divisible by 8 (each EFA block = 8 half-hour periods). "
            "Adjust the horizon passed to sample_scenarios_multimarket."
        )

    window_start = da_fc.index[0]
    efa_groups = _efa_block_groups(window_start, horizon)

    for dc_idx in [2, 3]:
        for blk_start, blk_end in efa_groups:
            U[:, blk_start:blk_end, dc_idx] = U[:, blk_start:blk_start+1, dc_idx]


    # Extract forecast quantile arrays, shape (1, horizon) for broadcasting over n
    def _apply_qf(u: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> np.ndarray:
        """Piecewise-linear interpolation: u ∈ [0,0.5) -> [q10,q50), u ∈ [0.5,1] -> [q50,q90]. Output bounded to [q10, q90]."""
        return np.where(
            u < 0.5,
            q10 + (q50 - q10) * (u / 0.5),
            q50 + (q90 - q50) * ((u - 0.5) / 0.5),
        )
    
    def _block_avg(arr: np.ndarray) -> np.ndarray:
        """Block-average using actual EFA block boundaries."""
        result = np.empty_like(arr, dtype=float)
        for blk_start, blk_end in efa_groups:
            result[blk_start:blk_end] = arr[blk_start:blk_end].mean()
        return result

    second_dim_quantiles = (
        (spread_fc["q10"].values, spread_fc["q50"].values, spread_fc["q90"].values)
        if spread_fc is not None
        else (bm_fc["q10"].values, bm_fc["q50"].values, bm_fc["q90"].values)
    )

    q = {
        "da":      (da_fc["q10"].values,     da_fc["q50"].values,     da_fc["q90"].values),
        "bm":      second_dim_quantiles,  # spread when spread_fc is active, BM otherwise
        "dc_low":  (_block_avg(dc_low_fc["q10"].values), _block_avg(dc_low_fc["q50"].values), _block_avg(dc_low_fc["q90"].values)),
        "dc_high": (_block_avg(dc_high_fc["q10"].values), _block_avg(dc_high_fc["q50"].values), _block_avg(dc_high_fc["q90"].values)),
    }



    for i, key in enumerate(["da", "bm", "dc_low", "dc_high"]):
        q10, q50, q90 = q[key]
        scenarios[key] = _apply_qf(U[:, :, i], q10[np.newaxis, :], q50[np.newaxis, :], q90[np.newaxis, :])


    if spread_fc is not None:
        scenarios["bm"] = scenarios["da"] + scenarios["bm"]

    scenarios["seed"] = seed
    return scenarios




def build_scenarios(horizon: int = 96, seed: int | None = None) -> dict:
    """
    Build correlated price scenarios for all markets.

    Loads latest forecasts and generates scenarios using Gaussian copula
    to preserve realistic cross-market correlation structure.
    """
    da = load_latest_forecast("da")
    bm = load_latest_forecast("bm")
    dc_low = load_latest_forecast("dc_low")
    dc_high = load_latest_forecast("dc_high")

    corr_matrix = _estimate_correlation_matrix(cutoff_date="2025-01-01")

    # Use multimarket scenario generation with correlation
    return sample_scenarios_multimarket(
        da.iloc[:horizon],
        bm.iloc[:horizon],
        dc_low.iloc[:horizon],
        dc_high.iloc[:horizon],
        n=N_SCENARIOS,
        seed=seed,
        corr_matrix=corr_matrix
    )
