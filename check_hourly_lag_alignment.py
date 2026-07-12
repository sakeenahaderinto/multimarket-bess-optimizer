"""
check_hourly_lag_alignment.py — Manual sanity check of the hourly DA feature
pipeline's lag construction, before committing to a full training run.

Checks:
  1. price_lag_24 == price.shift(24) exactly (within float tolerance)
  2. demand_lag_24 == demand.shift(24) exactly
  3. bm_imbalance_volume_lag_24 actually sums two real half-hour values,
     spot-checked against the raw 30-min series directly
  4. spread_lag_24's source ('spread') is NOT a standalone column -- confirms
     base.py's automatic _check_lag_alignment loop will silently skip this
     one (since 'spread' is never in df.columns), so it is checked manually
     here instead.

Usage:
    uv run phase2/check_hourly_lag_alignment.py
"""

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

from config import settings
from features.pipeline_da_hourly import build_da_hourly_features


def main() -> None:
    # ------------------------------------------------------------------
    # Load raw data -- mirrors pipeline.py steps 1-2 exactly, but stops
    # short of the canonical-30min reindex step, since the hourly pipeline
    # needs these at their native resolution.
    # ------------------------------------------------------------------
    bm_df = pd.read_parquet(settings.data_dir / "processed" / "bmrs_processed.parquet")
    da_df = pd.read_parquet(settings.data_dir / "processed" / "day_ahead_processed.parquet")
    weather_df = pd.concat(
        [pd.read_parquet(f) for f in sorted((settings.data_dir / "raw" / "weather").glob("*.parquet"))],
        ignore_index=True,
    )

    da_df["datetime"] = pd.to_datetime(da_df["datetime"], utc=True)
    da_df_hourly = da_df.set_index("datetime").sort_index()  # DA's true native (hourly) resolution

    bm_df["startTime"] = pd.to_datetime(bm_df["startTime"], utc=True)
    bm_df = bm_df.set_index("startTime").sort_index()

    weather_df = weather_df.set_index("date").sort_index()
    weather_df.index = pd.to_datetime(weather_df.index, utc=True)

    # Native 30-min series, BEFORE any canonical-index reindex --
    # pipeline.py's bm_price_aligned / bm_demand_aligned equivalents, but
    # kept at their raw resolution rather than reindexed onto the 30-min
    # canonical_index (the hourly pipeline does its own resampling from here).
    bm_sell_price_30min = bm_df["systemSellPrice"]
    bm_demand_30min      = bm_df["totalAcceptedOfferVolume"]

    # market_features.py's bm_imbalance_aligned source column, pre-lag,
    # pre-canonical-reindex
    bm_imbalance_30min = bm_df["netImbalanceVolume"]

    df = build_da_hourly_features(
        da_df_hourly, bm_demand_30min, bm_imbalance_30min, bm_sell_price_30min, weather_df
    )

    # ------------------------------------------------------------------
    # Check 1: price_lag_24
    # ------------------------------------------------------------------
    residual_price = (df["price_lag_24"] - df["price"].shift(24)).dropna()
    max_resid_price = residual_price.abs().max()
    print(f"price_lag_24 max residual vs price.shift(24): {max_resid_price:.6f}")
    assert max_resid_price < 1e-6, "price_lag_24 alignment FAILED"
    print("  PASS")

    # ------------------------------------------------------------------
    # Check 2: demand_lag_24
    # ------------------------------------------------------------------
    residual_demand = (df["demand_lag_24"] - df["demand"].shift(24)).dropna()
    max_resid_demand = residual_demand.abs().max()
    print(f"\ndemand_lag_24 max residual vs demand.shift(24): {max_resid_demand:.6f}")
    assert max_resid_demand < 1e-6, "demand_lag_24 alignment FAILED"
    print("  PASS")

    # ------------------------------------------------------------------
    # Check 3: bm_imbalance_volume_lag_24 -- spot-check the sum against
    # raw 30-min values directly, for a handful of real hours
    # ------------------------------------------------------------------
    print("\nbm_imbalance_volume_lag_24 spot-check (sum of two real half-hours):")
    sample_hours = df.dropna(subset=["bm_imbalance_volume_lag_24"]).index[:5]
    for h in sample_hours:
        source_hour = h - pd.Timedelta(hours=24)
        half1 = bm_imbalance_30min.reindex([source_hour]).iloc[0]
        half2 = bm_imbalance_30min.reindex([source_hour + pd.Timedelta(minutes=30)]).iloc[0]
        expected_sum = half1 + half2
        actual_val = df.loc[h, "bm_imbalance_volume_lag_24"]
        match = np.isclose(expected_sum, actual_val, atol=1e-6)
        print(
            f"  {h}: half1={half1:.2f} half2={half2:.2f} "
            f"expected_sum={expected_sum:.2f} actual={actual_val:.2f}  "
            f"{'PASS' if match else 'FAIL'}"
        )

    # ------------------------------------------------------------------
    # Check 4: confirm 'spread' is not a standalone column, so base.py's
    # automatic _check_lag_alignment loop silently skips spread_lag_24
    # ------------------------------------------------------------------
    print(f"\n'spread' in df.columns: {'spread' in df.columns}")
    print(
        "  Expected: False. This confirms base.py's automatic lag-alignment "
        "loop will SKIP spread_lag_24 entirely (since its regex-derived "
        "source_col 'spread' never exists as a column) -- it is NOT being "
        "verified automatically during training. Manual check below."
    )

    residual_spread = (
        df["spread_lag_24"] - (df["price"].shift(24) - df["price"].shift(24) * 0)
    )  # placeholder -- real check needs bm_sell_price_30min resampled the same way
    # Proper manual check: recompute spread_lag_24 independently and compare
    bm_sell_hourly = bm_sell_price_30min.resample("1h").mean().reindex(df.index)
    expected_spread_lag_24 = df["price"].shift(24) - bm_sell_hourly.shift(24)
    residual_spread = (df["spread_lag_24"] - expected_spread_lag_24).dropna()
    max_resid_spread = residual_spread.abs().max()
    print(f"\nspread_lag_24 max residual vs manual recomputation: {max_resid_spread:.6f}")
    assert max_resid_spread < 1e-6, "spread_lag_24 alignment FAILED"
    print("  PASS")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()