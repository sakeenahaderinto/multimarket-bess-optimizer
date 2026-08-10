import logging
import time
from datetime import UTC, datetime
from pathlib import Path
import re

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import settings
from features.lag_features import FORECAST_GATE_OFFSET

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"

MIN_TRAIN = 48 * 28      # 4 weeks before first fold
FOLD_STEP = 48 * 7       # refit weekly
VAL_WINDOW = FOLD_STEP          # 1-week validation window matches fold step for gapless OOF prediction
MIN_FOLD_ROWS = 48 * 7   # require at least a week of clean rows per fold
RANDOM_STATE = 42

QUANTILES = [0.1, 0.5, 0.9]

LGBM_SHARED_PARAMS = dict(
    objective="quantile",
    learning_rate=0.05,
    num_leaves=31,
    n_jobs=-1,
    verbose=-1,
    random_state=RANDOM_STATE,
)
CV_N_ESTIMATORS = 500

# ---------------------------------------------------------------------------
# Base Features (common across all models)
# ---------------------------------------------------------------------------

BASE_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "year_sin",
    "year_cos",
    "demand_lag_48",
    "demand_lag_96",
    "demand_lag_144",
    "demand_lag_336",
    "price_lag_48",
    "price_lag_96",
    "price_lag_144",
    "price_lag_336",
    "demand_roll_mean_8",
    "demand_roll_std_8",
    "price_roll_mean_8",
    "price_roll_std_8",
    "demand_roll_mean_48",
    "demand_roll_std_48",
    "price_roll_mean_48",
    "price_roll_std_48",
    "demand_roll_mean_336",
    "demand_roll_std_336",
    "price_roll_mean_336",
    "price_roll_std_336",
    "wind_speed",
    "cloud_cover",
    "bm_imbalance_volume_lag_48",
]


class BaseForecaster:
    model_id: str
    target_col: str
    feature_cols: list[str]

    MIN_TRAIN = 48 * 28
    FOLD_STEP = 48 * 7
    VAL_WINDOW = 48 * 7
    MIN_FOLD_ROWS = 48 * 7

    GATE_ORIGIN_COLS: frozenset = frozenset({
        "demand_lag_48",
        "price_lag_48",
        "bm_imbalance_volume_lag_48",
        "wind_change_lag_48",
        "spread_lag_48",
    })

    def load_features(self) -> pd.DataFrame:
        return pd.read_parquet(settings.data_dir / "features" / "features.parquet")

    def _pinball(self, actuals: np.ndarray, preds: np.ndarray, q: float) -> float:
        """Pinball loss for quantile regression evaluation."""
        errors = actuals - preds
        return float(np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors)))

    def _report_crossing_rate(self, df: pd.DataFrame, context: str) -> pd.DataFrame:
        """
        Correct quantile crossing in [q10, q50, q90] and log the crossing rate.

        A crossing rate >5% suggests the quantile models may not be well
        calibrated and should be investigated before using forecasts in the
        optimiser or reporting results in a paper.
        """
        q_cols = ["q10", "q50", "q90"]
        n_crossings = (
            (df["q10"] > df["q50"]) | (df["q50"] > df["q90"])
        ).sum()
        crossing_rate = n_crossings / len(df)

        logger.info(
            "%s [%s] quantile crossing rate: %.2f%% (%d of %d rows)",
            self.model_id, context, crossing_rate * 100, n_crossings, len(df),
        )
        if crossing_rate > 0.05:
            logger.warning(
                "%s [%s] crossing rate >5%% — consider reviewing quantile model calibration.",
                self.model_id, context,
            )

        if n_crossings > 0:
            df = df.copy()
            df[q_cols] = np.sort(df[q_cols].values, axis=1)

        return df
    
    @staticmethod
    def _check_lag_alignment(df: pd.DataFrame, target: str, lag_col: str, lag_n: int = 48) -> None:
        """
        Raise if lag_col != source.shift(lag_n) within floating-point tolerance.

        Correlation is computed but only logged (DEBUG) — not used to raise, because
        strong same-step correlation is expected from daily/weekly seasonality and
        is not reliable evidence of leakage.
        """
        if lag_col not in df.columns or target not in df.columns:
            return

        residual = (df[lag_col] - df[target].shift(lag_n)).dropna()

        if residual.empty:
            logger.warning(
                "Lag alignment check skipped for '%s'. No overlapping non-null rows between '%s' and its shift(%d).",
                lag_col, target, lag_n,
            )
            return

        max_residual = residual.abs().max()

        if max_residual > 1e-6:
            raise ValueError(
                f"Lag alignment error: '{lag_col}' != '{target}'.shift({lag_n}). "
                f"Max residual: {max_residual:.6f}. Check feature construction in pipeline.py."
            )

        corr_same = df[target].corr(df[lag_col])
        corr_shifted = df[target].shift(lag_n).corr(df[lag_col])
        logger.debug(
            "Lag alignment check  col=%s lag_n=%d  corr_at_t=%.3f  corr_at_t-%d=%.3f",
            lag_col, lag_n, corr_same, lag_n, corr_shifted,
        )


    def _fit_quantile(
        self,
        q: float,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[float, np.ndarray, int]:
        """Fit one quantile model for a single fold. Returns (q, val_preds, best_iteration)."""
        model = lgb.LGBMRegressor(
            **LGBM_SHARED_PARAMS, alpha=q, n_estimators=CV_N_ESTIMATORS
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )
        return q, model.predict(X_val), model.best_iteration_

    def train(self, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        """
        Walk-forward cross-validation training with OOF collection.

        Returns:
            models: dict[quantile -> LGBMRegressor]
            oof_df: DataFrame with OOF predictions (index=datetime, columns=[q10, q50, q90])
        """
        logger.info("Starting walk-forward CV for %s...", self.model_id)

        cols_needed = self.feature_cols + [self.target_col]

        all_preds = {q: [] for q in QUANTILES}
        all_actuals: list[float] = []
        fold_indices: list[int] = []
        n_iters_per_q = {q: [] for q in QUANTILES}

        oof_preds = {q: [] for q in QUANTILES}
        oof_actuals = []
        oof_timestamps = []

        folds_run = folds_skipped = 0
        fold_ends = range(self.MIN_TRAIN, len(df) - self.VAL_WINDOW, self.FOLD_STEP)
        total_folds = len(fold_ends)
        t0 = time.time()

        for i, fold_end in enumerate(fold_ends):
            # Cut training at the forecast origin (gate close) of the first validation day.
            # A fold boundary at midnight D would otherwise include D-1 actual prices
            # from 11:00–23:30 D-1, which were not available at the 11:00 D-1 decision time.
            first_val_ts = df.index[fold_end]
            forecast_origin = first_val_ts.normalize() - FORECAST_GATE_OFFSET
            train_slice = df[df.index < forecast_origin].dropna(subset=cols_needed)
            val_slice = df.iloc[fold_end : fold_end + self.VAL_WINDOW].dropna(subset=cols_needed)

            if len(train_slice) < self.MIN_FOLD_ROWS or len(val_slice) == 0:
                folds_skipped += 1
                continue

            X_train = train_slice[self.feature_cols]
            y_train = train_slice[self.target_col]
            X_val = val_slice[self.feature_cols]
            y_val = val_slice[self.target_col]

            for q in QUANTILES:
                _, preds, best_iter = self._fit_quantile(q, X_train, y_train, X_val, y_val)
                all_preds[q].extend(preds.tolist())
                oof_preds[q].extend(preds.tolist())
                n_iters_per_q[q].append(best_iter)


            all_actuals.extend(y_val.tolist())
            fold_indices.extend([i] * len(y_val))
            oof_actuals.extend(y_val.tolist())
            oof_timestamps.extend(val_slice.index.tolist())
            folds_run += 1

            if i > 0 and i % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / i * (total_folds - i)
                logger.info(
                    "  fold %d/%d  train=%d  val=%d  ETA %.0fs",
                    i, total_folds, len(train_slice), len(val_slice), eta,
                )

        logger.info("Folds run: %d  skipped: %d", folds_run, folds_skipped)

        if not all_actuals:
            raise RuntimeError(
                "No validation data collected — check feature columns and NaN counts."
            )

        # ------------------------------------------------------------------
        # CV Metrics
        # ------------------------------------------------------------------
        results_df = pd.DataFrame(
            {
                "fold": fold_indices,
                "actual": all_actuals,
                "pred_50": all_preds[0.5],
            }
        )
        results_df["ae"] = (results_df["actual"] - results_df["pred_50"]).abs()

        mid = results_df["fold"].median()
        early_mae = results_df.loc[results_df["fold"] <= mid, "ae"].mean()
        late_mae = results_df.loc[results_df["fold"] > mid, "ae"].mean()

        actuals_arr = np.array(all_actuals)
        pb10 = self._pinball(actuals_arr, np.array(all_preds[0.1]), 0.1)
        pb90 = self._pinball(actuals_arr, np.array(all_preds[0.9]), 0.9)

        logger.info(
            "CV metrics  MAE_overall=%.3f  MAE_early=%.3f  MAE_late=%.3f  "
            "pinball_q10=%.3f  pinball_q90=%.3f",
            results_df["ae"].mean(), early_mae, late_mae, pb10, pb90,
        )

        self._val_metrics = {
            "mae": float(results_df["ae"].mean()),
            "pinball_q10": pb10,
            "pinball_q90": pb90,
        }

        # ------------------------------------------------------------------
        # OOF DataFrame — crossing rate reported and corrected here
        # ------------------------------------------------------------------
        oof_df = None
        if oof_timestamps:
            oof_df = (
                pd.DataFrame(
                    {
                        "datetime": oof_timestamps,
                        "actual": oof_actuals,
                        "q10": oof_preds[0.1],
                        "q50": oof_preds[0.5],
                        "q90": oof_preds[0.9],
                    }
                )
                .set_index("datetime")
                .sort_index()
            )
            oof_df = self._report_crossing_rate(oof_df, context="OOF")

        # ------------------------------------------------------------------
        # Best tree count: median across folds per quantile
        # ------------------------------------------------------------------
        best_n_estimators = {}
        for q in QUANTILES:
            n = int(np.median(n_iters_per_q[q])) if n_iters_per_q[q] else 200
            best_n_estimators[q] = max(n, 50)
            logger.info("  q%d  median best_iteration=%d", int(q * 100), best_n_estimators[q])

        # ------------------------------------------------------------------
        # Final models — trained on full train_val set
        # ------------------------------------------------------------------
        df_clean = df.dropna(subset=cols_needed)
        X_full = df_clean[self.feature_cols]
        y_full = df_clean[self.target_col]

        logger.info("Training final models on %d rows...", len(X_full))

        final_models: dict[float, lgb.LGBMRegressor] = {}
        for q in QUANTILES:
            model = lgb.LGBMRegressor(
                **LGBM_SHARED_PARAMS,
                alpha=q,
                n_estimators=best_n_estimators[q],
            )
            model.fit(X_full, y_full)
            final_models[q] = model

        return final_models, oof_df

    def save_models(self, models: dict) -> dict[float, Path]:
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for quantile, model in models.items():
            path = settings.models_dir / f"{self.model_id}_q{int(quantile * 100)}.txt"
            model.booster_.save_model(str(path))
            paths[quantile] = path
            logger.info("  Saved %s", path)
        return paths

    def predict(self, models: dict, df: pd.DataFrame) -> pd.DataFrame:
        """Generate quantile forecasts, report crossing rate, and correct any crossings."""
        forecasts = pd.DataFrame(index=df.index)
        for quantile, model in models.items():
            forecasts[f"q{int(quantile * 100)}"] = model.predict(df[self.feature_cols])

        forecasts = self._report_crossing_rate(forecasts, context="OOS")
        return forecasts

    def save_oof_forecasts(self, oof_df: pd.DataFrame) -> None:
        """Save out-of-fold predictions for bias-free 2022-2024 backtest."""
        if oof_df is None or oof_df.empty:
            logger.warning("No OOF predictions to save for %s", self.model_id)
            return

        forecast_dir = settings.data_dir / "forecasts"
        forecast_dir.mkdir(parents=True, exist_ok=True)

        oof_path = forecast_dir / f"{self.model_id}_oof_2022_2024.parquet"
        oof_df[["q10", "q50", "q90"]].to_parquet(oof_path)
        logger.info("OOF predictions saved to %s  (%d rows)", oof_path, len(oof_df))

    def save_oos_forecasts(self, forecasts: pd.DataFrame) -> None:
        """Save true out-of-sample forecasts for 2025 backtest."""
        if forecasts.empty:
            logger.warning("No OOS predictions to save for %s", self.model_id)
            return

        forecast_dir = settings.data_dir / "forecasts"
        forecast_dir.mkdir(parents=True, exist_ok=True)

        oos_path = forecast_dir / f"{self.model_id}_oos_2025.parquet"
        forecasts.to_parquet(oos_path)
        logger.info("OOS forecasts saved to %s  (%d rows)", oos_path, len(forecasts))


    def run(self) -> None:
        """Main training pipeline: CV with OOF + final model with OOS forecasts."""
        logger.info("=" * 60)
        logger.info("Training %s  (target=%s)", self.model_id, self.target_col)
        logger.info("=" * 60)

        df = self.load_features()

        cols_needed = self.feature_cols + [self.target_col]
        missing = [c for c in cols_needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in features file: {missing}")

        nan_counts = df[cols_needed].isnull().sum()
        if nan_counts.any():
            logger.info("NaN counts in used columns:\n%s", nan_counts[nan_counts > 0])

        for lag_col in [c for c in self.feature_cols if re.search(r"_lag_\d+$", c)]:
            match = re.search(r"_lag_(\d+)$", lag_col)
            lag_n = int(match.group(1))
            if lag_col in self.GATE_ORIGIN_COLS:
                # gate-origin columns use a fixed daily lookup, not shift(lag_n) — skip exact-shift check
                continue
            source_col = lag_col[:match.start()]
            if source_col in df.columns:
                self._check_lag_alignment(df, target=source_col, lag_col=lag_col, lag_n=lag_n)

        logger.info("Lag alignment checks passed.")

        oos_start = pd.Timestamp(TEST_START_DATE, tz=df.index.tz)
        oos_forecast_origin = oos_start.normalize() - FORECAST_GATE_OFFSET
        df_train_val = df[df.index < oos_forecast_origin].copy()
        df_test = df[df.index >= oos_start].copy()
        logger.info(
            "Split: train_val=%d rows (cutoff %s = gate close for OOS start %s), "
            "test=%d rows (from %s onwards)",
            len(df_train_val), oos_forecast_origin, TEST_START_DATE,
            len(df_test), TEST_START_DATE,
        )

        models, oof_df = self.train(df_train_val)
        paths = self.save_models(models)
        self.save_oof_forecasts(oof_df)

        # OOS prediction: only requires features to be non-null
        df_test_pred = df_test.dropna(subset=self.feature_cols)
        if df_test_pred.empty:
            logger.warning("No valid test rows for %s — OOS forecast not saved.", self.model_id)
        else:
            forecasts = self.predict(models, df_test_pred)
            self.save_oos_forecasts(forecasts)

            # Test metrics: also require the target to be non-null
            df_test_eval = df_test_pred.dropna(subset=[self.target_col])
            if not df_test_eval.empty:
                actuals = df_test_eval[self.target_col].copy()
                q10 = forecasts.loc[df_test_eval.index, "q10"]
                q50 = forecasts.loc[df_test_eval.index, "q50"]
                q90 = forecasts.loc[df_test_eval.index, "q90"]

                if self.model_id in ["dc_low", "dc_high"]:
                    actuals = actuals.resample("4h", offset="23h").mean().dropna()
                    q10 = q10.resample("4h", offset="23h").mean().reindex(actuals.index)
                    q50 = q50.resample("4h", offset="23h").mean().reindex(actuals.index)
                    q90 = q90.resample("4h", offset="23h").mean().reindex(actuals.index)
                    logger.info("Evaluated %s test metrics at EFA block level (4h).", self.model_id)

                actuals_arr = actuals.values
                mae      = float(np.mean(np.abs(actuals_arr - q50.values)))
                pb10     = self._pinball(actuals_arr, q10.values, 0.1)
                pb90     = self._pinball(actuals_arr, q90.values, 0.9)
                coverage = float(np.mean((actuals_arr >= q10.values) & (actuals_arr <= q90.values)) * 100)

                logger.info(
                    "Test metrics [%s]: MAE=%.3f  pinball_q10=%.3f  pinball_q90=%.3f  "
                    "coverage(q10-q90)=%.1f%% (target: 80.0%%)",
                    self.model_id, mae, pb10, pb90, coverage,
                )
            else:
                logger.warning(
                    "No non-null target rows in test set for %s — metrics not computed.",
                    self.model_id,
                )

        logger.info("Forecaster %s completed", self.model_id)