import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, stats
import pandas as pd
from process_csv import ProcessCSV, Y_LIMIT, WIDE_FIG_SIZE, PARAM_FONT_SIZE, LABEL_FONT_SIZE


class Validate(ProcessCSV):
    """
    The class validate is responsible for performing validation as well as outputting the syn traces persistently
    """

    def __init__(self, stop_event):
        super().__init__()
        os.makedirs(self._OUTPUT_FILE, exist_ok=True)
        self.validation_path = os.path.join(self._OUTPUT_FILE, "validation.csv")
        self.stop_event = stop_event

        columns = [
            "feature_id",
            "median_real",
            "ci_real",
            "median_syn",
            "ci_syn",
            "overall_acf",
            "ks sample test",
            "max ks drift",
            "min ks drift"
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.validation_path, index=False)

    def compute_acf(self, x: np.ndarray, max_lag: int = 200) -> np.ndarray:
        x = x - np.mean(x)

        acf_vals = np.empty(max_lag + 1, dtype=float)
        var = float(np.dot(x, x))
        if var <= 0.0:
            acf_vals[:] = 0.0
            acf_vals[0] = 1.0
            return acf_vals

        for lag in range(max_lag + 1):
            acf_vals[lag] = np.dot(x[:-lag or None], x[lag:]) / var

        return acf_vals

    def perform_acf_validation(self, syn: np.ndarray, neural: np.ndarray, scatter=None) -> (np.ndarray, np.ndarray):

        lag_max = int(0.1 * 12345)

        real_acf, syn_acf = [], []

        lags = np.arange(lag_max + 1) * 0.00025
        for s, n in zip(syn, neural):
            r = self.compute_acf(n, lag_max)  # real
            ss = self.compute_acf(s, lag_max)  # synthetic

            real_acf.append(r)
            syn_acf.append(ss)

        if scatter is not None:
            real_mean = np.mean(np.asarray(real_acf), axis=0)
            syn_mean = np.mean(np.asarray(syn_acf), axis=0)

            plt.figure(figsize=(10, 5))
            plt.plot(lags, real_mean, label="Real", alpha=0.9)
            plt.plot(lags, syn_mean, label="Synthetic", alpha=0.9)

            plt.xlabel("Lag (seconds)", fontsize=LABEL_FONT_SIZE)
            plt.ylabel("Autocorrelation", fontsize=LABEL_FONT_SIZE)
            plt.tick_params(axis='both', labelsize=14, length=6, width=1.5)
            plt.ylim(0, 1)
            plt.legend(fontsize=14)
            plt.tight_layout()
            plt.show()

        return real_acf, syn_acf

    def overall_acf_distance(self, real_cycles: np.ndarray, syn_cycles: np.ndarray, alpha=0.05) -> (float, float):
        """
        real_cycles, syn_cycles: arrays/lists of shape (N, T)
        compute_acf(x, lag_max) -> (lag_max+1,) ACF including lag 0
        Returns: mean_distance, std_distance
        """

        lag_max = int(0.1 * 12345)

        dists = []
        for r, s in zip(real_cycles, syn_cycles):
            ar = self.compute_acf(r, lag_max)[1:]  # drop lag 0
            as_ = self.compute_acf(s, lag_max)[1:]
            dists.append(np.linalg.norm(ar - as_))

        dists = np.asarray(dists, dtype=float)

        mean = float(np.mean(dists))
        std = float(np.std(dists, ddof=1))
        return mean, std

    def summarize_validation_metrics_ci(self, cube: np.ndarray, alpha=0.05) -> (float, float):
        """
        Returns median ± CI over blocks
        """
        assert cube.ndim ==3
        block_avg = np.median(cube, axis=1)
        m = np.median(block_avg, axis=0)

        D = block_avg.shape[0]
        if D < 2:
            ci = np.zeros_like(m, dtype=float)
            return m, ci

        s = np.std(block_avg, axis=0, ddof=1)
        tval = t.ppf(1 - alpha / 2, df=D - 1)
        ci = tval * s / np.sqrt(D)

        return m, ci

    def perform_ks_test(self, syn: np.ndarray, neural: np.ndarray) -> (float, float, float):
        stat_block = []
        for s, n in zip(syn, neural):
            stat, _ = stats.ks_2samp(s, n)
            stat_block.append(stat)

        stat_block = np.asarray(stat_block)
        med = np.median(stat_block)
        lower = np.percentile(stat_block, 2.5)
        upper = np.percentile(stat_block, 97.5)
        ci_lower = med - lower
        ci_upper = upper - med
        return med, ci_lower, ci_upper

    def perform_all_validation(self, syn: np.ndarray, neural: np.ndarray, syn_cube: np.ndarray,
                               neural_cube: np.ndarray):
        while not self.stop_event.is_set():
            overall_acf_mean, overall_acf_std = self.overall_acf_distance(neural, syn)
            m_real, ci_real = self.summarize_validation_metrics_ci(neural_cube)
            m_syn, ci_syn = self.summarize_validation_metrics_ci(syn_cube)
            med, ci_low, ci_high = self.perform_ks_test(syn, neural)

            rows = []
            features = ["Median", "MAD", "Q90", "Skewness"]
            for i in range(len(m_real)):
                rows.append({
                    "feature_id": features[i],
                    "median_real": float(m_real[i]),
                    "ci_real": float(ci_real[i]),
                    "median_syn": float(m_syn[i]),
                    "ci_syn": float(ci_syn[i]),
                    "overall_acf": float(overall_acf_mean),
                    "ks sample": float(med),
                    "max ks drift": float(ci_high),
                    "min ks drift": float(ci_low)
                })

            df_new = pd.DataFrame(rows)

            df_new.to_csv(
                self.validation_path,
                mode="a",
                header=False,
                index=False,
                float_format="%.5f"
            )
            pass

    def display_results(self, t_surr: np.ndarray, syn_trace: np.ndarray, neural: np.ndarray, neural_t: np.ndarray,
                        nr: int):

        i = 0
        for syn, y, t in zip(syn_trace, neural, neural_t):
            if i == nr:
                break
            plt.figure(figsize=WIDE_FIG_SIZE)
            t = t - np.min(t)

            L = min(len(t), len(t_surr), len(y), len(syn))
            t_surr_plot = t_surr[:L]

            # plt.scatter(t_plot, y[:L], s=3, label="Collected electrical metric", color="tab:blue", alpha=0.6)
            plt.scatter(t_surr_plot, syn[:L], s=3, label="Synthetic surrogate", color="tab:orange", alpha=0.6)
            plt.xlabel("Time in seconds", fontsize=LABEL_FONT_SIZE)
            plt.ylabel("Electrical current", fontsize=LABEL_FONT_SIZE)
            plt.tick_params(axis='both', labelsize=14, length=6, width=1.5)
            plt.ylim(Y_LIMIT)
            plt.legend(markerscale=1.2, fontsize=14)
            legend = plt.legend(markerscale=1, fontsize=LABEL_FONT_SIZE)
            for handle in legend.legendHandles:
                handle.set_alpha(1)
                handle.set_sizes([100])
            plt.show()
            i += 1
