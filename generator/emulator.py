import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from model import syn_trace_res
from process_csv import Analyze


class Emulator(Analyze):
    """
    The emulator class is responsible for creating new traces, as well as timestamps.
    It has 2 available methods:

    # Surrogate based on mixing residuals
    This method does not create new residuals, it uses the current traces.

    # Surrogate synthesis
    This method creates new residuals based on the collected statistics

    """

    def __init__(self):
        super().__init__()
        self.aligned_t = np.linspace(0, 3.0, 12345)
        self.cycle_phases = self._align_cycles()

    def _align_cycles(self) -> np.ndarray:
        cycle_phases = []
        phase = np.linspace(0.0, 3.0, 12345)
        for n in self.neural_metric:
            original_t = np.linspace(0.0, 3.0, len(n))
            f = PchipInterpolator(original_t, n)
            cycle_phases.append(f(phase))

        return np.asarray(cycle_phases)

    def create_timestamp(self, start: int, length: int, dt=0.00025):
        return start + dt * np.arange(length)

    def _block_bootstrap(self, series: np.ndarray, block_len: int, out_len: int, rng):
        block_len = min(block_len, len(series))
        out = np.empty(out_len, dtype=float)
        pos = 0
        max_start = len(series) - block_len  # last valid start
        while pos < out_len:
            start = int(rng.integers(0, max_start + 1))
            take = min(block_len, out_len - pos)
            out[pos:pos + take] = series[start:start + take]
            pos += take
        return out

    def shuffle_surrogate(self, n_phase: int, n_surrogates: int, cluster_window: int = 31, block_len: int = 15,
                          mix_cluster_res: float = 0.25):
        """
        This is an alternative method of surrogate creation, it shuffles the already existing residuals in a weighted sum

        :param n_phase: The length of a phase
        :param cluster_window: The parameter used for SavGol smoothing filter
        :param n_surrogates: The number of generated surrogates
        :param block_len: The block length used for bootstraping peaks
        :param mix_cluster_res: The mixing coefficient for used for residual mixing
        :return: An array that containes n_surrogates of length n_phase
        """
        rng = np.random.default_rng()
        mean_cycle = np.median(self.cycle_phases, axis=0)
        residuals = self.cycle_phases - mean_cycle

        cluster_res = savgol_filter(residuals, window_length=cluster_window, polyorder=3, axis=1, mode="interp")
        peak_res = residuals - cluster_res
        surr_stats = []
        surrogate_block = []
        n_cycles = peak_res.shape[0]
        for _ in range(n_surrogates):
            m = max(1, int(np.round(mix_cluster_res * n_cycles)))
            idxs = rng.integers(0, n_cycles, size=m)
            weights = rng.random(m)
            weights /= weights.sum()
            cluster_surr = np.sum(cluster_res[idxs] * weights[:, None], axis=0)

            src = peak_res[int(rng.integers(0, n_cycles))]
            peak_surr = self._block_bootstrap(src, block_len=int(block_len), out_len=n_phase, rng=rng)

            surrogate = mean_cycle + cluster_surr + peak_surr

            surr_stats.append(
                [np.median(surrogate), np.std(surrogate),
                 np.min(surrogate), np.max(surrogate)])

            surrogate_block.append(surrogate)

        return np.asarray(surrogate_block), np.vstack(surr_stats)

    def synthesize_surrogates(self, res_cube: np.ndarray, segment_idx: np.ndarray, n_surrogates: int):
        """
        This function creates new surrogates based on the observed statistics

        :param res_cube: The residual cube is the computed 3D matrix of residuals
        :param segment_idx: Segment index list
        :param n_surrogates: The number of surrogates to be synthesised
        :return: An array that contains n_surrogates
        """
        rng = np.random.default_rng()
        syn_surr = []
        base_smooth = savgol_filter(self.cycle_phases, 31, 3, mode="interp")
        for i in range(n_surrogates):
            # Function that creates a new trace
            new_residuals = syn_trace_res(res_cube, segment_idx)
            random_index = rng.integers(0, self.cycle_phases.shape[0])

            base = base_smooth[random_index]
            y = base + new_residuals
            syn_surr.append(y)

        return syn_surr
