import numpy as np
from matplotlib import pyplot as plt


def gen_residuals(feat: np.ndarray, n, rng) -> np.ndarray:
    """
    The function generates residuals based on a computed feature vector

    @:param feat -> feature vector
    @:param n -> number of segments
    @:param rng -> Random number generator object

    @:return: y -> a new array of residuals
    """

    eps = 1e-12
    q50 = float(feat[0])
    mad = float(feat[1])
    q90 = float(feat[2])
    bowley = float(np.clip(feat[3], -1.0, 1.0))

    right_spread = max(q90 - q50, eps)
    p_grid = (0.01, 0.10, 0.50, 0.90, 0.99)
    left_tail_mult = 1.0
    right_tail_mult = 1.0

    # Core asymmetry driven by bowley
    # bowley=0 => symmetric core
    # bowley>0 => more mass/spread on the right, less on the left
    # bowley<0 => more on the left, less on the right
    right_core = (1.0 + bowley) * 0.5  # in [0,1]
    left_core = 1.0 - right_core

    # Define a core shoulder at p=0.10 and p=0.90
    # p=0.90 is pinned to q90 (given). p=0.10 is built from MAD + skew.
    q10 = q50 - mad * (0.5 + left_core)  # explicit, simple mapping

    # Outer tails: explicit, tunable
    q01 = q50 - left_tail_mult * mad
    q99 = q50 + right_tail_mult * right_spread

    p = np.array(p_grid, dtype=float)
    q = np.array([q01, q10, q50, q90, q99], dtype=float)

    # enforce monotone quantiles
    q = np.maximum.accumulate(q)

    # plt.figure()
    # plt.plot(p, q, marker="o")
    # plt.xlabel("P (CDF probability)")
    # plt.ylabel("Values")
    # plt.title("Piecewise-linear inverse CDF (quantile knots)")
    # plt.grid(True)
    # plt.show()

    u = rng.random(n)
    y = np.interp(u, p, q, left=q[0], right=q[-1])
    y -= np.median(y)
    return y


def syn_trace_res(feat_cube:np.ndarray, seg_cuts:np.ndarray, topk:int=2) -> np.ndarray:
    """
    The function creates a new synthetic trace segment-wise
    @:param feat_cube -> the cube of features
    @:param n -> the segments' indexes
    @:param topk -> random integer value

    @return: y -> a new array of residuals
    """
    rng = np.random.default_rng()

    X = np.asarray(feat_cube, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"feat_tensor must be 3D (D,N,F), got {X.shape}")

    D, N, F = X.shape

    new_trace = np.empty(12345, dtype=float)

    seg_amp = np.median(X[:, :, 1], axis=1)
    seg_amp = seg_amp / (np.max(seg_amp) + 1e-12)

    start = 0
    for d in range(D):
        if d == 13:
            end = 12345
        else:
            end = int(seg_cuts[d])

        nseg = end - start
        # choose cycles to mix
        k = min(max(topk, 1), N)
        idx = rng.choice(N, size=k, replace=False)
        w = np.zeros(N, dtype=float)
        w[idx] = 1.0 / k

        # mix feature vectors for this segment
        f_mix = (w[:, None] * X[d]).sum(axis=0)

        # generate residuals for this segment
        seg = gen_residuals(f_mix, nseg, rng=rng)

        # CRITICAL: gate residual strength by real segment activity
        seg *= seg_amp[d]
        new_trace[start:end] = seg
        start = end
    return new_trace


def extract_slice_features(s: np.ndarray, t: np.ndarray | None, show: bool) -> np.ndarray:
    """
    The function extracts the statistics of a segment. The statistics are used to characterize
    the non-Gaussian distribution
    @:param s -> the segment array
    @:param n -> the time array for that segment (used for plotting)
    @:param show -> flag used for displaying the histogram

    @result: arr -> a new array of the features that characterizes the non-Gaussian distribution
                    the values chosen are: [median, MAD, W90, Bowley skewness]
    """

    # ---- deviations ----
    abs_dev = np.abs(s - np.median(s))
    mad = np.median(abs_dev)

    # ---- quantiles ----
    q01, q05, q10, q25, q50, q75, q90, q95, q99 = np.percentile(
        s, [1, 5, 10, 25, 50, 75, 90, 95, 99]
    )

    # ---- spreads ----
    iqr = q75 - q25
    w90 = q95 - q05  # Q95-Q05

    # ---- shape ----
    bowley_skew = (q75 + q25 - 2.0 * q50) / (iqr + 1e-12)

    if t is not None and show:
        plt.scatter(t, s, s=3)
        plt.xlabel("Time in seconds", fontsize=16)
        plt.ylabel("Electrical current", fontsize=16)
        plt.tick_params(axis='both', labelsize=14, length=6, width=1.5)
        plt.legend(markerscale=1.2, fontsize=14)
        plt.show()

        plt.hist(s, bins=60)
        plt.show()

    return np.array([
        np.median(s),
        mad,
        w90,
        bowley_skew
    ])
