"""Pure NumPy reimplementation of the ScanMatch algorithm (Cristino et al., 2010),
ported from the original MATLAB ScanMatch toolbox (https://seis.bristol.ac.uk/~psidg/ScanMatch/).

Avoids the ScanMatchPy/MATLAB Runtime dependency, which requires Python 2.7/3.6/3.7
and a separately-installed MATLAB Runtime 9.7 (no macOS arm64 support).

Validated against a real MATLAB R2026b installation (via the MATLAB Engine API for
Python) running the original, open ScanMatch.m toolbox: for identical RoI sequences
and identical parameters (Xbin/Ybin/Threshold/GapValue), this module's
create_sub_matrix + needleman_wunsch reproduce MATLAB's ScanMatch.m score bit-for-bit,
and fixation_to_roi reproduces ScanMatch_GridMask's RoI numbering exactly (up to the
expected 0- vs 1-indexing offset). What is NOT reproducible is kreimanlab's own
findScore.m wrapper (used by the ScanMatchPy pip package) -- that function is
distributed only as compiled/encrypted p-code (MATLAB Compiler SDK "CTF" format) and
bakes in its own choice of Xbin/Ybin/Threshold/GapValue, which is not recoverable by
any means (not a licensing issue -- this is true even with a full MATLAB license).
So: the *algorithm* here is verified correct; the *exact scale* of the original
paper's cached ss_score/*.npy values is not reproducible, since that depends on
kreimanlab's own undisclosed parameter choice. Use the Xbin/Ybin/Threshold below
(or your own, documented choice) consistently for your own comparisons instead.

Ported pieces (MATLAB source -> here):
- ScanMatch_GridMask.m + ScanMatch_FixationToSequence.m -> fixation_to_roi
- ScanMatch_CreateSubMatrix.m -> create_sub_matrix
- ScanMatch_nwAlgo.m -> needleman_wunsch
- ScanMatch.m (scoring/normalization) -> scan_match_score / mean_scan_match_score

Temporal binning (ScanMatchInfo.TempBin) is not implemented: this pipeline does not
record fixation durations, so one symbol per fixation (TempBin=0) is the only
applicable mode anyway.
"""

import numpy as np


def fixation_to_roi(xy, Xres, Yres, Xbin, Ybin):
    """xy: (N, 2) array of (row, col) pixel coordinates.
    Returns (N,) int array of RoI ids in [0, Xbin*Ybin), row-major over (y_bin, x_bin).
    """
    xy = np.asarray(xy, dtype=float)
    rows = np.clip(xy[:, 0], 0, Yres - 1)
    cols = np.clip(xy[:, 1], 0, Xres - 1)
    bin_y = np.clip(np.floor(rows * Ybin / Yres).astype(int), 0, Ybin - 1)
    bin_x = np.clip(np.floor(cols * Xbin / Xres).astype(int), 0, Xbin - 1)
    return bin_y * Xbin + bin_x


def create_sub_matrix(Xbin, Ybin, threshold):
    """SubMatrix[i, j] = threshold - euclidean_dist(bin_i, bin_j), bin positions laid
    out the same way as fixation_to_roi (row-major over (y_bin, x_bin))."""
    n = Xbin * Ybin
    ys, xs = np.divmod(np.arange(n), Xbin)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    return threshold - dist


def needleman_wunsch(seq1, seq2, sub_matrix, gap=0.0):
    """Standard global alignment (Needleman-Wunsch); returns the raw (unnormalized)
    alignment score. seq1/seq2: 1D int arrays of RoI ids."""
    m, n = len(seq1), len(seq2)
    F = np.zeros((n + 1, m + 1))
    F[1:, 0] = gap * np.arange(1, n + 1)
    F[0, 1:] = gap * np.arange(1, m + 1)

    for j in range(1, m + 1):
        match_col = sub_matrix[seq2, seq1[j - 1]]
        for i in range(1, n + 1):
            diagonal = F[i - 1, j - 1] + match_col[i - 1]
            up = F[i - 1, j] + gap
            left = F[i, j - 1] + gap
            F[i, j] = max(diagonal, up, left)

    return F[n, m]


def scan_match_score(fix1, n1, fix2, n2, Xres, Yres, Xbin=12, Ybin=8, threshold=3.5, gap=0.0):
    """fix1/fix2: (NumFix, 2) padded fixation arrays for ONE stimulus (row, col pixels).
    n1/n2: actual fixation counts (entries beyond this are padding, ignored).
    Returns the normalized ScanMatch similarity score for this one stimulus."""
    n1, n2 = int(n1), int(n2)
    if n1 <= 0 or n2 <= 0:
        return 0.0

    seq1 = fixation_to_roi(np.asarray(fix1)[:n1], Xres, Yres, Xbin, Ybin)
    seq2 = fixation_to_roi(np.asarray(fix2)[:n2], Xres, Yres, Xbin, Ybin)

    sub_matrix = create_sub_matrix(Xbin, Ybin, threshold)
    raw_score = needleman_wunsch(seq1, seq2, sub_matrix, gap=gap)

    scale = threshold * max(len(seq1), len(seq2))
    return raw_score / scale if scale != 0 else 0.0


def mean_scan_match_score(fix_a, I_a, fix_b, I_b, Xres, Yres, Xbin=12, Ybin=8, threshold=3.5, gap=0.0):
    """fix_a/fix_b: (NumStimuli, NumFix, 2); I_a/I_b: (NumStimuli, 1) fixation counts.
    Averages scan_match_score(...) over all NumStimuli -- mirrors ScanMatchPy's
    findScore(fix1, fix2, I_fix1, I_fix2) semantics (one call = one subject/model-pair
    comparison, averaged over stimuli)."""
    fix_a = np.asarray(fix_a)
    fix_b = np.asarray(fix_b)
    I_a = np.asarray(I_a).reshape(-1)
    I_b = np.asarray(I_b).reshape(-1)

    num_stimuli = fix_a.shape[0]
    scores = np.zeros(num_stimuli)
    for i in range(num_stimuli):
        scores[i] = scan_match_score(fix_a[i], I_a[i], fix_b[i], I_b[i],
                                      Xres, Yres, Xbin=Xbin, Ybin=Ybin,
                                      threshold=threshold, gap=gap)
    return np.mean(scores)


def estimate_threshold(fix, I, Xres, Xbin):
    """Threshold = 2*std(saccade amplitude in px) / (Xres/Xbin), per the toolbox's own
    recommended formula (ScanMatch_CreateSubMatrix.m docstring)."""
    fix = np.asarray(fix)
    I = np.asarray(I).reshape(-1)

    amplitudes = []
    for i in range(fix.shape[0]):
        n = int(I[i])
        if n < 2:
            continue
        pts = fix[i, :n]
        diffs = np.diff(pts, axis=0)
        amplitudes.append(np.sqrt((diffs**2).sum(axis=1)))
    amplitudes = np.concatenate(amplitudes) if amplitudes else np.array([0.0])

    return 2 * np.std(amplitudes) / (Xres / Xbin)
