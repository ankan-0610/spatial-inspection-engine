import numpy as np
from scipy.signal import find_peaks
import cv2
from pathlib import Path

from report_rotation_variants import load, fuse_variant_frame, VARIANTS


def y_peaks(d, M=np.eye(3), stride=15, vox=0.02, bin_cm=1.0, cell=0.10):
    K, odo, dfs, cfs = load(d)
    # Build the full fused cloud with the chosen rotation convention.
    P = []
    for i in range(0, len(dfs), stride):
        frame_pts = fuse_variant_frame(K, odo, dfs, cfs, i, M)
        if frame_pts.size > 0:
            P.append(frame_pts)
    if not P:
        print(d)
        print("   no valid points")
        return
    P = np.vstack(P)
    y = P[:, 1]
    bins = np.arange(y.min(), y.max() + 0.01, bin_cm / 100.0)
    h, e = np.histogram(y, bins=bins)
    pk, _ = find_peaks(h, height=0.15 * h.max(), distance=10)

    def area(sel):
        cells = np.unique(np.floor(sel[:, [0, 2]] / cell).astype(int), axis=0)
        return len(cells) * cell * cell

    out = []
    for j in pk:
        yc = 0.5 * (e[j] + e[j + 1])
        sel = P[np.abs(y - yc) < 0.03]
        out.append((yc, len(sel), area(sel)))

    if not out:
        print(d)
        print("   no histogram peaks found")
        return

    floor_y = min(yc for yc, _, _ in out)
    floor_area = next(a for yc, _, a in out if yc == floor_y)
    print(d)
    print("   y_m   points   area_m2   area/floor")
    for yc, n, a in sorted(out):
        print(f"{yc:7.3f} {n:8d} {a:8.1f}  {a/floor_area:7.2f}")

    # show the actual RGB frame(s) around the ceiling-like peaks for manual inspection
    print("   frames to inspect:")
    for yc, _, _ in sorted(out):
        # pick the frame with world y nearest to this peak among sampled frames
        idx = np.argmin(np.abs(odo[::stride, 3] - yc))
        frame_idx = idx * stride
        if frame_idx >= len(dfs):
            continue
        print(f"     y={yc:6.3f} -> depth frame index {frame_idx} (camera y approx {odo[frame_idx, 3]:.3f})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python y_peaks_report.py <dataset_dir> [<dataset_dir> ...]")
        raise SystemExit(1)
    for ds in sys.argv[1:]:
        y_peaks(ds)
