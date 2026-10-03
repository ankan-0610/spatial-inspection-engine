import numpy as np
from scipy.signal import find_peaks
import open3d as o3d

from report_rotation_variants import load, fuse_variant_frame


def wall_room_segment(P, floor_y, wall_dist=0.15, z_eps=0.10):
    """Return points in the room, excluding the floor slab and near-wall thin shell."""
    if P.size == 0:
        return P
    # Keep points above the floor by a small margin and remove the wall shell.
    keep = (P[:, 1] > floor_y + 0.15) & (np.abs(P[:, 0]) < 3.0) & (np.abs(P[:, 2]) < 3.0)
    room = P[keep]
    if len(room) == 0:
        return room
    # Remove the near-wall shell in xz, computed from the room footprint.
    xz = room[:, [0, 2]]
    x_min = xz[:, 0].min()
    x_max = xz[:, 0].max()
    z_min = xz[:, 1].min()
    z_max = xz[:, 1].max()
    shell = ((xz[:, 0] < x_min + wall_dist) |
             (xz[:, 0] > x_max - wall_dist) |
             (xz[:, 1] < z_min + wall_dist) |
             (xz[:, 1] > z_max - wall_dist))
    return room[~shell]


def ceiling_candidates(P_room, floor_y, bin_cm=1.0, min_mass=0.05):
    y = P_room[:, 1]
    hi = y[y > floor_y + 1.8]
    if len(hi) < 3000:
        return []
    bins = np.arange(hi.min(), hi.max() + 0.01, bin_cm / 100)
    h, e = np.histogram(hi, bins=bins)
    pk, _ = find_peaks(h, height=min_mass * h.max(), distance=10)
    out = []
    for j in pk:
        yc = 0.5 * (e[j] + e[j + 1])
        out.append((yc - floor_y, int(h[j])))
    return out


def room_ceiling_report(d, M=np.eye(3), stride=15, max_depth=5.0):
    K, odo, dfs, cfs = load(d)
    points = []
    for i in range(0, len(dfs), stride):
        P = fuse_variant_frame(K, odo, dfs, cfs, i, M, max_depth=max_depth)
        if P.size > 0:
            points.append(P)
    if not points:
        print(d)
        print("  no points")
        return
    P = np.vstack(points)

    # Use the lowest large peak as the floor reference.
    y = P[:, 1]
    bins = np.arange(y.min(), y.max() + 0.01, 0.01)
    h, e = np.histogram(y, bins=bins)
    pk, _ = find_peaks(h, height=0.1 * h.max(), distance=10)
    floor_y = min(0.5 * (e[j] + e[j + 1]) for j in pk) if len(pk) else float(np.min(y))
    room = wall_room_segment(P, floor_y)
    cand = ceiling_candidates(room, floor_y)
    print(d)
    print(f"  floor_y={floor_y:.3f} | room_points={len(room)} | candidates={cand}")
    return floor_y, cand


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ceiling_from_walls.py <dataset_dir> [<dataset_dir> ...]")
        raise SystemExit(1)
    for ds in sys.argv[1:]:
        room_ceiling_report(ds)
