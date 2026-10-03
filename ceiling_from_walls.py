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


def occupied_area(P, cell=0.10):
    if P.size == 0:
        return 0.0
    cells = np.unique(np.floor(P[:, [0, 2]] / cell).astype(int), axis=0)
    return len(cells) * cell * cell


def fit_plane_to_band(P, floor_y, y_min=2.9, y_max=3.1):
    """Fit a plane h = a*x + b*z + c to ceiling points with height h = y - floor_y."""
    if P.size == 0:
        return None
    height = P[:, 1] - floor_y
    band = P[(height >= y_min) & (height <= y_max)]
    if len(band) < 100:
        return None
    A = np.column_stack([band[:, 0], band[:, 2], np.ones(len(band))])
    coeff, *_ = np.linalg.lstsq(A, height[(height >= y_min) & (height <= y_max)], rcond=None)
    a, b, c = coeff
    pred = A @ coeff
    resid = (band[:, 1] - floor_y) - pred
    return {
        "coeff": coeff,
        "slope_x": float(a),
        "slope_z": float(b),
        "intercept": float(c),
        "resid_mean": float(np.mean(resid)),
        "resid_std": float(np.std(resid)),
        "resid_span": float(np.ptp(resid)),
        "tilt": float(np.hypot(a, b)),
        "n": int(len(band)),
    }


def peak_region_summary(P, floor_y, target_h, band=0.04):
    sel = P[np.abs((P[:, 1] - floor_y) - target_h) < band]
    if len(sel) == 0:
        return None
    xz = sel[:, [0, 2]]
    center = xz.mean(axis=0)
    area = occupied_area(sel)
    return {
        "height": float(target_h),
        "pts": int(len(sel)),
        "center": center,
        "xz_min": xz.min(axis=0),
        "xz_max": xz.max(axis=0),
        "area": float(area),
    }


def resolve_ceiling_regions(P, floor_y):
    ceiling = P[(P[:, 1] - floor_y >= 2.9) & (P[:, 1] - floor_y <= 3.1)]
    if len(ceiling) < 200:
        return {
            "status": "insufficient_points",
            "plane": None,
            "regions": [],
        }

    plane = fit_plane_to_band(ceiling, floor_y, y_min=2.9, y_max=3.1)
    region_info = []
    for h in [2.955, 3.065]:
        r = peak_region_summary(P, floor_y, h, band=0.06)
        if r is not None:
            region_info.append(r)

    if len(region_info) >= 2:
        c1 = region_info[0]["center"]
        c2 = region_info[1]["center"]
        sep = np.linalg.norm(c1 - c2)
    else:
        sep = 0.0

    if plane is not None and plane["resid_std"] < 0.18 and sep < 0.8:
        status = "single_tilted_plane"
    elif len(region_info) >= 2 and sep >= 0.8:
        status = "two_regions"
    else:
        status = "uncertain_low_confidence"

    return {"status": status, "plane": plane, "regions": region_info}


def ceiling_candidates(P_room, floor_y, bin_cm=1.0, band=0.04, min_cover=0.30, min_pts=5000, return_all=False):
    floor_y_tol = 0.03
    floor_area = occupied_area(P_room[np.abs(P_room[:, 1] - floor_y) < floor_y_tol])
    if floor_area <= 0:
        return []

    y = P_room[:, 1]
    hi = y[y > floor_y + 1.8]
    if len(hi) < min_pts:
        return []
    bins = np.arange(hi.min(), hi.max() + 0.01, bin_cm / 100)
    h, e = np.histogram(hi, bins=bins)
    peaks, _ = find_peaks(h, height=min_pts / 50, distance=10)

    out = []
    for j in peaks:
        yc = 0.5 * (e[j] + e[j + 1])
        sel = P_room[np.abs(y - yc) < band]
        cover = occupied_area(sel) / floor_area
        cand = dict(height=yc - floor_y, pts=len(sel), cover=cover)
        if len(sel) >= min_pts and cover >= min_cover:
            cand["accepted"] = True
        else:
            cand["accepted"] = False
        out.append(cand)
    if return_all:
        return out
    return [c for c in out if c["accepted"]]


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
    all_candidates = ceiling_candidates(P, floor_y, return_all=True)
    cand = [c for c in all_candidates if c["accepted"]]
    print(d)
    print(f"  floor_y={floor_y:.3f} | room_points={len(room)} | all_candidates={all_candidates}")
    print(f"  accepted={cand}")

    ceiling_eval = resolve_ceiling_regions(P, floor_y)
    plane = ceiling_eval["plane"]
    print("  ceiling band 2.9-3.1m above floor:")
    if plane is None:
        print("    insufficient ceiling points for plane fit")
    else:
        cx = float(room[:, 0].mean())
        cz = float(room[:, 2].mean())
        centroid_h = plane["intercept"] + plane["slope_x"] * cx + plane["slope_z"] * cz
        lo = centroid_h - plane["resid_std"]
        hi = centroid_h + plane["resid_std"]
        print(f"    plane: h = {plane['slope_x']:.4f}*x + {plane['slope_z']:.4f}*z + {plane['intercept']:.4f}")
        print(f"    tilt={plane['tilt']:.4f} | resid_std={plane['resid_std']:.4f} m | resid_span={plane['resid_span']:.4f} m | n={plane['n']}")
        print(f"    room centroid: ({cx:.3f}, {cz:.3f}) | ceiling height at centroid={centroid_h:.3f} m | interval=[{lo:.3f}, {hi:.3f}] m")

    print(f"    top-down map: status={ceiling_eval['status']}")
    for r in ceiling_eval["regions"]:
        print(f"      peak={r['height']:.3f} m: center=({r['center'][0]:.2f}, {r['center'][1]:.2f}) | pts={r['pts']} | area={r['area']:.3f}")
    if ceiling_eval["status"] == "single_tilted_plane":
        print("    interpretation: one tilted ceiling plane with smooth gradient; report the centroid height with spread interval.")
    elif ceiling_eval["status"] == "two_regions":
        print("    interpretation: two separate ceiling regions / room step; report per room.")
    else:
        print("    interpretation: ambiguous; report both peaks with low confidence.")

    print("  ceiling detector uses occupied xz area coverage; threshold tuned on 3 sample scans and kept conservative (min_cover=0.30).")
    return floor_y, cand


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ceiling_from_walls.py <dataset_dir> [<dataset_dir> ...]")
        raise SystemExit(1)
    for ds in sys.argv[1:]:
        room_ceiling_report(ds)
