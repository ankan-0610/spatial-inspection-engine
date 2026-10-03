import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R

FLIP = np.diag([1.0, -1.0, -1.0])


def load(d):
    d = Path(d).expanduser()
    K = np.loadtxt(d / "camera_matrix.csv", delimiter=",")
    odo = np.loadtxt(d / "odometry.csv", delimiter=",", skiprows=1, usecols=range(13))
    return K, odo, sorted((d / "depth").glob("*.png")), sorted((d / "confidence").glob("*.png"))


def world_pts(K, odo, dfs, cfs, i, rgb_w=1920, max_depth=5.0):
    z = cv2.imread(str(dfs[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    c = cv2.imread(str(cfs[i]), cv2.IMREAD_UNCHANGED)
    h, w = z.shape
    s = w / rgb_w
    fx, fy, cx, cy = K[0, 0] * s, K[1, 1] * s, K[0, 2] * s, K[1, 2] * s
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    m = (c == 2) & (z > 0.2) & (z < max_depth)
    zz = z[m]
    pc = np.stack([(u[m] - cx) * zz / fx, (v[m] - cy) * zz / fy, zz], axis=1)

    Rw = R.from_quat(odo[i, 5:9]).as_matrix() @ FLIP
    world = (pc @ Rw.T + odo[i, 2:5]).astype(np.float32)
    return pc, world


def best_plane(P, dist=0.01):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    (a, b, c, d), idx = pc.segment_plane(dist, 3, 1500)
    n = np.array([a, b, c])
    if len(idx) == 0:
        return n, d, 0.0, float("inf")
    res = P[idx] @ n + d
    return n, d, len(idx) / len(P), res.std() * 100.0


def floor_track(dir, chunk=150, step=5, bin_cm=1.0):
    K, odo, dfs, cfs = load(dir)
    print(f"dataset: {Path(dir).name}")
    print("start  cam_y   floor_y  cam_h  peak_mass  slab_tilt_deg")
    for s0 in range(0, len(dfs) - chunk, chunk):
        idx = list(range(s0, s0 + chunk, step))
        pts = []
        for i in idx:
            _, P = world_pts(K, odo, dfs, cfs, min(max(i + 0, 0), len(dfs) - 1))
            pts.append(P)
        P = np.vstack(pts)

        cam_y = odo[idx, 3].mean()
        low = P[P[:, 1] < cam_y - 0.5]
        if len(low) < 2000:
            print(f"{s0:5d}  {cam_y:6.2f}   (too few low points)")
            continue

        bins = np.arange(low[:, 1].min(), low[:, 1].max() + 0.01, bin_cm / 100.0)
        h, e = np.histogram(low[:, 1], bins=bins)
        j = int(h.argmax())
        fy = 0.5 * (e[j] + e[j + 1])
        near = low[np.abs(low[:, 1] - fy) < 0.03]
        if len(near) < 200:
            print(f"{s0:5d}  {cam_y:6.2f}   (too few near-floor points)")
            continue

        n, d, frac, sd = best_plane(near, 0.01)
        tilt = np.degrees(np.arccos(min(1.0, abs(n[1]))))
        print(f"{s0:5d}  {cam_y:6.2f}  {fy:7.3f}  {cam_y - fy:5.2f}  {h[j]/len(low):8.2f}  {tilt:6.1f}")


def diagnose(dir, chunk=150, lag=0):
    K, odo, dfs, cfs = load(dir)
    up = np.array([0, 1.0, 0])

    print(f"dataset: {Path(dir).name}")
    print(f"frames: {len(dfs)} | odo rows: {len(odo)}")

    # 1. Single-frame diagnosis: is one frame's dominant plane flat?
    print("\nSingle-frame plane quality:")
    for i in np.linspace(0, len(dfs) - 1, 5, dtype=int):
        pc, _ = world_pts(K, odo, dfs, cfs, i)
        n, d, frac, sd = best_plane(pc)
        print(f"  frame {i:5d}: inlier frac {frac:.2f}, resid std {sd:.2f} cm")

    # 2. Chunk-local diagnosis: does the offset wander over time?
    print("\nChunk-local plane quality:")
    for s0 in range(0, len(dfs) - chunk, chunk):
        pts = []
        for i in range(s0, s0 + chunk, 5):
            j = min(max(i + lag, 0), len(dfs) - 1)
            _, P = world_pts(K, odo, dfs, cfs, j)
            pts.append(P[::7])
        P = np.vstack(pts)
        n, d, frac, sd = best_plane(P)
        ang = np.degrees(np.arccos(min(1.0, abs(n @ up))))
        print(
            f"  chunk {s0:5d}: tilt {ang:4.1f} deg, offset {d:+.3f} m, "
            f"inlier frac {frac:.2f}, resid {sd:.2f} cm"
        )

    print("\nFloor tracking over time:")
    floor_track(dir, chunk=chunk, step=5, bin_cm=1.0)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python diagnose_planes.py <dataset_dir> [lag]")
        raise SystemExit(1)

    dataset = sys.argv[1]
    lag = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    diagnose(dataset, lag=lag)
