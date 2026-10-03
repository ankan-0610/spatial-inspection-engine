import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R

FLIP = np.diag([1.0, -1.0, -1.0])

VARIANTS = {
    "I": np.eye(3),
    "x180": np.diag([1.0, -1.0, -1.0]),
    "y180": np.diag([-1.0, 1.0, -1.0]),
    "z180": np.diag([-1.0, -1.0, 1.0]),
}


def load(d):
    d = Path(d).expanduser()
    K = np.loadtxt(d / "camera_matrix.csv", delimiter=",")
    odo = np.loadtxt(d / "odometry.csv", delimiter=",", skiprows=1, usecols=range(13))
    return K, odo, sorted((d / "depth").glob("*.png")), sorted((d / "confidence").glob("*.png"))


def fuse_variant_frame(K, odo, dfs, cfs, i, M=np.eye(3), rgb_w=1920, max_depth=5.0):
    z = cv2.imread(str(dfs[i]), cv2.IMREAD_UNCHANGED)
    if z is None:
        return np.empty((0, 3), dtype=np.float32)
    z = z.astype(np.float32) / 1000.0
    c = cv2.imread(str(cfs[i]), cv2.IMREAD_UNCHANGED)
    if c is None:
        return np.empty((0, 3), dtype=np.float32)
    h, w = z.shape
    s = w / rgb_w
    fx, fy, cx, cy = K[0, 0] * s, K[1, 1] * s, K[0, 2] * s, K[1, 2] * s
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    m = (c == 2) & (z > 0.2) & (z < max_depth)
    zz = z[m]
    if zz.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    p = np.stack([(u[m] - cx) * zz / fx, (v[m] - cy) * zz / fy, zz], axis=1)
    Rw = R.from_quat(odo[i, 5:9]).as_matrix() @ M
    return (p @ Rw.T + odo[i, 2:5]).astype(np.float32)


def fuse_variant(d, M, stride=40, max_depth=5.0, vox=0.05, rgb_w=1920):
    K, odo, dfs, cfs = load(d)
    buf = []
    for i in range(0, len(dfs), stride):
        P = fuse_variant_frame(K, odo, dfs, cfs, i, M, rgb_w=rgb_w, max_depth=max_depth)
        if P.size > 0:
            buf.append(P)
    if not buf:
        return np.empty((0, 3), dtype=np.float32), 0.0
    P = np.vstack(buf)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(vox)
    return np.asarray(pc.points), odo[::stride, 3].mean()


def report(d):
    print(d)
    for name, M in VARIANTS.items():
        P, cam_y = fuse_variant(d, M)
        q = np.percentile(P[:, 1] - cam_y, [1, 10, 50, 90, 99])
        rest = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        top = []
        for _ in range(3):
            if len(rest.points) == 0:
                break
            (a, b, c, dd), idx = rest.segment_plane(0.03, 3, 2000)
            kind = "horiz" if abs(b) > 0.95 else "vert" if abs(b) < 0.1 else "other"
            top.append(f"{kind}:{len(idx) / len(P):.2f}")
            rest = rest.select_by_index(idx, invert=True)
        print(f"  {name:5s} y-rel-cam pctl[1,10,50,90,99]={np.round(q,2)}  top planes={top}")


def floor_ceiling_track(dir, M=None, chunk=300, step=10, bin_cm=1.0):
    if M is None:
        M = np.eye(3)
    K, odo, dfs, cfs = load(dir)
    print("start   cam_y   floor_y  ceil_y   cam_h   ceil_h   floor_mass  ceil_mass")
    for s0 in range(0, len(dfs) - chunk, chunk):
        frames = []
        for i in range(s0, s0 + chunk, step):
            P = fuse_variant_frame(K, odo, dfs, cfs, i, M)
            if P.size > 0:
                frames.append(P)
        if not frames:
            print(f"{s0:5d}  {odo[s0, 3]:6.2f}   n/a   n/a  n/a  n/a      n/a       n/a")
            continue
        P = np.vstack(frames)
        cam_y = odo[s0:s0 + chunk, 3].mean()

        def peak(sel):
            if len(sel) < 2000:
                return None, 0.0
            bins = np.arange(sel.min(), sel.max() + 0.01, bin_cm / 100.0)
            h, e = np.histogram(sel, bins=bins)
            j = int(h.argmax())
            return 0.5 * (e[j] + e[j + 1]), h[j] / len(sel)

        fy, fm = peak(P[P[:, 1] < cam_y - 0.8, 1])
        cy_, cm = peak(P[P[:, 1] > cam_y + 0.8, 1])
        f = f"{fy:7.3f}" if fy is not None else "   n/a "
        c = f"{cy_:7.3f}" if cy_ is not None else "   n/a "
        ch = f"{cam_y - fy:5.2f}" if fy is not None else " n/a "
        hh = f"{cy_ - fy:5.2f}" if (fy is not None and cy_ is not None) else " n/a "
        print(f"{s0:5d}  {cam_y:6.2f}  {f}  {c}  {ch}  {hh}   {fm:8.2f}   {cm:8.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python report_rotation_variants.py <dataset_dir>")
        raise SystemExit(1)
    for ds in sys.argv[1:]:
        report(ds)
