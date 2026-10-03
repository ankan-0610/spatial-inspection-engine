import numpy as np, cv2, open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R

FLIP = np.diag([1.0, -1.0, -1.0])      # ARKit cam (x right, y up, z back) -> OpenCV cam

def load(d):
    d = Path(d).expanduser()
    K = np.loadtxt(d/"camera_matrix.csv", delimiter=",")
    # The odometry CSV contains two trailing blank columns. Load only the
    # numeric pose fields so NumPy does not choke on empty strings.
    odo = np.loadtxt(d/"odometry.csv", delimiter=",", skiprows=1, usecols=range(13))
    return K, odo, sorted((d/"depth").glob("*.png")), sorted((d/"confidence").glob("*.png"))

def fuse(d, stride=20, max_depth=5.0, vox=0.03, rgb_w=1920):
    K, odo, dfs, cfs = load(d)
    h, w = cv2.imread(str(dfs[0]), cv2.IMREAD_UNCHANGED).shape
    s = w / rgb_w
    fx, fy, cx, cy = K[0,0]*s, K[1,1]*s, K[0,2]*s, K[1,2]*s
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    acc, buf = o3d.geometry.PointCloud(), []
    def flush():
        nonlocal acc, buf
        if not buf: return
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(buf)))
        acc = (acc + p).voxel_down_sample(vox); buf = []
    for n, i in enumerate(range(0, len(dfs), stride)):
        z = cv2.imread(str(dfs[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        c = cv2.imread(str(cfs[i]), cv2.IMREAD_UNCHANGED)
        m = (c == 2) & (z > 0.2) & (z < max_depth)
        zz = z[m]
        p = np.stack([(u[m]-cx)*zz/fx, (v[m]-cy)*zz/fy, zz], 1)
        t, q = odo[i, 2:5], odo[i, 5:9]
        Rw = R.from_quat(q).as_matrix() @ FLIP
        buf.append((p @ Rw.T + t).astype(np.float32))
        if n % 40 == 39: flush()
    flush()
    return acc

def planes(pcd, k=8, dist=0.02, min_inl=3000):
    up = np.array([0, 1, 0.0]); out = []
    rest = pcd
    for _ in range(k):
        if len(rest.points) < min_inl: break
        (a, b, c, dd), idx = rest.segment_plane(dist, 3, 2000)
        if len(idx) < min_inl: break
        n = np.array([a, b, c]); inl = rest.select_by_index(idx)
        res = np.asarray(inl.points) @ n + dd
        kind = "horiz" if abs(n @ up) > 0.95 else "vert" if abs(n @ up) < 0.1 else "other"
        out.append(dict(kind=kind, n=n.round(3), d=round(dd, 3), inliers=len(idx),
                        resid_std_cm=round(res.std()*100, 2)))
        rest = rest.select_by_index(idx, invert=True)
    return out

if __name__ == "__main__":
    import sys
    pcd = fuse(sys.argv[1])
    print("points:", len(pcd.points))
    for p in planes(pcd): print(p)