import numpy as np, cv2
from pathlib import Path

def inspect(d):
    d = Path(d)
    K = np.loadtxt(d/"camera_matrix.csv", delimiter=",")
    # The odometry CSV has two trailing blank columns in each row; read only the
    # numeric fields that actually contain pose data.
    odo = np.loadtxt(d/"odometry.csv", delimiter=",", skiprows=1, usecols=range(13))
    dfiles = sorted((d/"depth").glob("*.png"))
    cfiles = sorted((d/"confidence").glob("*.png"))
    cap = cv2.VideoCapture(str(d/"rgb.mp4"))
    W, H = int(cap.get(3)), int(cap.get(4))
    n_rgb, fps = int(cap.get(7)), cap.get(5)
    z = cv2.imread(str(dfiles[len(dfiles)//2]), cv2.IMREAD_UNCHANGED)
    c = cv2.imread(str(cfiles[len(cfiles)//2]), cv2.IMREAD_UNCHANGED)
    print(d.name)
    print(" K:\n", K)
    print(" odometry rows:", len(odo), "| depth:", len(dfiles), "| conf:", len(cfiles),
          "| rgb frames:", n_rgb, "@", fps, "fps | rgb size:", (W, H))
    print(" depth:", z.shape, z.dtype, "range(mm):", z[z>0].min(), z.max())
    print(" conf values:", np.unique(c, return_counts=True))
    print(" odo first rows:\n", odo[:2])

for s in ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
    inspect(f"~/Downloads/{s}".replace("~", str(Path.home())))
