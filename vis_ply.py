import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def load_point_cloud(ply_path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd is None or pcd.is_empty():
        raise RuntimeError(f"无法读取点云或点云为空: {ply_path}")
    return pcd


def colorize_by_z(
    pcd: o3d.geometry.PointCloud,
    z_min: float = None,
    z_max: float = None,
    cmap_name: str = "jet",
):
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.shape[0] == 0:
        raise RuntimeError("点云为空，无法着色")

    z = pts[:, 2]

    if z_min is None:
        z_min = float(np.min(z))
    if z_max is None:
        z_max = float(np.max(z))

    if z_max <= z_min:
        z_max = z_min + 1e-6

    z_norm = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(z_norm)[:, :3].astype(np.float64)  # 去掉 alpha
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return z_min, z_max


def add_coordinate_frame(size=1.0):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def main():
    parser = argparse.ArgumentParser(description="加载 PLY 点云，并按 Z 高度着色可视化")
    parser.add_argument("ply_path", type=str, help="输入 PLY 文件路径")
    parser.add_argument(
        "--voxel",
        type=float,
        default=1.0,
        help="体素降采样大小；<=0 表示不降采样",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=None,
        help="手动指定颜色映射的最小 Z",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=None,
        help="手动指定颜色映射的最大 Z",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="jet",
        help="matplotlib colormap 名称，如 jet / viridis / turbo",
    )
    parser.add_argument(
        "--axis-size",
        type=float,
        default=1.0,
        help="坐标轴大小",
    )
    args = parser.parse_args()

    ply_path = Path(args.ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"文件不存在: {ply_path}")

    print(f"[INFO] 读取点云: {ply_path}")
    pcd = load_point_cloud(ply_path)

    n0 = np.asarray(pcd.points).shape[0]
    print(f"[INFO] 原始点数: {n0}")

    if args.voxel is not None and args.voxel > 0:
        print(f"[INFO] 进行体素降采样: voxel = {args.voxel}")
        pcd = pcd.voxel_down_sample(args.voxel)
        n1 = np.asarray(pcd.points).shape[0]
        print(f"[INFO] 降采样后点数: {n1}")

    z_min, z_max = colorize_by_z(
        pcd,
        z_min=args.z_min,
        z_max=args.z_max,
        cmap_name=args.cmap,
    )
    print(f"[INFO] 颜色映射范围: z_min = {z_min:.6f}, z_max = {z_max:.6f}")

    axis = add_coordinate_frame(size=args.axis_size)

    o3d.visualization.draw_geometries(
        [pcd, axis],
        window_name="PLY Colored by Z",
        width=1280,
        height=800,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()

"""
python vis_ply.py yanghaitang/models/pc/0/sampled_recon.ply
"""
