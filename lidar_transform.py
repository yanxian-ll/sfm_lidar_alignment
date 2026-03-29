import argparse
from pathlib import Path

import numpy as np
import laspy

try:
    import open3d as o3d
except Exception:
    o3d = None


def read_transformation_matrix(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    matrix = np.array([list(map(float, line.split())) for line in lines], dtype=np.float64)

    if matrix.shape[0] > 4:
        matrix = matrix[:4, :]

    if matrix.shape != (4, 4):
        raise ValueError(f"变换矩阵尺寸错误，应为 4x4: {file_path}, 当前为 {matrix.shape}")

    return matrix



def compose_transformations(transform_dir):
    """
    按顺序依次应用：
    transform_manual.txt -> transform_icp.txt -> transform_refine.txt

    如果点先做 T1，再做 T2，再做 T3，
    则总矩阵为：T = T3 @ T2 @ T1
    """
    transform_dir = Path(transform_dir)

    t_manual = read_transformation_matrix(transform_dir / "transform_manual.txt")
    t_icp = read_transformation_matrix(transform_dir / "transform_icp.txt")
    t_refine = read_transformation_matrix(transform_dir / "transform_refine.txt")

    T = np.eye(4, dtype=np.float64)

    if t_manual is not None:
        print("transform with pair select")
        T = t_manual @ T

    if t_icp is not None:
        print("transform with ICP")
        T = t_icp @ T

    if t_refine is not None:
        print("transform with 2D feature match")
        T = t_refine @ T

    return T



def has_las_rgb(header):
    dim_names = set(header.point_format.dimension_names)
    return {"red", "green", "blue"}.issubset(dim_names)



def convert_las_rgb_to_u8(r, g, b):
    """
    LAS 里 RGB 常见是 uint16（0~65535），这里统一转成 uint8（0~255）
    """
    r = np.asarray(r)
    g = np.asarray(g)
    b = np.asarray(b)

    if r.dtype == np.uint8 and g.dtype == np.uint8 and b.dtype == np.uint8:
        return r, g, b

    max_val = max(
        np.max(r) if r.size > 0 else 0,
        np.max(g) if g.size > 0 else 0,
        np.max(b) if b.size > 0 else 0,
    )

    if max_val <= 255:
        return r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)

    r8 = np.clip(r / 256.0, 0, 255).astype(np.uint8)
    g8 = np.clip(g / 256.0, 0, 255).astype(np.uint8)
    b8 = np.clip(b / 256.0, 0, 255).astype(np.uint8)
    return r8, g8, b8



def convert_o3d_rgb_to_u8(colors, default_color=(180, 180, 180)):
    colors = np.asarray(colors)
    if colors.size == 0:
        return None
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("PLY 颜色维度异常，应为 Nx3")

    if np.issubdtype(colors.dtype, np.integer):
        max_val = int(colors.max()) if colors.size > 0 else 0
        if max_val <= 255:
            return np.clip(colors, 0, 255).astype(np.uint8)
        return np.clip(colors / 256.0, 0, 255).astype(np.uint8)

    max_val = float(colors.max()) if colors.size > 0 else 0.0
    if max_val <= 1.0 + 1e-6:
        return np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    if max_val <= 255.0 + 1e-6:
        return np.clip(colors, 0, 255).astype(np.uint8)
    return np.clip(colors / 256.0, 0, 255).astype(np.uint8)



def write_binary_ply_header(f, num_points, with_color=True):
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if with_color:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    header.append("end_header\n")
    f.write("\n".join(header).encode("ascii"))



def transform_xyz_chunk(x, y, z, T):
    xyz = np.stack([x, y, z], axis=1).astype(np.float64)
    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    xyz_h = np.concatenate([xyz, ones], axis=1)
    xyz_t = xyz_h @ T.T
    return xyz_t[:, 0].astype(np.float32), xyz_t[:, 1].astype(np.float32), xyz_t[:, 2].astype(np.float32)



def write_points_to_binary_ply(
    out_ply_path,
    x,
    y,
    z,
    rgb=None,
    default_color=(180, 180, 180),
):
    out_ply_path = Path(out_ply_path)
    out_ply_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(x)
    with open(out_ply_path, "wb") as f:
        write_binary_ply_header(f, n, with_color=True)

        if rgb is None:
            r = np.full((n,), default_color[0], dtype=np.uint8)
            g = np.full((n,), default_color[1], dtype=np.uint8)
            b = np.full((n,), default_color[2], dtype=np.uint8)
        else:
            rgb = np.asarray(rgb)
            if rgb.shape != (n, 3):
                raise ValueError(f"rgb 尺寸应为 ({n}, 3)，当前为 {rgb.shape}")
            r = rgb[:, 0].astype(np.uint8)
            g = rgb[:, 1].astype(np.uint8)
            b = rgb[:, 2].astype(np.uint8)

        ply_dtype = np.dtype([
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ])
        arr = np.empty(n, dtype=ply_dtype)
        arr["x"] = np.asarray(x, dtype=np.float32)
        arr["y"] = np.asarray(y, dtype=np.float32)
        arr["z"] = np.asarray(z, dtype=np.float32)
        arr["red"] = r
        arr["green"] = g
        arr["blue"] = b
        arr.tofile(f)



def save_las_to_ply_chunked(
    las_path,
    out_ply_path,
    T,
    chunk_size=2_000_000,
    default_color=(180, 180, 180),
):
    las_path = Path(las_path)
    out_ply_path = Path(out_ply_path)
    out_ply_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"reading las header: {las_path}")
    with laspy.open(str(las_path)) as reader:
        total_points = reader.header.point_count
        use_rgb = has_las_rgb(reader.header)

        print(f"total points: {total_points}")
        print(f"has rgb: {use_rgb}")
        print(f"writing ply: {out_ply_path}")

        with open(out_ply_path, "wb") as f:
            write_binary_ply_header(f, total_points, with_color=True)

            processed = 0
            chunk_id = 0

            for chunk in reader.chunk_iterator(chunk_size):
                chunk_id += 1

                x = np.asarray(chunk.x, dtype=np.float64)
                y = np.asarray(chunk.y, dtype=np.float64)
                z = np.asarray(chunk.z, dtype=np.float64)

                xt, yt, zt = transform_xyz_chunk(x, y, z, T)
                n = xt.shape[0]

                if use_rgb:
                    r = np.asarray(chunk.red)
                    g = np.asarray(chunk.green)
                    b = np.asarray(chunk.blue)
                    r, g, b = convert_las_rgb_to_u8(r, g, b)
                else:
                    r = np.full((n,), default_color[0], dtype=np.uint8)
                    g = np.full((n,), default_color[1], dtype=np.uint8)
                    b = np.full((n,), default_color[2], dtype=np.uint8)

                ply_dtype = np.dtype([
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ])

                arr = np.empty(n, dtype=ply_dtype)
                arr["x"] = xt
                arr["y"] = yt
                arr["z"] = zt
                arr["red"] = r
                arr["green"] = g
                arr["blue"] = b
                arr.tofile(f)

                processed += n
                print(f"[chunk {chunk_id}] processed {processed}/{total_points}")

    print("transformation done!")
    print(f"saving accomplished, file path: {out_ply_path}")



def save_ply_to_ply(
    ply_path,
    out_ply_path,
    T,
    default_color=(180, 180, 180),
):
    ply_path = Path(ply_path)
    out_ply_path = Path(out_ply_path)
    out_ply_path.parent.mkdir(parents=True, exist_ok=True)

    if o3d is None:
        raise RuntimeError("读取 PLY 需要 open3d，但当前环境未安装 open3d")

    print(f"reading ply: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd is None or pcd.is_empty():
        raise RuntimeError(f"无法读取点云或点云为空: {ply_path}")

    points = np.asarray(pcd.points, dtype=np.float64)
    print(f"total points: {points.shape[0]}")
    print(f"has rgb: {pcd.has_colors()}")
    print(f"writing ply: {out_ply_path}")

    xt, yt, zt = transform_xyz_chunk(points[:, 0], points[:, 1], points[:, 2], T)

    rgb = None
    if pcd.has_colors():
        rgb = convert_o3d_rgb_to_u8(np.asarray(pcd.colors), default_color=default_color)

    write_points_to_binary_ply(
        out_ply_path=out_ply_path,
        x=xt,
        y=yt,
        z=zt,
        rgb=rgb,
        default_color=default_color,
    )

    print("transformation done!")
    print(f"saving accomplished, file path: {out_ply_path}")



def save_point_cloud_to_ply(
    cloud_path,
    out_ply_path,
    T,
    chunk_size=2_000_000,
    default_color=(180, 180, 180),
):
    cloud_path = Path(cloud_path)
    suffix = cloud_path.suffix.lower()

    if suffix in [".las", ".laz"]:
        save_las_to_ply_chunked(
            las_path=cloud_path,
            out_ply_path=out_ply_path,
            T=T,
            chunk_size=chunk_size,
            default_color=default_color,
        )
        return

    if suffix == ".ply":
        save_ply_to_ply(
            ply_path=cloud_path,
            out_ply_path=out_ply_path,
            T=T,
            default_color=default_color,
        )
        return

    raise ValueError(f"暂不支持的输入格式: {cloud_path.suffix}，当前支持 .las / .laz / .ply")



def main():
    parser = argparse.ArgumentParser(description="transform point cloud (LAS/LAZ/PLY) to PLY")
    parser.add_argument("--lidar", type=str, required=True, help="输入点云路径，支持 LAS/LAZ/PLY")
    parser.add_argument("--transform", type=str, required=True, help="变换矩阵 txt 所在目录")
    parser.add_argument("--out_lidar_name", type=str, default="lidar_full.ply", help="输出 ply 文件名")
    parser.add_argument("--chunk_size", type=int, default=100_000_000, help="LAS/LAZ 分块大小；PLY 分支不使用")
    parser.add_argument(
        "--default_color",
        type=int,
        nargs=3,
        default=[180, 180, 180],
        help="输入点云无颜色时使用的默认 RGB",
    )
    args = parser.parse_args()

    T = compose_transformations(args.transform)
    out_ply_path = Path(args.transform) / args.out_lidar_name

    save_point_cloud_to_ply(
        cloud_path=args.lidar,
        out_ply_path=out_ply_path,
        T=T,
        chunk_size=args.chunk_size,
        default_color=tuple(args.default_color),
    )


if __name__ == "__main__":
    main()
    

"""
# 将原始激光粗对齐SFM，保存为 lidar_full.ply

python lidar_transform.py --lidar nanfang/cloud_merged.las \
    --transform nanfang/transform \
    --out_lidar_name lidar_full.ply

python lidar_transform.py --lidar yanghaitang/cloud_merged.las \
    --transform yanghaitang/transform \
    --out_lidar_name lidar_full.ply

python lidar_transform.py --lidar xiaoxiang/cloud_merged.las \
    --transform xiaoxiang/transform \
    --out_lidar_name lidar_full.ply
"""


"""
# 将原始激光精对齐SFM，保存为 lidar_finall.ply

python lidar_transform.py --lidar nanfang/cloud_merged.las \
    --transform nanfang/transform \
    --out_lidar_name lidar_finall.ply

python lidar_transform.py --lidar yanghaitang/cloud_merged.las \
    --transform yanghaitang/transform \
    --out_lidar_name lidar_finall.ply

python lidar_transform.py --lidar xiaoxiang/cloud_merged.las \
    --transform xiaoxiang/transform \
    --out_lidar_name lidar_finall.ply
"""

