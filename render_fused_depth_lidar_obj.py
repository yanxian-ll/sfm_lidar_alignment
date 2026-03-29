# -*- coding: utf-8 -*-
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # 必须放在 import cv2 之前

import cv2
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

import numpy as np
import open3d as o3d
from tqdm import tqdm

# =========================
# IO
# =========================
def load_point_cloud(ply_path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd is None or pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {ply_path}")
    return pcd


def find_all_plys(ply_root: Path):
    if not ply_root.exists():
        raise FileNotFoundError(f"输入路径不存在: {ply_root}")

    if ply_root.is_file():
        if ply_root.suffix.lower() != ".ply":
            raise ValueError(f"输入文件不是 ply: {ply_root}")
        return [ply_root]

    plys = sorted(ply_root.glob("*.ply"))
    if len(plys) == 0:
        raise FileNotFoundError(f"目录下没有找到 ply 文件: {ply_root}")

    return plys


def resolve_cams_dir(cam_root: Path):
    if cam_root.name == "cams" and cam_root.is_dir():
        return cam_root

    cams_dir = cam_root / "cams"
    if cams_dir.is_dir():
        return cams_dir

    raise FileNotFoundError(f"未找到 cams 目录: {cam_root} 或 {cams_dir}")


def find_cam_txts(cams_dir: Path):
    txts = sorted(cams_dir.glob("*.txt"))
    if len(txts) == 0:
        raise FileNotFoundError(f"cams 下没有找到 txt 文件: {cams_dir}")
    return txts


def read_view_list_txt(view_list_txt: str, pair_list_txt: str):
    names = []

    if view_list_txt is not None:
        path = Path(view_list_txt)
        if not path.exists():
            raise FileNotFoundError(f"视角名单文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    names.append(name)

    if pair_list_txt is not None:
        path = Path(pair_list_txt)
        if not path.exists():
            raise FileNotFoundError(f"视角名单文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                pair = line.strip()
                if not pair:
                    continue
                ss = pair.split()
                if len(ss) >= 2:
                    names.append(ss[0].strip())
                    names.append(ss[1].strip())

    if len(names) == 0:
        return None

    return list(set(names))


def filter_cam_txts_by_view_list(cam_txts, selected_names):
    if selected_names is None:
        return cam_txts

    stem_map = {p.stem: p for p in cam_txts}
    name_map = {p.name: p for p in cam_txts}

    selected = []
    missing = []

    for x in selected_names:
        if x in stem_map:
            selected.append(stem_map[x])
        elif x in name_map:
            selected.append(name_map[x])
        elif x.endswith(".txt") and x[:-4] in stem_map:
            selected.append(stem_map[x[:-4]])
        elif (x + ".txt") in name_map:
            selected.append(name_map[x + ".txt"])
        else:
            missing.append(x)

    if len(missing) > 0:
        print("[WARN] 以下视角在 cams 目录中未找到：")
        for x in missing:
            print(f"       {x}")

    if len(selected) == 0:
        raise RuntimeError("根据 view list 过滤后，没有任何有效视角。")

    return selected


def resolve_input_jpg(image_root: Path, name: str) -> Path:
    candidates = [
        image_root / f"{name}.jpg",
        image_root / f"{name}.jpeg",
        image_root / f"{name}.png",
        image_root / f"{name}.JPG",
        image_root / f"{name}.JPEG",
        image_root / f"{name}.PNG",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到原始图像: {name}.jpg/.jpeg/.png in {image_root}")


def read_and_resize_image_to_cam(image_path: Path, out_w: int, out_h: int):
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"无法读取图像: {image_path}")

    in_h, in_w = img_bgr.shape[:2]
    if in_w == out_w and in_h == out_h:
        return img_bgr

    resized = cv2.resize(img_bgr, (int(out_w), int(out_h)), interpolation=cv2.INTER_AREA)
    return resized


def save_png_image(out_path: Path, img_bgr):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), img_bgr)
    if not ok:
        raise RuntimeError(f"保存 PNG 失败: {out_path}")


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_image_index(image_root: Path):
    """
    预扫描 image_root，只建立一次 stem -> path 映射，避免每个视角重复猜扩展名。
    """
    if image_root is None:
        return None

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    index = {}
    for p in image_root.iterdir():
        if not p.is_file():
            continue
        if p.suffix not in exts:
            continue
        stem = p.stem
        # 优先保留第一次出现的，避免重复覆盖
        if stem not in index:
            index[stem] = p
    return index


def resolve_input_image_from_index(image_index: dict, image_root: Path, name: str) -> Path:
    if image_index is not None and name in image_index:
        return image_index[name]
    return resolve_input_jpg(image_root, name)


# =========================
# Camera TXT Parser
# =========================
def parse_cam_txt(cam_txt_path: Path):
    with open(cam_txt_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines()]

    lines = [x for x in raw_lines if x != ""]

    def find_line_idx(prefix):
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                return i
        return -1

    ext_idx = find_line_idx("extrinsic")
    intr_idx = find_line_idx("intrinsic")
    hwfov_idx = find_line_idx("h w fov")

    if ext_idx < 0 or intr_idx < 0 or hwfov_idx < 0:
        raise ValueError(f"相机文件格式错误: {cam_txt_path}")

    ext_lines = lines[ext_idx + 1: ext_idx + 5]
    intr_lines = lines[intr_idx + 1: intr_idx + 4]
    hwfov_line = lines[hwfov_idx + 1]

    if len(ext_lines) != 4:
        raise ValueError(f"extrinsic 行数不正确: {cam_txt_path}")
    if len(intr_lines) != 3:
        raise ValueError(f"intrinsic 行数不正确: {cam_txt_path}")

    extrinsic = np.array(
        [[float(v) for v in line.split()] for line in ext_lines],
        dtype=np.float64,
    )
    intrinsic = np.array(
        [[float(v) for v in line.split()] for line in intr_lines],
        dtype=np.float64,
    )

    parts = hwfov_line.split()
    if len(parts) != 3:
        raise ValueError(f"h w fov 行格式错误: {cam_txt_path}")

    h = int(float(parts[0]))
    w = int(float(parts[1]))
    fov = float(parts[2])

    return {
        "name": cam_txt_path.stem,
        "path": str(cam_txt_path),
        "extrinsic": extrinsic,   # world -> camera
        "intrinsic": intrinsic,
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "h": h,
        "w": w,
        "fov": fov,
    }


def scale_camera(cam: dict, max_image_edge: int):
    if max_image_edge is None or max_image_edge <= 0:
        return cam

    h = int(cam["h"])
    w = int(cam["w"])
    max_edge = max(h, w)

    if max_edge <= max_image_edge:
        return cam

    scale = float(max_image_edge) / float(max_edge)

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    new_intrinsic = cam["intrinsic"].copy().astype(np.float64)
    new_intrinsic[0, 0] *= scale
    new_intrinsic[1, 1] *= scale
    new_intrinsic[0, 2] *= scale
    new_intrinsic[1, 2] *= scale

    cam_scaled = {
        "name": cam["name"],
        "path": cam["path"],
        "extrinsic": cam["extrinsic"].copy(),
        "intrinsic": new_intrinsic,
        "fx": float(new_intrinsic[0, 0]),
        "fy": float(new_intrinsic[1, 1]),
        "cx": float(new_intrinsic[0, 2]),
        "cy": float(new_intrinsic[1, 2]),
        "h": new_h,
        "w": new_w,
        "fov": cam["fov"],
    }
    return cam_scaled


def save_cam_txt(cam: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ext = cam["extrinsic"]
    intr = cam["intrinsic"]
    h = cam["h"]
    w = cam["w"]
    fov = cam["fov"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("extrinsic opencv(x Right, y Down, z Forward) world2camera\n")
        for i in range(4):
            f.write(f"{ext[i,0]:.12f} {ext[i,1]:.12f} {ext[i,2]:.12f} {ext[i,3]:.12f}\n")
        f.write("\n")

        f.write("intrinsic: fx fy cx cy (pixel)\n")
        for i in range(3):
            f.write(f"{intr[i,0]:.12f} {intr[i,1]:.12f} {intr[i,2]:.12f}\n")
        f.write("\n")

        f.write("h w fov\n")
        f.write(f"{h} {w} {fov:.12f}\n")


# =========================
# Open3D Visualizer Cache
# =========================
class VisCache:
    def __init__(self, point_size=2.0, visible=False):
        self.cache = {}
        self.point_size = float(point_size)
        self.visible = bool(visible)

    def get(self, w: int, h: int):
        key = (int(w), int(h))
        if key in self.cache:
            return self.cache[key]["vis"], key

        vis = o3d.visualization.Visualizer()
        ok = vis.create_window(
            window_name=f"o3d_{w}x{h}",
            width=int(w),
            height=int(h),
            visible=self.visible,
        )
        if not ok:
            raise RuntimeError(
                "Open3D Visualizer create_window failed.\n"
                "Linux 无显示环境请尝试 xvfb-run；Windows 请确保有图形界面。"
            )

        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        opt.point_size = self.point_size

        self.cache[key] = {
            "vis": vis,
            "geom_tag": None,
        }
        return vis, key

    def ensure_geometry(self, key, pcd: o3d.geometry.PointCloud, geom_tag):
        item = self.cache[key]
        if item["geom_tag"] == geom_tag:
            return

        vis = item["vis"]
        vis.clear_geometries()
        vis.add_geometry(pcd, reset_bounding_box=True)
        item["geom_tag"] = geom_tag

    def close_all(self):
        for v in self.cache.values():
            try:
                v["vis"].destroy_window()
            except Exception:
                pass
        self.cache.clear()


# =========================
# Camera / Render
# =========================
def _prepare_one_camera_worker(
    idx,
    cam_txt,
    max_image_edge,
    cam_out_dir,
    image_root,
    images_dir,
    image_index,
):
    cam = parse_cam_txt(cam_txt)
    cam = scale_camera(cam, max_image_edge)

    out_cam_path = cam_out_dir / f"{cam['name']}.txt"
    if not out_cam_path.exists():
        save_cam_txt(cam, out_cam_path)

    if image_root is not None and images_dir is not None:
        out_img_path = images_dir / f"{cam['name']}.png"
        if not out_img_path.exists():
            src_img_path = resolve_input_image_from_index(image_index, image_root, cam["name"])

            # 小优化：
            # 如果原图本来就是 PNG，且尺寸一致，可以直接复制，避免 decode+resize+encode
            direct_copy = False
            if src_img_path.suffix.lower() == ".png":
                img = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"无法读取图像: {src_img_path}")
                in_h, in_w = img.shape[:2]
                if in_w == cam["w"] and in_h == cam["h"]:
                    shutil.copy2(str(src_img_path), str(out_img_path))
                    direct_copy = True

            if not direct_copy:
                img_bgr = read_and_resize_image_to_cam(src_img_path, cam["w"], cam["h"])
                save_png_image(out_img_path, img_bgr)

    return idx, cam


def prepare_cameras(
    cam_txts,
    max_image_edge: int,
    cam_out_dir: Path,
    image_root: Path = None,
    images_dir: Path = None,
    prepare_workers: int = 0,
):
    """
    加速版：
    - 并行解析/缩放/写 cam
    - 并行读图/resize/写 png
    - 保持返回 cams 的顺序与 cam_txts 一致
    """
    if len(cam_txts) == 0:
        return []

    cam_out_dir.mkdir(parents=True, exist_ok=True)
    if images_dir is not None:
        images_dir.mkdir(parents=True, exist_ok=True)

    image_index = build_image_index(image_root) if image_root is not None else None

    if prepare_workers is None or prepare_workers <= 0:
        # IO + OpenCV 混合任务，线程数不宜过大
        prepare_workers = min(16, max(1, (os.cpu_count() or 8)))

    # 视角太少时，串行反而更简单
    if len(cam_txts) <= 4 or prepare_workers == 1:
        cams = []
        for idx, cam_txt in enumerate(tqdm(cam_txts, desc="Prepare Cameras")):
            _, cam = _prepare_one_camera_worker(
                idx=idx,
                cam_txt=cam_txt,
                max_image_edge=max_image_edge,
                cam_out_dir=cam_out_dir,
                image_root=image_root,
                images_dir=images_dir,
                image_index=image_index,
            )
            cams.append(cam)
        return cams

    results = [None] * len(cam_txts)

    with ThreadPoolExecutor(max_workers=prepare_workers) as ex:
        futures = []
        for idx, cam_txt in enumerate(cam_txts):
            fut = ex.submit(
                _prepare_one_camera_worker,
                idx,
                cam_txt,
                max_image_edge,
                cam_out_dir,
                image_root,
                images_dir,
                image_index,
            )
            futures.append(fut)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Prepare Cameras"):
            idx, cam = fut.result()
            results[idx] = cam

    return results


def set_camera_from_world2cam(
    vis: o3d.visualization.Visualizer,
    K: np.ndarray,
    world2cam: np.ndarray,
    w: int,
    h: int,
):
    params = o3d.camera.PinholeCameraParameters()

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=int(w),
        height=int(h),
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
    )
    params.intrinsic = intrinsic
    params.extrinsic = world2cam.astype(np.float64)

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)


def render_depth_only(vis: o3d.visualization.Visualizer):
    vis.poll_events()
    vis.update_renderer()
    depth_f = np.asarray(
        vis.capture_depth_float_buffer(do_render=True),
        dtype=np.float32,
    )
    return depth_f.astype(np.float32)


# =========================
# Depth / Mask Utils
# =========================
def is_valid_depth(depth):
    return np.isfinite(depth) & (depth > 1e-8)


def fuse_depth_keep_nearest(prev_depth: np.ndarray, curr_depth: np.ndarray):
    if prev_depth is None:
        return curr_depth.astype(np.float32)

    if prev_depth.shape != curr_depth.shape:
        raise ValueError(f"深度图尺寸不一致: prev={prev_depth.shape}, curr={curr_depth.shape}")

    prev_valid = is_valid_depth(prev_depth)
    curr_valid = is_valid_depth(curr_depth)

    fused = np.zeros_like(curr_depth, dtype=np.float32)

    only_prev = prev_valid & (~curr_valid)
    only_curr = (~prev_valid) & curr_valid
    both_valid = prev_valid & curr_valid

    fused[only_prev] = prev_depth[only_prev]
    fused[only_curr] = curr_depth[only_curr]
    fused[both_valid] = np.minimum(prev_depth[both_valid], curr_depth[both_valid])

    return fused.astype(np.float32)


def make_depth_vis(depth, invalid_color=(0, 0, 0), percentile_min=1.0, percentile_max=99.0):
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 1e-8)

    vis = np.zeros((h, w, 3), dtype=np.uint8)
    if not np.any(valid):
        return vis

    d = depth[valid]
    dmin = np.percentile(d, percentile_min)
    dmax = np.percentile(d, percentile_max)
    if dmax <= dmin:
        dmax = dmin + 1e-6

    depth_norm = np.zeros_like(depth, dtype=np.float32)
    depth_norm[valid] = np.clip((depth[valid] - dmin) / (dmax - dmin), 0.0, 1.0)

    depth_inv = np.zeros_like(depth_norm, dtype=np.float32)
    depth_inv[valid] = 1.0 - depth_norm[valid]

    depth_u8 = np.zeros((h, w), dtype=np.uint8)
    depth_u8[valid] = np.clip(depth_inv[valid] * 255.0, 0, 255).astype(np.uint8)

    vis_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    vis[~valid] = np.array(invalid_color, dtype=np.uint8)
    return vis


def save_depth_npy(out_path: Path, depth):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), depth.astype(np.float32))


def save_depth_exr(out_path: Path, depth):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), depth.astype(np.float32))
    if not ok:
        raise RuntimeError(f"保存 EXR 失败: {out_path}")


def load_depth_npy(in_path: Path):
    if not in_path.exists():
        return None
    return np.load(str(in_path)).astype(np.float32)


def load_depth_exr(in_path: Path):
    if not in_path.exists():
        return None
    depth = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"读取 EXR 失败: {in_path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32)


def save_depth_vis_png(out_path: Path, vis_rgb):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(out_path), vis_bgr)
    if not ok:
        raise RuntimeError(f"保存深度可视化失败: {out_path}")


def save_mask_png(out_path: Path, mask_bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    ok = cv2.imwrite(str(out_path), mask_u8)
    if not ok:
        raise RuntimeError(f"保存 mask 失败: {out_path}")


def resolve_mask_file(mask_root: Path, name: str):
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".npy", ".PNG", ".JPG", ".JPEG"]
    for ext in exts:
        p = mask_root / f"{name}{ext}"
        if p.exists():
            return p
    return None


def load_optional_binary_mask(mask_root: Path, name: str, h: int, w: int, dilate_px: int = 0):
    if mask_root is None:
        return None

    mask_path = resolve_mask_file(mask_root, name)
    if mask_path is None:
        return None

    if mask_path.suffix.lower() == ".npy":
        mask = np.load(str(mask_path))
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"无法读取掩码: {mask_path}")

    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2)
    else:
        mask = mask > 0

    mask = mask.astype(np.uint8)

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if dilate_px > 0:
        k = int(dilate_px) * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask.astype(bool)


def load_depth_auto(npy_dir: Path, exr_dir: Path, name: str):
    depth = None
    if npy_dir is not None:
        depth = load_depth_npy(npy_dir / f"{name}.npy")
        if depth is not None:
            return depth
    if exr_dir is not None:
        depth = load_depth_exr(exr_dir / f"{name}.exr")
        if depth is not None:
            return depth
    return None


def save_depth_auto(name: str, depth: np.ndarray, npy_dir: Path, exr_dir: Path, vis_dir: Path):
    if npy_dir is not None:
        save_depth_npy(npy_dir / f"{name}.npy", depth)
    if exr_dir is not None:
        save_depth_exr(exr_dir / f"{name}.exr", depth)
    if vis_dir is not None:
        save_depth_vis_png(vis_dir / f"{name}.png", make_depth_vis(depth))


def fuse_lidar_obj_depth(
    lidar_depth: np.ndarray,
    obj_depth: np.ndarray,
    veg_mask: np.ndarray = None,
    replace_abs_thr: float = 1.5,
    replace_rel_thr: float = 0.15,
):
    """
    融合规则：
    1) 激光无效、OBJ有效：直接用 OBJ
    2) 二者都有效：
       - 只有差异很大，且 OBJ 更近，且非植被，才替换为 OBJ
       - 其他情况保留激光
    """
    if lidar_depth is None and obj_depth is None:
        return None, {}

    if lidar_depth is None:
        out = obj_depth.astype(np.float32)
        stats = {
            "num_fill_from_obj": int(np.sum(is_valid_depth(obj_depth))),
            "num_replace_from_obj": 0,
            "num_veg_protected": 0,
            "num_keep_lidar": 0,
            "num_valid_fused": int(np.sum(is_valid_depth(out))),
        }
        return out, stats

    if obj_depth is None:
        out = lidar_depth.astype(np.float32)
        stats = {
            "num_fill_from_obj": 0,
            "num_replace_from_obj": 0,
            "num_veg_protected": 0,
            "num_keep_lidar": int(np.sum(is_valid_depth(out))),
            "num_valid_fused": int(np.sum(is_valid_depth(out))),
        }
        return out, stats

    if lidar_depth.shape != obj_depth.shape:
        raise ValueError(f"lidar_depth.shape != obj_depth.shape: {lidar_depth.shape} vs {obj_depth.shape}")

    h, w = lidar_depth.shape[:2]
    if veg_mask is not None and veg_mask.shape[:2] != (h, w):
        raise ValueError("veg_mask shape mismatch")

    lidar_valid = is_valid_depth(lidar_depth)
    obj_valid = is_valid_depth(obj_depth)
    both_valid = lidar_valid & obj_valid

    out = np.zeros_like(lidar_depth, dtype=np.float32)
    out[lidar_valid] = lidar_depth[lidar_valid]

    # 1) 激光无效处，OBJ 直接补
    fill_from_obj = (~lidar_valid) & obj_valid
    out[fill_from_obj] = obj_depth[fill_from_obj]

    # 2) 激光有效且 OBJ 有效时，按规则替换
    diff = np.abs(lidar_depth - obj_depth)
    rel = diff / np.maximum(lidar_depth, 1e-6)
    big_diff = (diff > float(replace_abs_thr)) & (rel > float(replace_rel_thr))
    obj_is_closer = obj_depth < lidar_depth

    replace_mask = both_valid & big_diff & obj_is_closer

    veg_protected = np.zeros_like(replace_mask, dtype=bool)
    if veg_mask is not None:
        veg_protected = replace_mask & veg_mask
        replace_mask = replace_mask & (~veg_mask)

    out[replace_mask] = obj_depth[replace_mask]

    stats = {
        "num_lidar_valid": int(np.sum(lidar_valid)),
        "num_obj_valid": int(np.sum(obj_valid)),
        "num_both_valid": int(np.sum(both_valid)),
        "num_fill_from_obj": int(np.sum(fill_from_obj)),
        "num_replace_from_obj": int(np.sum(replace_mask)),
        "num_veg_protected": int(np.sum(veg_protected)),
        "num_keep_lidar": int(np.sum(lidar_valid) - int(np.sum(replace_mask))),
        "num_valid_fused": int(np.sum(is_valid_depth(out))),
    }

    debug = {
        "fill_from_obj": fill_from_obj,
        "replace_from_obj": replace_mask,
        "veg_protected": veg_protected,
    }
    return out.astype(np.float32), stats, debug


# =========================
# Output Dirs
# =========================
def ensure_dirs(output_root: Path, save_npy: bool, save_exr: bool, save_vis: bool, save_images: bool):
    output_root.mkdir(parents=True, exist_ok=True)

    cam_out_dir = output_root / "cams"
    cam_out_dir.mkdir(parents=True, exist_ok=True)

    image_dir = None
    if save_images:
        image_dir = output_root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

    def build_depth_dirs(prefix: str):
        npy_dir = None
        exr_dir = None
        vis_dir = None

        if save_npy:
            npy_dir = output_root / f"depth_{prefix}_npy"
            npy_dir.mkdir(parents=True, exist_ok=True)

        if save_exr:
            exr_dir = output_root / f"depth_{prefix}_exr"
            exr_dir.mkdir(parents=True, exist_ok=True)

        if save_vis:
            vis_dir = output_root / f"depth_{prefix}_vis"
            vis_dir.mkdir(parents=True, exist_ok=True)

        return {"npy": npy_dir, "exr": exr_dir, "vis": vis_dir}

    debug_dir = output_root / "fusion_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    return {
        "images": image_dir,
        "cams": cam_out_dir,
        "lidar": build_depth_dirs("lidar"),
        "obj": build_depth_dirs("obj"),
        "fused": build_depth_dirs("fused"),
        "debug": debug_dir,
    }


# =========================
# Render Source Depth
# =========================
def render_source_depths(
    source_name: str,
    ply_paths,
    cams,
    vis_cache: VisCache,
    out_dirs: dict,
):
    print(f"\n[INFO] 开始渲染 {source_name} 点云，共 {len(ply_paths)} 个 ply")

    for ply_idx, ply_path in enumerate(ply_paths):
        print(f"[INFO] {source_name} ply ({ply_idx + 1}/{len(ply_paths)}): {ply_path}")
        pcd = load_point_cloud(str(ply_path))
        n_points = np.asarray(pcd.points).shape[0]
        print(f"[INFO] 当前点云点数: {n_points}")

        geom_tag = str(ply_path.resolve())

        for cam in tqdm(cams, desc=f"Render {source_name}:{ply_path.name}"):
            name = cam["name"]
            h, w = cam["h"], cam["w"]

            vis, key = vis_cache.get(w, h)
            vis_cache.ensure_geometry(key, pcd, geom_tag)

            set_camera_from_world2cam(
                vis,
                cam["intrinsic"],
                cam["extrinsic"],
                w,
                h,
            )

            curr_depth = render_depth_only(vis)

            prev_depth = load_depth_auto(
                out_dirs["npy"],
                out_dirs["exr"],
                name,
            )

            fused_depth = fuse_depth_keep_nearest(prev_depth, curr_depth)
            save_depth_auto(name, fused_depth, out_dirs["npy"], out_dirs["exr"], out_dirs["vis"])


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="分别渲染激光点云和 OBJ 点云深度图，再按规则融合。"
    )
    parser.add_argument("lidar_ply_root", type=str, help="激光点云 ply 文件或目录")
    parser.add_argument("obj_ply_root", type=str, help="OBJ 采样点云 ply 文件或目录")
    parser.add_argument("cam_root", type=str, help="相机目录，或其上一级目录（内部应包含 cams/）")
    parser.add_argument("output_root", type=str, help="输出目录")

    parser.add_argument("--image-root", type=str, default=None, help="原始图像目录")
    parser.add_argument("--veg-mask-root", type=str, default=None, help="植被掩码目录，非零视为植被")
    parser.add_argument("--view-list-txt", type=str, default=None, help="一行一个视角名")
    parser.add_argument("--pair-list-txt", type=str, default=None, help="一行两个视角名")
    parser.add_argument("--max-image-edge", type=int, default=1600, help="最大图像边")
    parser.add_argument("--prepare-workers", type=int, default=8, help="Prepare Cameras 的并行线程数，<=0 表示自动")

    parser.add_argument("--lidar-point-size", type=float, default=2.0, help="激光点大小")
    parser.add_argument("--obj-point-size", type=float, default=2.0, help="OBJ 点大小")
    parser.add_argument("--visible", action="store_true", help="显示 Open3D 窗口")

    parser.add_argument("--replace-abs-thr", type=float, default=1.5, help="差异绝对阈值，单位米")
    parser.add_argument("--replace-rel-thr", type=float, default=0.15, help="差异相对阈值，相对激光深度")
    parser.add_argument("--veg-dilate-px", type=int, default=2, help="植被掩码膨胀像素，保护边界")

    parser.add_argument("--save-depth-npy", action="store_true", help="保存 .npy")
    parser.add_argument("--save-depth-exr", action="store_true", help="保存 .exr")
    parser.add_argument("--save-depth-vis", action="store_true", help="保存可视化 png")
    parser.add_argument("--save-debug-mask", action="store_true", help="保存融合调试 mask")

    args = parser.parse_args()

    if (not args.save_depth_npy) and (not args.save_depth_exr) and (not args.save_depth_vis):
        args.save_depth_exr = True

    lidar_ply_root = Path(args.lidar_ply_root)
    obj_ply_root = Path(args.obj_ply_root)
    cam_root = Path(args.cam_root)
    output_root = Path(args.output_root)

    image_root = None
    if args.image_root is not None:
        image_root = Path(args.image_root)
        if not image_root.exists() or not image_root.is_dir():
            raise FileNotFoundError(f"原始图像目录不存在: {image_root}")

    veg_mask_root = None
    if args.veg_mask_root is not None:
        veg_mask_root = Path(args.veg_mask_root)
        if not veg_mask_root.exists() or not veg_mask_root.is_dir():
            raise FileNotFoundError(f"植被掩码目录不存在: {veg_mask_root}")

    dirs = ensure_dirs(
        output_root,
        save_npy=args.save_depth_npy,
        save_exr=args.save_depth_exr,
        save_vis=args.save_depth_vis,
        save_images=(image_root is not None),
    )

    lidar_plys = find_all_plys(lidar_ply_root)
    obj_plys = find_all_plys(obj_ply_root)

    print(f"[INFO] 激光 ply 数量: {len(lidar_plys)}")
    print(f"[INFO] OBJ ply 数量: {len(obj_plys)}")

    cams_dir = resolve_cams_dir(cam_root)
    cam_txts = find_cam_txts(cams_dir)

    selected_names = read_view_list_txt(args.view_list_txt, args.pair_list_txt)
    cam_txts = filter_cam_txts_by_view_list(cam_txts, selected_names)
    print(f"[INFO] 实际处理视角数: {len(cam_txts)} | {cams_dir}")

    cams = prepare_cameras(
        cam_txts=cam_txts,
        max_image_edge=args.max_image_edge,
        cam_out_dir=dirs["cams"],
        image_root=image_root,
        images_dir=dirs["images"],
        prepare_workers=args.prepare_workers,
    )

    lidar_cache = VisCache(point_size=args.lidar_point_size, visible=args.visible)
    obj_cache = VisCache(point_size=args.obj_point_size, visible=args.visible)

    try:
        # 1) 渲染激光深度
        render_source_depths(
            source_name="lidar",
            ply_paths=lidar_plys,
            cams=cams,
            vis_cache=lidar_cache,
            out_dirs=dirs["lidar"],
        )

        # 2) 渲染 OBJ 深度
        render_source_depths(
            source_name="obj",
            ply_paths=obj_plys,
            cams=cams,
            vis_cache=obj_cache,
            out_dirs=dirs["obj"],
        )

    finally:
        lidar_cache.close_all()
        obj_cache.close_all()

    # 3) 按规则融合
    print("\n[INFO] 开始融合激光深度和 OBJ 深度")
    fusion_summary = {
        "replace_abs_thr": float(args.replace_abs_thr),
        "replace_rel_thr": float(args.replace_rel_thr),
        "veg_dilate_px": int(args.veg_dilate_px),
        "per_view": {},
    }

    debug_fill_dir = dirs["debug"] / "fill_from_obj"
    debug_replace_dir = dirs["debug"] / "replace_from_obj"
    debug_veg_dir = dirs["debug"] / "veg_protected"

    for cam in tqdm(cams, desc="Fuse lidar + obj"):
        name = cam["name"]
        h, w = cam["h"], cam["w"]

        lidar_depth = load_depth_auto(dirs["lidar"]["npy"], dirs["lidar"]["exr"], name)
        obj_depth = load_depth_auto(dirs["obj"]["npy"], dirs["obj"]["exr"], name)

        veg_mask = load_optional_binary_mask(
            veg_mask_root,
            name,
            h=h,
            w=w,
            dilate_px=args.veg_dilate_px,
        )

        fused_ret = fuse_lidar_obj_depth(
            lidar_depth=lidar_depth,
            obj_depth=obj_depth,
            veg_mask=veg_mask,
            replace_abs_thr=args.replace_abs_thr,
            replace_rel_thr=args.replace_rel_thr,
        )

        if fused_ret is None:
            continue

        if len(fused_ret) == 2:
            fused_depth, stats = fused_ret
            debug = None
        else:
            fused_depth, stats, debug = fused_ret

        if fused_depth is None:
            continue

        save_depth_auto(
            name,
            fused_depth,
            dirs["fused"]["npy"],
            dirs["fused"]["exr"],
            dirs["fused"]["vis"],
        )

        if args.save_debug_mask and debug is not None:
            save_mask_png(debug_fill_dir / f"{name}.png", debug["fill_from_obj"])
            save_mask_png(debug_replace_dir / f"{name}.png", debug["replace_from_obj"])
            save_mask_png(debug_veg_dir / f"{name}.png", debug["veg_protected"])

        fusion_summary["per_view"][name] = stats

    save_json(output_root / "summary_fusion.json", fusion_summary)
    print(f"\n[INFO] 完成，输出目录: {output_root}")


if __name__ == "__main__":
    main()

"""
python render_fused_depth_lidar_obj.py \
    ./lidar/nanfang/transform/lidar_finall.ply \
    ./recon/nanfang/models/pc/0 \
    ./recon/nanfang/cams \
    ./scene_fused_render/nanfang \
    --image-root ./undistort/nanfang/ImageCorrection/undistort \
    --max-image-edge 1600 \
    --save-depth-exr --save-depth-vis \
    --lidar-point-size 3.0 \
    --obj-point-size 3.0 \
    --replace-abs-thr 1.5 \
    --replace-rel-thr 0.15 \
    --save-debug-mask

"""