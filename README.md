# sfm_lidar_alignment

A practical Python toolkit for aligning LiDAR point clouds with SfM / photogrammetric reconstruction, refining the transform with view-based matching and ICP, and producing fused depth or point-cloud outputs for downstream reconstruction and evaluation.

This repository is designed for a workflow where:
- a LiDAR cloud is available in LAS/LAZ/PLY,
- an SfM / mesh reconstruction is available as OBJ files and camera parameters,
- an initial coarse registration is obtained manually or with CloudCompare,
- the alignment is further refined using rendered depth, image feature matching, 3D lifting, and ICP.

---

## Main capabilities

- Shift raw LAS/LAZ to a local coordinate system.
- Downsample LiDAR point clouds for efficient manual registration and debugging.
- Convert reconstruction OBJ files into PLY point clouds by either:
  - using original mesh vertices, or
  - sampling points from triangle surfaces.
- Estimate a reasonable sampling spacing from LiDAR density.
- Split large reconstructed point clouds into smaller PLY chunks for more stable rendering.
- Render RGB/depth from LiDAR or reconstructed point clouds using camera files in `cams/`.
- Select downward-looking views for refinement.
- Select reliable stereo pairs with geometric overlap, feature matching, reprojection filtering, and optional depth consistency.
- Refine the LiDAR-to-SfM transform with:
  - 2D feature matching,
  - depth lifting to 3D correspondences,
  - global similarity / rigid estimation,
  - optional ICP.
- Evaluate pair quality and global alignment quality.
- Fuse LiDAR depth with reconstruction depth to compensate for missing regions.
- Apply the final transform chain to the full LiDAR cloud.

---

## Repository structure

### Data preparation and point-cloud generation
- `shift_las_to_local.py`  
  Shift LAS/LAZ into a local coordinate system, optionally using `metadata.xml`.
- `downsample_las_to_ply.py`  
  Chunked voxel downsampling for very large LAS/LAZ files, exported as PLY.
- `downsample_obj_to_ply.py`  
  Recursively read OBJ meshes, use mesh vertices as points, downsample, and merge into one PLY.
- `sample_obj_to_ply.py`  
  Sample dense surface points from OBJ meshes and merge them into one PLY.
- `estimate_spacing_from_lidar.py`  
  Estimate a reasonable reconstruction sampling spacing from LiDAR point spacing statistics.
- `split_ply.py`  
  Split a large PLY into smaller chunks.

### Rendering
- `render_depth_from_lidar.py`  
  Render LiDAR RGB and depth from camera views.
- `render_depth_from_ply.py`  
  Render depth from one or multiple reconstructed PLY chunks and keep the nearest depth per pixel.
- `render_fused_depth_lidar_obj.py`  
  Render LiDAR depth and reconstruction depth separately, then fuse them with simple rules.
- `render_lowres_depth.py`  
  Low-resolution rendering utility.

### View selection, matching, refinement, and evaluation
- `random_select_downward_views.py`  
  Select approximately downward-looking views and spatially spread them with grid sampling.
- `select_stereo_pairs_with_depth_consistency.py`  
  Select reliable stereo pairs for evaluation / refinement.
- `evaluate_pair_alignment_metrics.py`  
  Evaluate reprojection-based alignment quality on selected stereo pairs.
- `refine_transform_v2.py`  
  Estimate a refined transform using lifted 3D correspondences and ICP.
- `depth_world_cloud_metric.py`  
  Compare two rendered depth sets by lifting them to world-space point clouds.

### Masking and utilities
- `sam3_generate_exclusion_masks.py`  
  Generate simple exclusion masks (mainly vegetation / water) for evaluation or fusion.
- `djiterra2wai.py`, `vis_ply.py`, and other scripts  
  Utility scripts for format conversion, inspection, and debugging.

---

## Expected inputs

### 1. Camera files
Most scripts expect a directory like:

```text
project/
└── cams/
    ├── 00000000.txt
    ├── 00000001.txt
    └── ...
```

The camera text files should follow the repository's camera format with:
- `extrinsic` (world-to-camera),
- `intrinsic`,
- `h w fov`.

Some evaluation scripts also support BlendedMVS / MVSNet-style camera text files.

### 2. Images
When RGB images are needed, scripts expect images to be matched by stem name, for example:

```text
images/
├── 00000000.jpg
├── 00000001.jpg
└── ...
```

### 3. Depth outputs
Rendered depths are usually saved in one or more of:
- `depth_npy/`
- `depth_exr/`
- `depth_vis/`

### 4. Transform files
The final transform chain is composed in this order:

```text
transform_manual.txt -> transform_icp.txt -> transform_refine.txt
```

If a point cloud is transformed by `T1`, then `T2`, then `T3`, the final transform applied by `lidar_transform.py` is:

```text
T = T3 @ T2 @ T1
```

This makes it easy to combine:
- a manual coarse alignment,
- an optional coarse ICP alignment,
- a final refinement estimated from rendered views.

---

## Installation

There is currently no `requirements.txt`, so install dependencies manually.

### Core dependencies
```bash
pip install numpy opencv-python open3d laspy tqdm scipy pillow
```

### Optional dependencies
- `torch` for `sam3_generate_exclusion_masks.py`
- your local SAM3 package / checkpoint for exclusion-mask generation

### Notes
- OpenEXR support in OpenCV is required if you want to save `.exr` depth maps.
- `Open3D` rendering may require a desktop environment or an off-screen/Xvfb setup on Linux.

---

## Recommended workflow

### Step 0. Shift raw LiDAR to a local coordinate system
This is helpful when the original LAS/LAZ uses large world coordinates.

```bash
python shift_las_to_local.py \
    --input_path scene/cloud_merged.las \
    --replace_original \
    --metadata_xml_path recon/models/pc/0/terra_obj/metadata.xml
```

### Step 1. Create lightweight point clouds for coarse registration
Generate a downsampled LiDAR PLY:

```bash
python downsample_las_to_ply.py \
    scene/cloud_merged.las \
    scene/downsample_lidar.ply \
    --voxel-size 0.3
```

Generate a reconstruction PLY from OBJ vertices:

```bash
python downsample_obj_to_ply.py \
    recon/models/pc/0/terra_obj \
    recon/models/pc/downsample_recon.ply \
    --voxel-size 0.5
```

If you need denser reconstruction sampling, first estimate a spacing from LiDAR density:

```bash
python estimate_spacing_from_lidar.py scene/transform/lidar_full.ply
```

Then sample the reconstruction surfaces:

```bash
python sample_obj_to_ply.py \
    recon/models/pc/0/terra_obj \
    recon/models/pc/sampled_recon.ply \
    --spacing 0.12
```

### Step 2. Split large reconstruction PLYs if needed
If rendering a single dense reconstruction PLY is unstable or too slow, split it into chunks:

```bash
python split_ply.py recon/models/pc/sampled_recon.ply
```

### Step 3. Perform coarse registration externally
Use CloudCompare or your preferred tool to compute an initial LiDAR-to-SfM transform.

Save the coarse transforms in a folder such as:

```text
scene/transform/
├── transform_manual.txt
└── transform_icp.txt
```

### Step 4. Apply the current transform chain to the LiDAR cloud
Once coarse transforms exist, convert the LiDAR cloud into the reconstruction coordinate system:

```bash
python lidar_transform.py \
    --lidar scene/cloud_merged.las \
    --transform scene/transform \
    --out_lidar_name lidar_finall.ply
```

### Step 5. Select stable downward-looking views
Pick a spatially distributed set of views for refinement:

```bash
python random_select_downward_views.py \
    recon \
    scene/selected_views.txt
```

### Step 6. Render LiDAR depth and reconstruction depth
Render LiDAR RGB/depth for the selected views:

```bash
python render_depth_from_lidar.py \
    scene/transform/lidar_finall.ply \
    recon \
    scene/lidar_render \
    --save-depth-npy --save-depth-vis \
    --view-list-txt scene/selected_views.txt \
    --point-size 3.0
```

Render reconstruction depth from one or many PLY chunks:

```bash
python render_depth_from_ply.py \
    recon/models/pc/0 \
    recon \
    recon/recon_render \
    --save-depth-npy --save-depth-vis \
    --image-root undistort/ImageCorrection/undistort \
    --view-list-txt scene/selected_views.txt \
    --point-size 3.0
```

### Step 7. Optionally generate exclusion masks
If vegetation, water, or unstable image regions should be excluded during evaluation:

```bash
python sam3_generate_exclusion_masks.py \
    --images_dir recon/recon_render/images \
    --selected_views_txt scene/selected_views.txt \
    --out_dir scene/eval_masks \
    --device cuda:0 \
    --input_long_edge 512 \
    --checkpoint_path sam3/checkpoints/sam3.pt
```

Outputs:

```text
scene/eval_masks/
├── binary_masks/
└── overlays/
```

### Step 8. Select reliable stereo pairs
Build a compact, high-quality pair set for refinement / evaluation:

```bash
python select_stereo_pairs_with_depth_consistency.py \
    --image_dir recon/recon_render/images \
    --cam_dir recon/recon_render/cams \
    --depth_dir recon/recon_render/depth_npy \
    --out_dir recon/pair_selected \
    --num_pairs 50 \
    --max_image_long_edge 1600
```

Outputs:

```text
recon/pair_selected/
├── visualizations/
├── match_data/
├── selected_pairs.json
├── selected_pairs.txt
└── summary.json
```

### Step 9. Refine the transform
This is the core refinement stage.

`refine_transform_v2.py` supports four modes:
- `only_match`
- `only_icp`
- `icp_then_match`
- `match_then_icp`

Recommended default:

```bash
python refine_transform_v2.py \
    --root_a scene/lidar_render \
    --root_b recon/recon_render \
    --out_dir scene/transform_refine \
    --mode match_then_icp \
    --num_images 50 \
    --estimate_scale \
    --enable_translation_filter
```

Where:
- `root_a` and `root_b` should each contain `cams/`, `images/`, and `depth_npy/` or `depth/`.
- The script matches same-stem images, lifts 2D correspondences using depth, estimates a global transform, and optionally runs ICP on view-lifted point clouds.

Important outputs include:

```text
scene/transform_refine/
├── stage_match/
├── stage_icp/
├── transform_refine.txt
└── summary.json
```

### Step 10. Evaluate alignment quality
#### Pair-based evaluation
```bash
python evaluate_pair_alignment_metrics.py \
    --pair_dir recon/pair_selected \
    --cam_dir recon/recon_render/cams \
    --depth_dir scene/lidar_render/depth_npy \
    --out_dir scene/eval_pair
```

Outputs:
- `metrics_per_pair.csv`
- `metrics_per_pair.json`
- `summary.txt`
- `summary.json`

#### World-space depth comparison
```bash
python depth_world_cloud_metric.py \
    --depth_dir_a scene/lidar_render/depth_npy \
    --depth_dir_b recon/recon_render/depth_npy \
    --cam_dir recon/recon_render/cams \
    --exclude_mask_dir scene/eval_masks/binary_masks \
    --out_dir scene/eval_world \
    --save_ply
```

This script lifts both depth sets back into world coordinates and computes point-cloud alignment metrics.

### Step 11. Apply the final transform to the full LiDAR cloud
After refinement, place `transform_refine.txt` in the transform folder and run:

```bash
python lidar_transform.py \
    --lidar scene/cloud_merged.las \
    --transform scene/transform \
    --out_lidar_name lidar_final_refined.ply
```

### Step 12. Fuse LiDAR and reconstruction depth
If you want LiDAR depth where available and reconstruction depth elsewhere:

```bash
python render_fused_depth_lidar_obj.py \
    scene/transform/lidar_finall.ply \
    recon/models/pc/0 \
    recon/cams \
    scene/fused_render \
    --image-root undistort/ImageCorrection/undistort \
    --max-image-edge 1600 \
    --save-depth-npy --save-depth-vis \
    --replace-abs-thr 1.5 \
    --replace-rel-thr 0.15 \
    --save-debug-mask
```

Fusion logic:
- fill invalid LiDAR depth with reconstruction depth,
- only replace valid LiDAR depth when the difference is large enough and the reconstruction depth is closer,
- optionally protect vegetation regions using a mask.

---

## Typical directory layout

A practical project layout may look like this:

```text
workspace/
├── lidar/
│   └── scene/
│       ├── cloud_merged.las
│       ├── transform/
│       │   ├── transform_manual.txt
│       │   ├── transform_icp.txt
│       │   └── transform_refine.txt
│       ├── lidar_render/
│       │   ├── cams/
│       │   ├── images/
│       │   └── depth_npy/
│       ├── selected_views.txt
│       └── eval_masks/
│           ├── binary_masks/
│           └── overlays/
└── recon/
    └── scene/
        ├── cams/
        ├── models/
        │   └── pc/
        │       └── 0/
        ├── recon_render/
        │   ├── cams/
        │   ├── images/
        │   └── depth_npy/
        └── pair_selected/
            ├── match_data/
            ├── visualizations/
            ├── selected_pairs.json
            ├── selected_pairs.txt
            └── summary.json
```

---

## Important notes and assumptions

1. **Camera naming must be consistent.**  
   Most scripts match images, cameras, depths, and masks by file stem.

2. **Rendered resolution matters.**  
   If you use `--max-image-edge`, camera intrinsics are scaled and saved to the output `cams/` directory. Downstream scripts should use the scaled camera files together with the rendered images/depths.

3. **Depth format is flexible.**  
   Many scripts support `.npy`, `.exr`, `.pfm`, and some image-based formats, but `.npy` is usually the safest intermediate format.

4. **Large scenes may require chunking.**  
   For very dense reconstructions, using `split_ply.py` and rendering multiple chunks with nearest-depth fusion is often more stable than rendering one huge PLY.

5. **The repository assumes a same-stem workflow.**  
   A view named `00001234` should ideally correspond to:
   - `cams/00001234.txt`
   - `images/00001234.jpg` or `.png`
   - `depth_npy/00001234.npy`
   - `binary_masks/00001234.png`

6. **SAM3 is optional.**  
   If you do not need exclusion masks, you can skip `sam3_generate_exclusion_masks.py`.

---

## Suggested minimal pipeline

If you only want the shortest useful path:

1. Shift LAS to local coordinates.  
2. Downsample LiDAR and reconstruction.  
3. Compute coarse alignment manually / in CloudCompare.  
4. Apply coarse transform to LiDAR.  
5. Select stable downward views.  
6. Render LiDAR and reconstruction depth.  
7. Select reliable stereo pairs.  
8. Run `refine_transform_v2.py` with `match_then_icp`.  
9. Apply `transform_refine.txt` to the full LiDAR cloud.  
10. Evaluate and optionally fuse depth.

---

## Troubleshooting

### Open3D window creation failed
On headless Linux machines, use an X server / `xvfb-run`, or switch to a machine with GUI support.

### EXR writing failed
Make sure your OpenCV build supports OpenEXR. Several scripts explicitly set:

```python
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
```

but OpenCV still needs EXR support in the underlying build.

### Very slow rendering
Try one or more of the following:
- reduce `--max-image-edge`,
- increase point size only as needed,
- split reconstruction PLYs into chunks,
- store intermediate depth as `.npy` instead of `.exr`.

### Poor refinement quality
Common causes:
- unstable views with vegetation / water / weak LiDAR coverage,
- too many oblique views,
- poor coarse initialization,
- inconsistent camera/image stems,
- rendered depth generated at a different scale than the camera files used in refinement.

---

## Acknowledgment of coarse alignment
This repository assumes that the initial LiDAR-to-SfM alignment may come from external tools such as CloudCompare. The Python scripts here are mainly focused on:
- point-cloud preparation,
- rendering-based view selection,
- transform refinement,
- evaluation,
- final fusion.

---

## License

No license file is currently included in the repository. Add one if you plan to distribute or reuse the code publicly.
