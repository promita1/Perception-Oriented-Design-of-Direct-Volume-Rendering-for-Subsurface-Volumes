import os
import math
import itertools
import numpy as np
import xarray as xr
import vtk
from vtk.util import numpy_support
from PIL import Image, ImageDraw

# =========================================================
# USER SETTINGS
# =========================================================
NC_PATH = "data/CVM17_L3.nc"
SCALAR_VAR_NAME = "vs"

OUTPUT_ROOT = "outputs"
RENDER_DIR = os.path.join(OUTPUT_ROOT, "renders")
ATLAS_DIR = os.path.join(OUTPUT_ROOT, "atlas")
LOG_DIR = os.path.join(OUTPUT_ROOT, "logs")

ATLAS_PATH = os.path.join(ATLAS_DIR, "dvr_atlas.png")
PREPROCESS_LOG_PATH = os.path.join(LOG_DIR, "preprocessing_log.txt")
CONDITION_LOG_PATH = os.path.join(LOG_DIR, "condition_log.txt")

# Crop ratios
CROP_X = (0.20, 0.75)
CROP_Y = (0.20, 0.75)
CROP_Z = (0.10, 0.90)

# Downsample for interactivity
DOWNSAMPLE = 1

# Invalid values
INVALID_VALUES = [-999, -999.0]

# Render settings
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
BACKGROUND_RGB = (0.08, 0.08, 0.10)

# Atlas settings
ATLAS_TILE_WIDTH = 320
ATLAS_TILE_HEIGHT = 240
ATLAS_LABEL_HEIGHT = 50
ATLAS_COLUMNS = 3  # 3 columns for shading groups works nicely

# Preview only one interactive window at the end
SHOW_INTERACTIVE_PREVIEW = True

# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ensure_all_dirs():
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(RENDER_DIR)
    ensure_dir(ATLAS_DIR)
    ensure_dir(LOG_DIR)


def sanitize_filename(text):
    return text.replace(" ", "_").replace("/", "_")


def load_dataset(nc_path):
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"Could not find dataset file: {nc_path}")
    ds = xr.open_dataset(nc_path)
    return ds


def validate_variable(ds, var_name):
    if var_name not in ds.data_vars:
        raise ValueError(f"Variable '{var_name}' not found. Available: {list(ds.data_vars)}")
    da = ds[var_name]
    if da.ndim != 3:
        raise ValueError(f"Variable '{var_name}' is not 3D. Shape: {da.shape}")
    if not np.issubdtype(da.dtype, np.number):
        raise ValueError(f"Variable '{var_name}' is not numeric.")
    return da


def crop_volume(volume, crop_x, crop_y, crop_z):
    nx, ny, nz = volume.shape
    x0, x1 = int(nx * crop_x[0]), int(nx * crop_x[1])
    y0, y1 = int(ny * crop_y[0]), int(ny * crop_y[1])
    z0, z1 = int(nz * crop_z[0]), int(nz * crop_z[1])
    return volume[x0:x1, y0:y1, z0:z1]


def sanitize_volume(volume):
    vol = np.array(volume, dtype=np.float32)

    for bad in INVALID_VALUES:
        vol[vol == bad] = np.nan

    vol[~np.isfinite(vol)] = np.nan
    nan_ratio = float(np.isnan(vol).mean())

    valid = vol[~np.isnan(vol)]
    if valid.size == 0:
        raise ValueError("All values are invalid after sanitization.")

    fill_value = np.percentile(valid, 1)
    vol = np.where(np.isnan(vol), fill_value, vol)

    return vol, nan_ratio


def normalize_volume(volume):
    orig_min = float(np.min(volume))
    orig_max = float(np.max(volume))

    if np.isclose(orig_min, orig_max):
        raise ValueError("Volume has near-constant values; cannot normalize.")

    norm = (volume - orig_min) / (orig_max - orig_min)
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)

    return norm, orig_min, orig_max


def volume_to_vtk_image(volume):
    # volume expected as (X, Y, Z)
    nx, ny, nz = volume.shape

    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(nx, ny, nz)
    vtk_img.SetSpacing(1.0, 1.0, 1.0)
    vtk_img.SetOrigin(0.0, 0.0, 0.0)

    flat = np.ravel(volume, order="F")
    vtk_array = numpy_support.numpy_to_vtk(
        num_array=flat,
        deep=True,
        array_type=vtk.VTK_FLOAT
    )
    vtk_array.SetName("Scalars")
    vtk_img.GetPointData().SetScalars(vtk_array)

    return vtk_img


def save_preprocessing_log(path, var_name, original_shape, cropped_shape, orig_min, orig_max, nan_ratio):
    with open(path, "w", encoding="utf-8") as f:
        f.write("DVR preprocessing summary\n")
        f.write("=========================\n")
        f.write(f"Selected variable: {var_name}\n")
        f.write(f"Original shape: {original_shape}\n")
        f.write(f"Cropped/downsampled shape: {cropped_shape}\n")
        f.write(f"Original scalar min: {orig_min}\n")
        f.write(f"Original scalar max: {orig_max}\n")
        f.write(f"NaN ratio after invalid cleanup: {nan_ratio:.6f}\n")
        f.write(f"CROP_X: {CROP_X}\n")
        f.write(f"CROP_Y: {CROP_Y}\n")
        f.write(f"CROP_Z: {CROP_Z}\n")
        f.write(f"DOWNSAMPLE: {DOWNSAMPLE}\n")


def save_condition_log(path, conditions):
    with open(path, "w", encoding="utf-8") as f:
        f.write("27 DVR conditions\n")
        f.write("=================\n")
        for i, cond in enumerate(conditions, start=1):
            f.write(
                f"{i:02d}. tf={cond['tf_name']} | sample={cond['sample_name']} | shading={cond['shading_name']} | file={cond['filename']}\n"
            )


# =========================================================
# TRANSFER FUNCTIONS
# =========================================================
def build_transfer_function(tf_name):
    color_tf = vtk.vtkColorTransferFunction()
    opacity_tf = vtk.vtkPiecewiseFunction()

    if tf_name == "tf1_baseline":
        # Balanced baseline
        color_tf.AddRGBPoint(0.00, 0.05, 0.05, 0.08)
        color_tf.AddRGBPoint(0.25, 0.18, 0.28, 0.45)
        color_tf.AddRGBPoint(0.50, 0.50, 0.62, 0.72)
        color_tf.AddRGBPoint(0.75, 0.80, 0.72, 0.50)
        color_tf.AddRGBPoint(1.00, 0.95, 0.95, 0.95)

        opacity_tf.AddPoint(0.00, 0.00)
        opacity_tf.AddPoint(0.10, 0.00)
        opacity_tf.AddPoint(0.25, 0.03)
        opacity_tf.AddPoint(0.45, 0.08)
        opacity_tf.AddPoint(0.65, 0.18)
        opacity_tf.AddPoint(0.85, 0.35)
        opacity_tf.AddPoint(1.00, 0.55)

    elif tf_name == "tf2_boundary":
        # Stronger contrast transitions for boundary emphasis
        color_tf.AddRGBPoint(0.00, 0.03, 0.03, 0.05)
        color_tf.AddRGBPoint(0.20, 0.15, 0.20, 0.40)
        color_tf.AddRGBPoint(0.40, 0.30, 0.45, 0.75)
        color_tf.AddRGBPoint(0.60, 0.82, 0.68, 0.28)
        color_tf.AddRGBPoint(0.80, 0.95, 0.82, 0.45)
        color_tf.AddRGBPoint(1.00, 1.00, 0.98, 0.95)

        opacity_tf.AddPoint(0.00, 0.00)
        opacity_tf.AddPoint(0.12, 0.00)
        opacity_tf.AddPoint(0.28, 0.02)
        opacity_tf.AddPoint(0.42, 0.15)
        opacity_tf.AddPoint(0.55, 0.05)
        opacity_tf.AddPoint(0.68, 0.28)
        opacity_tf.AddPoint(0.82, 0.42)
        opacity_tf.AddPoint(1.00, 0.60)

    elif tf_name == "tf3_continuity":
        # Smoother broad opacity spread for structure continuity
        color_tf.AddRGBPoint(0.00, 0.06, 0.06, 0.08)
        color_tf.AddRGBPoint(0.20, 0.16, 0.22, 0.34)
        color_tf.AddRGBPoint(0.45, 0.35, 0.48, 0.65)
        color_tf.AddRGBPoint(0.70, 0.62, 0.72, 0.78)
        color_tf.AddRGBPoint(0.90, 0.85, 0.84, 0.75)
        color_tf.AddRGBPoint(1.00, 0.96, 0.96, 0.92)

        opacity_tf.AddPoint(0.00, 0.00)
        opacity_tf.AddPoint(0.08, 0.00)
        opacity_tf.AddPoint(0.20, 0.02)
        opacity_tf.AddPoint(0.35, 0.06)
        opacity_tf.AddPoint(0.50, 0.12)
        opacity_tf.AddPoint(0.65, 0.22)
        opacity_tf.AddPoint(0.80, 0.32)
        opacity_tf.AddPoint(1.00, 0.48)

    else:
        raise ValueError(f"Unknown transfer function preset: {tf_name}")

    return color_tf, opacity_tf


# =========================================================
# SHADING / PROPERTY
# =========================================================
def build_volume_property(color_tf, opacity_tf, shading_name):
    prop = vtk.vtkVolumeProperty()
    prop.SetColor(color_tf)
    prop.SetScalarOpacity(opacity_tf)
    prop.SetInterpolationTypeToLinear()

    if shading_name == "off":
        prop.ShadeOff()

    elif shading_name == "basic":
        prop.ShadeOn()
        prop.SetAmbient(0.20)
        prop.SetDiffuse(0.75)
        prop.SetSpecular(0.15)
        prop.SetSpecularPower(12.0)

    elif shading_name == "depth":
        prop.ShadeOn()
        prop.SetAmbient(0.10)
        prop.SetDiffuse(0.85)
        prop.SetSpecular(0.25)
        prop.SetSpecularPower(18.0)

    else:
        raise ValueError(f"Unknown shading preset: {shading_name}")

    return prop


# =========================================================
# SAMPLING / MAPPER
# =========================================================
def build_mapper(vtk_img, sample_name):
    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputData(vtk_img)
    mapper.SetBlendModeToComposite()
    mapper.SetAutoAdjustSampleDistances(0)

    if sample_name == "low":
        mapper.SetSampleDistance(2.0)
    elif sample_name == "medium":
        mapper.SetSampleDistance(1.0)
    elif sample_name == "high":
        mapper.SetSampleDistance(0.5)
    else:
        raise ValueError(f"Unknown sampling preset: {sample_name}")

    return mapper


# =========================================================
# RENDERING
# =========================================================
def setup_renderer(volume_actor):
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume_actor)
    renderer.SetBackground(*BACKGROUND_RGB)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(WINDOW_WIDTH, WINDOW_HEIGHT)
    render_window.SetOffScreenRendering(1)

    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.Azimuth(35)
    camera.Elevation(20)
    camera.Zoom(1.2)
    renderer.ResetCameraClippingRange()

    return renderer, render_window


def save_screenshot(render_window, filepath):
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.ReadFrontBufferOff()
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filepath)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()


def render_condition(vtk_img, tf_name, sample_name, shading_name, output_path):
    color_tf, opacity_tf = build_transfer_function(tf_name)
    vol_prop = build_volume_property(color_tf, opacity_tf, shading_name)
    mapper = build_mapper(vtk_img, sample_name)

    volume_actor = vtk.vtkVolume()
    volume_actor.SetMapper(mapper)
    volume_actor.SetProperty(vol_prop)

    _, render_window = setup_renderer(volume_actor)
    render_window.Render()
    save_screenshot(render_window, output_path)


def show_interactive_preview(vtk_img, tf_name="tf1_baseline", sample_name="medium", shading_name="basic"):
    color_tf, opacity_tf = build_transfer_function(tf_name)
    vol_prop = build_volume_property(color_tf, opacity_tf, shading_name)
    mapper = build_mapper(vtk_img, sample_name)

    volume_actor = vtk.vtkVolume()
    volume_actor.SetMapper(mapper)
    volume_actor.SetProperty(vol_prop)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume_actor)
    renderer.SetBackground(*BACKGROUND_RGB)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(WINDOW_WIDTH, WINDOW_HEIGHT)
    render_window.SetWindowName("Interactive DVR Preview")

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.Azimuth(35)
    camera.Elevation(20)
    camera.Zoom(1.2)
    renderer.ResetCameraClippingRange()

    render_window.Render()
    interactor.Initialize()
    interactor.Start()


# =========================================================
# ATLAS
# =========================================================
def load_and_resize_image(path, width, height):
    img = Image.open(path).convert("RGB")
    return img.resize((width, height))


def create_atlas(condition_records, atlas_path, columns=ATLAS_COLUMNS):
    rows = math.ceil(len(condition_records) / columns)

    atlas_width = columns * ATLAS_TILE_WIDTH
    atlas_height = rows * (ATLAS_TILE_HEIGHT + ATLAS_LABEL_HEIGHT)

    atlas = Image.new("RGB", (atlas_width, atlas_height), color=(20, 20, 24))
    draw = ImageDraw.Draw(atlas)

    for idx, rec in enumerate(condition_records):
        row = idx // columns
        col = idx % columns

        x = col * ATLAS_TILE_WIDTH
        y = row * (ATLAS_TILE_HEIGHT + ATLAS_LABEL_HEIGHT)

        tile = load_and_resize_image(rec["filepath"], ATLAS_TILE_WIDTH, ATLAS_TILE_HEIGHT)
        atlas.paste(tile, (x, y))

        label = f"{rec['tf_name']} | {rec['sample_name']} | {rec['shading_name']}"
        draw.text((x + 8, y + ATLAS_TILE_HEIGHT + 10), label, fill=(235, 235, 235))

    atlas.save(atlas_path)
    return atlas_path


# =========================================================
# CONDITION GENERATION
# =========================================================
def generate_conditions():
    tf_names = ["tf1_baseline", "tf2_boundary", "tf3_continuity"]
    sample_names = ["low", "medium", "high"]
    shading_names = ["off", "basic", "depth"]

    conditions = []
    for tf_name, sample_name, shading_name in itertools.product(tf_names, sample_names, shading_names):
        filename = f"{sanitize_filename(tf_name)}__{sanitize_filename(sample_name)}__{sanitize_filename(shading_name)}.png"
        filepath = os.path.join(RENDER_DIR, filename)
        conditions.append({
            "tf_name": tf_name,
            "sample_name": sample_name,
            "shading_name": shading_name,
            "filename": filename,
            "filepath": filepath
        })
    return conditions


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_all_dirs()

    print("Loading dataset...")
    ds = load_dataset(NC_PATH)
    da = validate_variable(ds, SCALAR_VAR_NAME)

    print(f"Selected variable: {SCALAR_VAR_NAME}")
    print(f"Original shape: {da.shape}")

    vol = da.values
    vol, nan_ratio = sanitize_volume(vol)

    vol = crop_volume(vol, CROP_X, CROP_Y, CROP_Z)
    print(f"Shape after crop: {vol.shape}")

    if DOWNSAMPLE > 1:
        vol = vol[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE]
        print(f"Shape after downsample x{DOWNSAMPLE}: {vol.shape}")

    vol, orig_min, orig_max = normalize_volume(vol)
    print(f"Scalar range before normalization: min={orig_min:.4f}, max={orig_max:.4f}")
    print(f"Scalar range after normalization: min={vol.min():.4f}, max={vol.max():.4f}")

    save_preprocessing_log(
        PREPROCESS_LOG_PATH,
        SCALAR_VAR_NAME,
        da.shape,
        vol.shape,
        orig_min,
        orig_max,
        nan_ratio
    )
    print(f"Saved preprocessing log: {PREPROCESS_LOG_PATH}")

    print("Converting to vtkImageData...")
    vtk_img = volume_to_vtk_image(vol)

    print("Generating 27 conditions...")
    conditions = generate_conditions()
    save_condition_log(CONDITION_LOG_PATH, conditions)
    print(f"Saved condition log: {CONDITION_LOG_PATH}")

    for i, cond in enumerate(conditions, start=1):
        print(
            f"[{i:02d}/27] Rendering "
            f"{cond['tf_name']} | {cond['sample_name']} | {cond['shading_name']}"
        )
        render_condition(
            vtk_img,
            cond["tf_name"],
            cond["sample_name"],
            cond["shading_name"],
            cond["filepath"]
        )

    print("Creating atlas...")
    create_atlas(conditions, ATLAS_PATH, columns=ATLAS_COLUMNS)
    print(f"Saved atlas image: {ATLAS_PATH}")

    if SHOW_INTERACTIVE_PREVIEW:
        print("Opening one interactive preview window...")
        show_interactive_preview(
            vtk_img,
            tf_name="tf1_baseline",
            sample_name="medium",
            shading_name="basic"
        )

    print("Done.")


if __name__ == "__main__":
    main()