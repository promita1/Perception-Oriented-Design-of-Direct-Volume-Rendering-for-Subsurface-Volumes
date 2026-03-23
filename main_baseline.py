import os
import numpy as np
import xarray as xr
import vtk
from vtk.util import numpy_support


# -----------------------------
# USER SETTINGS
# -----------------------------
NC_PATH = "data/CVM17_L3.nc"
OUTPUT_DIR = "outputs/baseline"
SCREENSHOT_PATH = os.path.join(OUTPUT_DIR, "baseline_render.png")

# First run with AUTO_DETECT_VAR = True
AUTO_DETECT_VAR = True

# If AUTO_DETECT_VAR = False, put your chosen 3D scalar variable name here
SCALAR_VAR_NAME = None  # example: "Vs"

# Cropping ratios for a smaller interactive subvolume
# These are fractions of the full volume dimensions
CROP_X = (0.20, 0.75)
CROP_Y = (0.20, 0.75)
CROP_Z = (0.20, 0.75)

# Optional downsample step after crop
DOWNSAMPLE = 1   # use 1 first; later 2 or 3 if too slow

# Invalid/masked background values
INVALID_VALUES = [-999, -999.0]

# Window size
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900


# -----------------------------
# HELPERS
# -----------------------------


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def inspect_dataset(ds):
    print("\n=== DATASET SUMMARY ===")
    print(ds)

    print("\n=== VARIABLES ===")
    for var_name, da in ds.data_vars.items():
        print(f"\nVariable: {var_name}")
        print(f"  dims   : {da.dims}")
        print(f"  shape  : {da.shape}")
        print(f"  dtype  : {da.dtype}")


def find_candidate_3d_vars(ds):
    candidates = []
    for var_name, da in ds.data_vars.items():
        if len(da.shape) == 3 and np.issubdtype(da.dtype, np.number):
            candidates.append(var_name)
    return candidates


def choose_variable(ds):
    candidates = find_candidate_3d_vars(ds)

    if not candidates:
        raise ValueError("No numeric 3D variables found in the netCDF file.")

    print("\n=== CANDIDATE 3D VARIABLES ===")
    for i, name in enumerate(candidates):
        print(f"[{i}] {name}  shape={ds[name].shape}")

    if AUTO_DETECT_VAR:
        chosen = candidates[0]
        print(f"\nAUTO_DETECT_VAR=True -> temporarily selecting: {chosen}")
        print("If this is not the right variable, copy its correct name into SCALAR_VAR_NAME and rerun.")
        return chosen

    if SCALAR_VAR_NAME is None:
        raise ValueError("Set SCALAR_VAR_NAME to one of the candidate 3D variable names.")
    if SCALAR_VAR_NAME not in candidates:
        raise ValueError(f"SCALAR_VAR_NAME='{SCALAR_VAR_NAME}' is not a valid 3D numeric variable.")
    return SCALAR_VAR_NAME


def crop_volume(volume, crop_x, crop_y, crop_z):
    nx, ny, nz = volume.shape

    x0, x1 = int(nx * crop_x[0]), int(nx * crop_x[1])
    y0, y1 = int(ny * crop_y[0]), int(ny * crop_y[1])
    z0, z1 = int(nz * crop_z[0]), int(nz * crop_z[1])

    cropped = volume[x0:x1, y0:y1, z0:z1]
    return cropped


def sanitize_volume(volume):
    vol = np.array(volume, dtype=np.float32)

    # Replace invalid markers with NaN
    for bad in INVALID_VALUES:
        vol[vol == bad] = np.nan

    # Replace inf with NaN
    vol[~np.isfinite(vol)] = np.nan

    # If too many NaNs, fail early
    nan_ratio = np.isnan(vol).mean()
    print(f"\nNaN ratio after invalid-value cleanup: {nan_ratio:.4f}")

    # Fill NaNs with lower percentile background
    valid = vol[~np.isnan(vol)]
    if valid.size == 0:
        raise ValueError("All values became invalid after sanitization.")

    fill_value = np.percentile(valid, 1)
    vol = np.where(np.isnan(vol), fill_value, vol)

    return vol


def normalize_volume(volume):
    vmin = np.min(volume)
    vmax = np.max(volume)
    print(f"Before normalization: min={vmin:.4f}, max={vmax:.4f}")

    if np.isclose(vmax, vmin):
        raise ValueError("Volume has near-constant values; cannot normalize.")

    norm = (volume - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)

    print(f"After normalization: min={norm.min():.4f}, max={norm.max():.4f}")
    return norm, vmin, vmax


def volume_to_vtk_image(volume):
    """
    Input volume shape expected as (X, Y, Z)
    VTK image dimensions are set as (X, Y, Z)
    """
    nx, ny, nz = volume.shape

    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(nx, ny, nz)
    vtk_img.SetSpacing(1.0, 1.0, 1.0)
    vtk_img.SetOrigin(0.0, 0.0, 0.0)

    # Flatten in Fortran order so VTK interprets the 3D volume correctly
    flat = np.ravel(volume, order="F")
    vtk_array = numpy_support.numpy_to_vtk(
        num_array=flat,
        deep=True,
        array_type=vtk.VTK_FLOAT
    )
    vtk_array.SetName("Scalars")

    vtk_img.GetPointData().SetScalars(vtk_array)
    return vtk_img


def create_baseline_transfer_functions():
    # Color transfer function
    color_tf = vtk.vtkColorTransferFunction()
    color_tf.AddRGBPoint(0.00, 0.05, 0.05, 0.08)   # dark bluish
    color_tf.AddRGBPoint(0.30, 0.20, 0.35, 0.55)
    color_tf.AddRGBPoint(0.55, 0.55, 0.65, 0.70)
    color_tf.AddRGBPoint(0.80, 0.85, 0.75, 0.50)
    color_tf.AddRGBPoint(1.00, 0.95, 0.95, 0.95)

    # Opacity transfer function
    opacity_tf = vtk.vtkPiecewiseFunction()
    opacity_tf.AddPoint(0.00, 0.00)
    opacity_tf.AddPoint(0.10, 0.00)
    opacity_tf.AddPoint(0.25, 0.03)
    opacity_tf.AddPoint(0.45, 0.08)
    opacity_tf.AddPoint(0.65, 0.18)
    opacity_tf.AddPoint(0.85, 0.35)
    opacity_tf.AddPoint(1.00, 0.55)

    return color_tf, opacity_tf


def create_volume_property(color_tf, opacity_tf):
    prop = vtk.vtkVolumeProperty()
    prop.SetColor(color_tf)
    prop.SetScalarOpacity(opacity_tf)
    prop.SetInterpolationTypeToLinear()
    prop.ShadeOn()

    # Basic shading
    prop.SetAmbient(0.20)
    prop.SetDiffuse(0.75)
    prop.SetSpecular(0.15)
    prop.SetSpecularPower(12.0)

    return prop


def create_volume_mapper(vtk_img):
    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputData(vtk_img)

    # Baseline sampling
    mapper.SetAutoAdjustSampleDistances(0)
    mapper.SetSampleDistance(0.8)
    mapper.SetBlendModeToComposite()

    return mapper


def save_screenshot(render_window, filepath):
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filepath)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()


def main():
    ensure_dir(OUTPUT_DIR)

    if not os.path.exists(NC_PATH):
        raise FileNotFoundError(f"Could not find file: {NC_PATH}")

    print(f"Loading netCDF file: {NC_PATH}")
    ds = xr.open_dataset(NC_PATH)

    inspect_dataset(ds)
    var_name = choose_variable(ds)

    print(f"\nSelected variable: {var_name}")
    da = ds[var_name]

    # Convert to numpy
    vol = da.values
    print(f"Original raw shape: {vol.shape}")

    if vol.ndim != 3:
        raise ValueError(f"Selected variable '{var_name}' is not 3D.")

    # Convert to float and sanitize
    vol = sanitize_volume(vol)

    # Crop
    vol = crop_volume(vol, CROP_X, CROP_Y, CROP_Z)
    print(f"Shape after crop: {vol.shape}")

    # Downsample
    if DOWNSAMPLE > 1:
        vol = vol[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE]
        print(f"Shape after downsample x{DOWNSAMPLE}: {vol.shape}")

    # Normalize
    vol, orig_min, orig_max = normalize_volume(vol)

    # Build vtkImageData
    vtk_img = volume_to_vtk_image(vol)

    # Transfer functions and property
    color_tf, opacity_tf = create_baseline_transfer_functions()
    vol_prop = create_volume_property(color_tf, opacity_tf)

    # Mapper and volume actor
    mapper = create_volume_mapper(vtk_img)

    volume_actor = vtk.vtkVolume()
    volume_actor.SetMapper(mapper)
    volume_actor.SetProperty(vol_prop)

    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume_actor)
    renderer.SetBackground(0.08, 0.08, 0.10)

    # Render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(WINDOW_WIDTH, WINDOW_HEIGHT)
    render_window.SetWindowName("Baseline DVR Render")

    # Interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Camera setup
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.Azimuth(35)
    camera.Elevation(20)
    camera.Zoom(1.2)
    renderer.ResetCameraClippingRange()

    # First render
    render_window.Render()

    # Save screenshot
    save_screenshot(render_window, SCREENSHOT_PATH)
    print(f"\nSaved baseline screenshot to: {SCREENSHOT_PATH}")

    print("\nControls:")
    print("- Drag with mouse to rotate")
    print("- Scroll to zoom")
    print("- Close the VTK window when done")

    interactor.Initialize()
    interactor.Start()


if __name__ == "__main__":
    main()