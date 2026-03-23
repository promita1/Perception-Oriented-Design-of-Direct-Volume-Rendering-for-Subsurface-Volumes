import os
import numpy as np
import xarray as xr
import vtk
from vtk.util import numpy_support

# =========================================================
# USER SETTINGS
# =========================================================
NC_PATH = "data/CVM17_L3.nc"
SCALAR_VAR_NAME = "vs"

OUTPUT_DIR = "outputs/interactive"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CROP_X = (0.20, 0.75)
CROP_Y = (0.20, 0.75)
CROP_Z = (0.10, 0.90)
DOWNSAMPLE = 1

INVALID_VALUES = [-999, -999.0]

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
BACKGROUND_RGB = (0.08, 0.08, 0.10)

# =========================================================
# PREPROCESSING
# =========================================================
def sanitize_volume(volume):
    vol = np.array(volume, dtype=np.float32)
    for bad in INVALID_VALUES:
        vol[vol == bad] = np.nan
    vol[~np.isfinite(vol)] = np.nan

    valid = vol[~np.isnan(vol)]
    if valid.size == 0:
        raise ValueError("All values are invalid after sanitization.")

    fill_value = np.percentile(valid, 1)
    vol = np.where(np.isnan(vol), fill_value, vol)
    return vol


def crop_volume(volume):
    nx, ny, nz = volume.shape
    x0, x1 = int(nx * CROP_X[0]), int(nx * CROP_X[1])
    y0, y1 = int(ny * CROP_Y[0]), int(ny * CROP_Y[1])
    z0, z1 = int(nz * CROP_Z[0]), int(nz * CROP_Z[1])
    return volume[x0:x1, y0:y1, z0:z1]


def normalize_volume(volume):
    vmin = float(np.min(volume))
    vmax = float(np.max(volume))
    if np.isclose(vmin, vmax):
        raise ValueError("Volume has near-constant values.")
    norm = (volume - vmin) / (vmax - vmin)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def volume_to_vtk_image(volume):
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


def load_preprocessed_vtk_image():
    ds = xr.open_dataset(NC_PATH)
    if SCALAR_VAR_NAME not in ds.data_vars:
        raise ValueError(f"Variable '{SCALAR_VAR_NAME}' not found in dataset.")

    vol = ds[SCALAR_VAR_NAME].values
    vol = sanitize_volume(vol)
    vol = crop_volume(vol)

    if DOWNSAMPLE > 1:
        vol = vol[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE]

    vol = normalize_volume(vol)
    return volume_to_vtk_image(vol)


# =========================================================
# PRESETS
# =========================================================
TF_NAMES = ["tf1_baseline", "tf2_boundary", "tf3_continuity"]
SAMPLE_NAMES = ["low", "medium", "high"]
SHADING_NAMES = ["off", "basic", "depth"]


def build_transfer_function(tf_name):
    color_tf = vtk.vtkColorTransferFunction()
    opacity_tf = vtk.vtkPiecewiseFunction()

    if tf_name == "tf1_baseline":
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
        raise ValueError(f"Unknown transfer function: {tf_name}")

    return color_tf, opacity_tf


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


def save_screenshot(render_window, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.ReadFrontBufferOff()
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filepath)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()

    print(f"Saved screenshot: {filepath}")


# =========================================================
# INTERACTIVE CONTROLLER
# =========================================================
class DVRController:
    def __init__(self, vtk_img):
        self.vtk_img = vtk_img

        self.tf_idx = 0
        self.sample_idx = 1
        self.shading_idx = 1

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*BACKGROUND_RGB)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.render_window.SetWindowName("Interactive DVR Prototype")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetDisplayPosition(20, 20)
        self.text_actor.GetTextProperty().SetFontSize(20)
        self.text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        self.renderer.AddActor2D(self.text_actor)

        self.volume_actor = None
        self.camera_initialized = False

        self.update_volume()

        self.interactor.AddObserver("KeyPressEvent", self.on_key_press)

    def current_tf(self):
        return TF_NAMES[self.tf_idx]

    def current_sample(self):
        return SAMPLE_NAMES[self.sample_idx]

    def current_shading(self):
        return SHADING_NAMES[self.shading_idx]

    def update_text(self):
        msg = (
            f"TF: {self.current_tf()} | "
            f"Sampling: {self.current_sample()} | "
            f"Shading: {self.current_shading()}\n"
            f"Keys: 1=TF  2=Sampling  3=Shading  S=Save Screenshot"
        )
        self.text_actor.SetInput(msg)

    def update_volume(self):
        if self.volume_actor is not None:
            self.renderer.RemoveVolume(self.volume_actor)

        color_tf, opacity_tf = build_transfer_function(self.current_tf())
        mapper = build_mapper(self.vtk_img, self.current_sample())
        prop = build_volume_property(color_tf, opacity_tf, self.current_shading())

        self.volume_actor = vtk.vtkVolume()
        self.volume_actor.SetMapper(mapper)
        self.volume_actor.SetProperty(prop)

        self.renderer.AddVolume(self.volume_actor)
        self.update_text()

        if not self.camera_initialized:
            self.renderer.ResetCamera()
            camera = self.renderer.GetActiveCamera()
            camera.Azimuth(35)
            camera.Elevation(20)
            camera.Zoom(1.2)
            self.renderer.ResetCameraClippingRange()
            self.camera_initialized = True

        self.render_window.Render()

    def on_key_press(self, obj, event):
        key = self.interactor.GetKeySym().lower()

        if key == "1":
            self.tf_idx = (self.tf_idx + 1) % len(TF_NAMES)
            print(f"Switched TF -> {self.current_tf()}")
            self.update_volume()

        elif key == "2":
            self.sample_idx = (self.sample_idx + 1) % len(SAMPLE_NAMES)
            print(f"Switched Sampling -> {self.current_sample()}")
            self.update_volume()

        elif key == "3":
            self.shading_idx = (self.shading_idx + 1) % len(SHADING_NAMES)
            print(f"Switched Shading -> {self.current_shading()}")
            self.update_volume()

        elif key == "s":
            filename = (
                f"{self.current_tf()}__{self.current_sample()}__{self.current_shading()}__interactive.png"
            )
            save_screenshot(self.render_window, filename)

        self.interactor.Render()

    def start(self):
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()


def main():
    vtk_img = load_preprocessed_vtk_image()
    app = DVRController(vtk_img)
    app.start()


if __name__ == "__main__":
    main()