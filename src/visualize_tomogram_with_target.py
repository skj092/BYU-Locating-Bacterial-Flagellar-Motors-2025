from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
import os
import pandas as pd
import random

# Load dataset
path = Path('data')  # update if needed
train = pd.read_csv(path / 'train_labels.csv')
train = train[train['Number of motors'] > 1]

# Pick a random tomogram with at least one motor
tomo_ids = train['tomo_id'].unique().tolist()
tomo_id = random.choice(tomo_ids)
# tomo_id = 'tomo_00e047'
# tomo_id = 'tomo_00e463'
print(f'Selected tomogram: {tomo_id}')
print(train[train['tomo_id'] == tomo_id])


# Load image slices
folder_path = path / 'train' / tomo_id
slice_files = sorted(
    [f for f in os.listdir(folder_path) if f.endswith('.jpg')])
slices = [imageio.imread(os.path.join(folder_path, f)) for f in slice_files]
volume = np.stack(slices, axis=0)

# Normalize and scale to 0–255 (invert contrast for better viewing)
volume = (1.0 - (volume - np.min(volume)) /
          (np.max(volume) - np.min(volume))) * 255
volume = volume.astype(np.uint8)

# Convert numpy array to VTK image data
vtk_data = numpy_to_vtk(volume.ravel(), deep=True,
                        array_type=vtk.VTK_UNSIGNED_CHAR)
image_data = vtk.vtkImageData()
image_data.SetDimensions(volume.shape[::-1])  # (x, y, z)
image_data.GetPointData().SetScalars(vtk_data)

# Setup VTK volume rendering
mapper = vtk.vtkSmartVolumeMapper()
mapper.SetInputData(image_data)

volume_property = vtk.vtkVolumeProperty()
composite_function = vtk.vtkPiecewiseFunction()
composite_function.AddPoint(0, 0.0)
composite_function.AddPoint(255, 0.01)  # adjust for contrast
volume_property.SetScalarOpacity(composite_function)

color = vtk.vtkColorTransferFunction()
color.AddRGBPoint(0, 0.0, 0.0, 0.0)
color.AddRGBPoint(255, 1.0, 1.0, 1.0)
volume_property.SetColor(color)

volume_actor = vtk.vtkVolume()
volume_actor.SetMapper(mapper)
volume_actor.SetProperty(volume_property)

# Setup renderer
renderer = vtk.vtkRenderer()
renderer.AddVolume(volume_actor)
renderer.SetBackground(0, 0, 0)

# Overlay motor target(s) as red spheres
motor_coords = train[train['tomo_id'] == tomo_id][[
    'Motor axis 2', 'Motor axis 1', 'Motor axis 0']].values  # (x, y, z)

for x, y, z in motor_coords:
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(x, y, z)
    sphere.SetRadius(10)  # Tune as needed
    sphere.Update()

    mapper_sphere = vtk.vtkPolyDataMapper()
    mapper_sphere.SetInputConnection(sphere.GetOutputPort())

    actor_sphere = vtk.vtkActor()
    actor_sphere.SetMapper(mapper_sphere)
    actor_sphere.GetProperty().SetColor(1, 0, 0)  # Red
    actor_sphere.GetProperty().SetOpacity(1.0)

    renderer.AddActor(actor_sphere)

# Setup render window
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(1280, 720)

# Interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
interactor_style = vtk.vtkInteractorStyleTrackballCamera()
interactor.SetInteractorStyle(interactor_style)

# Start rendering
render_window.Render()
interactor.Start()
