from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
import os
import pandas as pd
import random

path = Path(
    '~/code/BYU-Locating-Bacterial-Flagellar-Motors-2025/data').expanduser()

train = pd.read_csv(path/'train_labels.csv')

tomo_ids = train.tomo_id.to_list()
tomo_id = random.choice(tomo_ids)
# tomo_id = "tomo_00e463"
print('selected tomo id:', tomo_id)
folder_path = f"{path}/train/{tomo_id}"

print(train[train['tomo_id'] == tomo_id])

slice_files = sorted([file for file in os.listdir(
    folder_path) if file.endswith('.jpg')])
slices = [imageio.imread(os.path.join(folder_path, file))
          for file in slice_files]

volume = np.stack(slices, axis=0)
volume = (1.0 - (volume - np.min(volume)) /
          (np.max(volume) - np.min(volume))) * 255
volume = volume.astype(np.uint8)

vtk_data = numpy_to_vtk(volume.ravel(), deep=True,
                        array_type=vtk.VTK_UNSIGNED_CHAR)
image_data = vtk.vtkImageData()
image_data.SetDimensions(volume.shape[::-1])
image_data.GetPointData().SetScalars(vtk_data)

mapper = vtk.vtkSmartVolumeMapper()
mapper.SetInputData(image_data)

volume_property = vtk.vtkVolumeProperty()
composite_function = vtk.vtkPiecewiseFunction()
composite_function.AddPoint(0, 0.0)
composite_function.AddPoint(255, 0.01)
volume_property.SetScalarOpacity(composite_function)

color = vtk.vtkColorTransferFunction()
color.AddRGBPoint(0, 0.0, 0.0, 0.0)
color.AddRGBPoint(255, 1.0, 1.0, 1.0)
volume_property.SetColor(color)

volume_actor = vtk.vtkVolume()
volume_actor.SetMapper(mapper)
volume_actor.SetProperty(volume_property)

renderer = vtk.vtkRenderer()
renderer.AddVolume(volume_actor)
renderer.SetBackground(0, 0, 0)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(1280, 720)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

interactor_style = vtk.vtkInteractorStyleTrackballCamera()
interactor.SetInteractorStyle(interactor_style)

render_window.Render()
interactor.Start()
