import warnings
import numpy as np

try:
    import vtk
except ImportError:
    raise ImportError("VTK 7 with Python wrappers is required")
try:
    from dipy.viz import fvtk
except ImportError:
    raise ImportError("Currently requires DIPY fvtk")

if vtk.VTK_MAJOR_VERSION < 7:
    warnings.warn("Only tested with VTK 7.0")


def vtk_volume_mip(vol, opacity_level=2048, opacity_window=4096,
                   trilinear=True):
    im = vtk.vtkImageData()
    im.SetDimensions(vol.shape[0], vol.shape[1], vol.shape[2])
    im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
    for i in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for k in range(vol.shape[2]):
                im.SetScalarComponentFromFloat(i, j, k, 0, vol[i, j, k])

    color = vtk.vtkColorTransferFunction()
    opacity = vtk.vtkPiecewiseFunction()

    prop = vtk.vtkVolumeProperty()
    prop.SetColor(color)
    prop.SetScalarOpacity(opacity)
    prop.ShadeOn()
    if trilinear:
        prop.SetInterpolationTypeToLinear()
    else:
        prop.SetInterpolationTypeToNearest()

    mapper = vtk.vtkSmartVolumeMapper()
    #mapper.SetRequestedRenderMode(mapper.DefaultRenderMode)
    mapper.SetRequestedRenderModeToDefault()
    mapper.SetInputData(im)


    # set color of MIP to white
    color.AddRGBSegment(0.0, 1.0, 1.0, 1.0, 255.0, 1.0, 1.0, 1.0 )
    # configure opacity ramp based on specified window and level
    opacity.AddSegment(opacity_level - 0.5*opacity_window, 0.0,
                       opacity_level + 0.5*opacity_window, 1.0 )
    # set the mapper to create a MIP
    mapper.SetBlendModeToMaximumIntensity()

    vtkvol = vtk.vtkVolume()
    vtkvol.SetMapper(mapper)
    vtkvol.SetProperty(prop)
    return vtkvol


def test_vtkvol():
    vol = np.load('/home/lee8rx/my_git/pyvolplot/examples/angio3D_downsampled_abs.npz')['angio3D']
    opacity_level = 2048
    opacity_window = 4096
    trilinear = True
    size = [600, 600]
    frame_rate = 10
    vol = opacity_window*vol/vol.max()
    vtkvol = vtk_volume_mip(vol, opacity_level=opacity_level,
                            opacity_window=opacity_window, trilinear=trilinear)
    if True:
        from dipy.viz import fvtk
        r = fvtk.ren()
        r.add(vtkvol)
        fvtk.show(r)
    else:
        ren = vtk.vtkRenderer()
        ren.AddVolume(vtkvol)
        window = vtk.vtkRenderWindow()
        window.AddRenderer(ren)
        window.SetSize(size[0], size[1])

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(window)
        iren.SetDesiredUpdateRate(frame_rate)
        iren.GetInteractorStyle().SetDefaultRenderer(ren)

        window.Render()
        iren.Start()
        del ren, window, iren
