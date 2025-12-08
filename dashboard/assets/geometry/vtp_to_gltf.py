import vtk
import os
import glob

# Create 'out' folder if it doesn't exist
output_dir = "out"
os.makedirs(output_dir, exist_ok=True)

# Get all .vtp files in the current directory
vtp_files = glob.glob("*.vtp")

for vtp_file in vtp_files:
    print(f"Processing {vtp_file}...")

    # Read the VTP file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file)
    reader.Update()
    polydata = reader.GetOutput()

    # Triangulate the mesh
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(polydata)
    triangle_filter.Update()
    triangulated = triangle_filter.GetOutput()

    # Create mapper and actor for the polydata
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(triangulated)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create renderer and add actor
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # white background

    # Create render window (needed by vtkGLTFExporter)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

    # Setup GLTF exporter
    exporter = vtk.vtkGLTFExporter()

    # Output filename
    base_name = os.path.splitext(os.path.basename(vtp_file))[0]
    gltf_file_path = os.path.join(output_dir, f"{base_name}.gltf")

    exporter.SetFileName(gltf_file_path)
    exporter.SetRenderWindow(render_window)
    exporter.Write()

    print(f"Saved to {gltf_file_path}")

