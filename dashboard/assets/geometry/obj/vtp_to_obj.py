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

    # Generate output path
    base_name = os.path.splitext(os.path.basename(vtp_file))[0]
    obj_file_path = os.path.join(output_dir, f"{base_name}.obj")

    # Write to OBJ
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(obj_file_path)
    writer.SetInputData(triangulated)
    writer.Write()

    print(f"Saved to {obj_file_path}")

