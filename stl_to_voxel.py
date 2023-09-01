import subprocess
import os
import glob
import sys
import argparse


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import numpy as np
except ModuleNotFoundError:
    install_package("numpy")
    import numpy as np

try:
    from scipy.io import savemat, loadmat
except ModuleNotFoundError:
    install_package("scipy")
    from scipy.io import savemat

try:
    import open3d as o3d
except ModuleNotFoundError:
    install_package("open3d")
    import open3d as o3d

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    install_package("tqdm")
    from tqdm import tqdm

def convert_stl_folder_to_voxel(input_folder: str, output_folder: str) -> None:
    """
    Convert all .stl files in the input_folder to .mat files and save them in the output_folder.
    
    :param input_folder: Path to the folder containing .stl files.
    :param output_folder: Path to the folder where .mat files will be saved.
    """
    
    # Check if output_folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all .stl files in the input_folder
    for filepath in tqdm(glob.glob(os.path.join(input_folder, '*.stl')), desc="Conversion Progress"):
        
        # Read the STL file using Open3D
        mesh = o3d.io.read_triangle_mesh(filepath)
        mesh.compute_vertex_normals()

        # Fit the mesh to a unit cube
        mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
                   center=mesh.get_center())

        # Create a voxel grid from the mesh
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.009)

        # Initialize voxel_matrix as zeros
        voxel_matrix = np.zeros((128, 128, 128), dtype=int)
        
        # Fill in the voxel_matrix based on the voxel grid
        for voxel in voxel_grid.get_voxels():
            x, y, z = voxel.grid_index
            voxel_matrix[x, y, z] = 1

        # Generate output file path
        output_filepath = os.path.join(output_folder, os.path.basename(filepath).replace('.stl', '.mat'))

        # Save the voxel_matrix as a .mat file
        savemat(output_filepath, {'volume': voxel_matrix})

    print("Conversion Complete")
    print(f"Files Saved in {output_folder}")

def visualize_mat_file(mat_filepath: str) -> None:
    """
    Visualize a .mat file containing a voxel grid using Open3D.
    
    :param mat_filepath: The path to the .mat file to visualize.
    """
    
    # Load .mat file
    mat_data = loadmat(mat_filepath)
    voxel_matrix = mat_data['volume']

    # Create point cloud data for VoxelGrid
    points = []
    for x in range(voxel_matrix.shape[0]):
        for y in range(voxel_matrix.shape[1]):
            for z in range(voxel_matrix.shape[2]):
                if voxel_matrix[x, y, z] == 1:
                    points.append([x, y, z])
    
    # Convert to point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    
    # Create VoxelGrid from point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
    
    # Visualize the VoxelGrid
    o3d.visualization.draw_geometries([voxel_grid])

def main():
    parser = argparse.ArgumentParser(description='Convert STL files in a folder to MAT files and/or visualize a MAT file.')
    parser.add_argument('--input_folder', type=str, help='Path to the folder containing .stl files.', default=None)
    parser.add_argument('--output_folder', type=str, help='Path to the folder where .mat files will be saved.', default=None)
    parser.add_argument('--vis', type=str, help='Path to a .mat file to visualize.', default=None)

    args = parser.parse_args()

    # If both input_folder and output_folder are provided, convert STL to MAT
    if args.input_folder and args.output_folder:
        convert_stl_folder_to_voxel(args.input_folder, args.output_folder)

    # If only the --vis argument is provided, visualize the .mat file
    elif args.vis:
        visualize_mat_file(args.vis)

    else:
        print("Please provide either input_folder and output_folder for conversion or --vis for visualization.")

if __name__ == "__main__":
    main()
