import numpy as np
import matplotlib.pyplot as plt
from functions import File_Functions, Video_Functions, Audio_Functions

def calculate_reprojection_error(gcps, homography_matrix, x_range=None, cam=0):
    """
    Calculate the reprojection error given ground control points and a homography matrix.
    
    Args:
        gcps (tuple): A tuple containing two sets of points. 
                      gcps[0] - Real-world points (x, y)
                      gcps[1] - Image points (x, y)
        homography_matrix (numpy array): 3x3 Homography matrix for transformation.
        x_range (tuple, optional): A tuple specifying the range of x coordinates (min_x, max_x) to filter the real-world points.
    
    Returns:
        float: Mean reprojection error for the selected points.
    """

    # Convert to float32 numpy arrays
    img_pts = np.array(gcps[1], dtype=np.float32)
    rw_pts = np.array(gcps[0], dtype=np.float32)

    # Add a third dimension to the image points (homogeneous coordinates)
    img_pts = np.column_stack((img_pts, np.ones((img_pts.shape[0], 1))))

    # Apply homography to the source points
    projected_pts = homography_matrix @ img_pts.T

    # Normalize the projected points by dividing by the third coordinate
    projected_pts = projected_pts[:2] / projected_pts[2]

    # Transpose projected points for easier manipulation (back to Nx2)
    projected_pts = projected_pts.T
    
    # If an x_range is provided, filter points based on the real-world x coordinate
    if x_range is not None:
        min_x, max_x = x_range
        # Create a mask based on the x-coordinates of real-world points
        mask = (rw_pts[:, 0] >= min_x) & (rw_pts[:, 0] <= max_x)
        
        # Apply the mask to real-world points
        rw_pts = rw_pts[mask]
        
        # Apply the same mask to the corresponding projected points
        projected_pts = projected_pts[mask]

    # Calculate the Euclidean distance between actual and projected points
    errors = np.linalg.norm(rw_pts - projected_pts, axis=1)

    # Return the average error
    mean_error = np.mean(errors)
    print(f"Reprojection Error: {mean_error}")

    # Call the function to visualize points
    visualize_points(rw_pts, projected_pts, title="Camera 1 Reprojection")
    
    return mean_error

def visualize_points(rw_pts, projected_pts, title="Reprojection Visualization"):
    """
    Visualizes real-world points and projected points on a 2D plot.
    
    Args:
        rw_pts (numpy array): Real-world points (Nx2).
        projected_pts (numpy array): Projected points (Nx2) after applying the homography.
        title (str): Title of the plot.
    """
    # Ensure the points are in the correct shape
    rw_pts = np.array(rw_pts)
    projected_pts = np.array(projected_pts)
    
    plt.figure(figsize=(8, 8))
    
    # Plot real-world points
    plt.scatter(rw_pts[:, 0], rw_pts[:, 1], color='blue', label='Real-world points', s=50)
    
    # Plot projected points
    plt.scatter(projected_pts[:, 0], projected_pts[:, 1], color='red', label='Projected points', s=50)
    
    # Draw lines between corresponding points to show error
    for i in range(len(rw_pts)):
        plt.plot([rw_pts[i, 0], projected_pts[i, 0]], [rw_pts[i, 1], projected_pts[i, 1]], 'gray', linestyle='--')

    # Set plot labels and title
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Show the plot
    plt.show()

if __name__ == "__main__":

    #instantiate classes
    ff = File_Functions()
    af = Audio_Functions()
    vf = Video_Functions()

    #load gcps files:
    gcps1 = ff.load_fn("Select gcps file for camera 1")
    gcps2 = ff.load_fn("Select gcps file for camera 2")
    gcps3 = ff.load_fn("Select gcps file for camera 3")
    gcps4 = ff.load_fn("Select gcps file for camera 4")

    gcpss = [gcps1, gcps2, gcps3, gcps4]

    targets = []
    for i, gcps in enumerate(gcpss):
        gcps = ff.import_gcps(gcps)
        targets.append(gcps)

        print("GCPS as it goes into targets list: ", gcps[0])

        #Generate homography matricies
    homo_mats = []
    for i, target in enumerate(targets):
        print("Target as as the for loop begins:", target[0])
        x_range = (0, 2438)

        homography = vf.find_homography(i+1, target)
        homo_mats.append(homography)

        print("Target after homography happens: ", target[0])
        calculate_reprojection_error(target, homography, x_range = x_range, cam = i)
