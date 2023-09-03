import matplotlib.pyplot as plt 
import numpy as np
# import nibabel as ni 
import torch
import open3d as o3d
from matplotlib.widgets import Slider

# vis = o3d.visualization.VisualizerWithViewports()


def save_and_display_image(x, y):
    assert x.shape == y.shape
    x = torch.reshape(x, (x.shape[0], 128, 128, 128)).cpu().detach().numpy()
    y = torch.reshape(y, (y.shape[0], 128, 128, 128)).cpu().detach().numpy()
    y[y<2] = 0
    print(x.shape, y.shape)
    # Create voxel grid
    for i in range(len(x)):
        x_indices, y_indices = np.transpose(np.nonzero(x[i])), np.transpose(np.nonzero(y[i]))
        x_cloud, y_cloud = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        x_cloud.points, y_cloud.points = o3d.utility.Vector3dVector(x_indices), o3d.utility.Vector3dVector(y_indices)

        # Display voxel grid
        o3d.visualization.draw_geometries([y_cloud])
    # print(np.min(y[0]), np.max(y[0]))

# def display_image(x, y):
#     assert x.shape == y.shape
#     if x.dim() == 5:
#         x = torch.reshape(x, (x.shape[0], 128, 128, 128)).cpu().detach().numpy()
#         y = torch.reshape(y, (y.shape[0], 128, 128, 128)).cpu().detach().numpy()
#         for i in range(len(x)):
#             rows = 2
#             columns = 1
#             fig=plt.figure()
#             for idx in range(rows*columns):
#                 fig.add_subplot(rows, columns, idx+1)
#                 if idx < columns:
#                     plt.imshow(x[i, :, 60, :], cmap="gray", origin="lower")
#                 else:
#                     plt.imshow(y[i, :, 60, :], cmap="gray", origin="lower")
#             plt.show()
# def display_image(x, y):
#     assert x.shape == y.shape
#     if x.dim() == 5:
#         x = torch.reshape(x, (x.shape[0], 128, 128, 128)).cpu().detach().numpy()
#         y = torch.reshape(y, (y.shape[0], 128, 128, 128)).cpu().detach().numpy()
#         y[y<2] = 0
#         fig, ax = plt.subplots(2, 1)
#         plt.subplots_adjust(bottom=0.25)

#         i_slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
#         i_slider = Slider(i_slider_ax, 'Index', 0, 127, valinit=0, valstep=1)

#         def update(val):
#             i = int(i_slider.val)
#             ax[0].clear()
#             ax[1].clear()
#             ax[0].imshow(x[0, :, i, :], cmap="gray", origin="lower")
#             ax[1].imshow(y[0, :, i, :], cmap="gray", origin="lower")
#             ax[0].set_title("Input Image")
#             ax[1].set_title("Output Image")
#             fig.canvas.draw()

#         i_slider.on_changed(update)
#         update(0)  # Initial update

#         plt.show()