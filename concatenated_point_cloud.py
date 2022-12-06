from plot_utils import plot_pointcloud
from CLIP.clip import ClipWrapper, saliency_configs, imagenet_templates
import imageio
import typer
from pprint import pprint 
import numpy as np
from point_cloud import transform_pointcloud
import pickle
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

def generate_relevancy_image(img, labels, prompts):
    """
    Generates a multi-scale relevancy for image at `file_path`
    """
    assert img.dtype == np.uint8
    h, w, c = img.shape
    grads = ClipWrapper.get_clip_saliency(
        img=img,
        text_labels=np.array(labels),
        prompts=prompts,
        **saliency_configs["ours"](h),
    )[0]
    
    if(len(labels)>1):
        grads -= grads.mean(axis=0)
    
    grads = grads.cpu().numpy()
    return grads
    
def get_depth_pointcloud(depth_img, cam_intr, cam_pose=None):
    """Get 3D pointcloud from depth image.

    Args:
        depth_img_path: the file for the HxW float array of depth values in meters aligned with color_img
        cam_intr: the file for 3x3 float array of camera intrinsic parameters
        cam_pose: (optional) 3x4 float array of camera pose matrix

    Returns:
        cam_pts: Nx3 float array of 3D points in camera/world coordinates
    """
    
    img_h = depth_img.shape[0]
    img_w = depth_img.shape[1]

    # Project depth into 3D pointcloud in camera coordinates
    pixel_x, pixel_y = np.meshgrid(
        np.linspace(0, img_w - 1, img_w), np.linspace(0, img_h - 1, img_h)
    )
    cam_pts_x = np.multiply(pixel_x - cam_intr[0, 2], depth_img / cam_intr[0, 0])
    cam_pts_y = np.multiply(pixel_y - cam_intr[1, 2], depth_img / cam_intr[1, 1])
    cam_pts_z = depth_img
    cam_pts = (np.array([cam_pts_x, cam_pts_y, cam_pts_z]).transpose(1, 2, 0).reshape(-1, 3))

    if cam_pose is not None:
        cam_pts = transform_pointcloud(cam_pts, cam_pose)
        
    cam_pts[:,[1]] = -cam_pts[:,[1]]
    cam_pts[:,[1, 2]] = cam_pts[:,[2, 1]]
    
    cam_pts -= cam_pts.mean(axis=0)
        
    min_dist = np.inf
    min_point = None
    for i, pt in enumerate(cam_pts):
        r_dist = np.linalg.norm(pt-np.array([0,0,0]))
        if(r_dist<min_dist):
            min_dist = min(min_dist, r_dist)
            min_point = pt 
    
    cam_pts -= min_point
    
    return cam_pts


def return_cam_pts(depth_image, cam_intr, cam_pose):
    cam_pts = get_depth_pointcloud(depth_image, cam_intr, cam_pose)     
    return cam_pts
    
def return_relevancy_pts(image, labels, prompts):
    '''Visualizes the pointcloud 
    Args:
        relevancy_img_path: the file path for the relevancy image path
        depth_image_path: the file path for the depth values in meters aligned with relevancy_img_path
        cam_intr: the file for 3x3 float array of camera intrinsic parameters
        cam_pose: (optional) 3x4 float array of camera pose matrix
    '''
    image = np.array(image)
    relevancy_grads = generate_relevancy_image(image, labels, prompts)    
    relevancy_pts = relevancy_grads.flatten()

    return relevancy_pts


def visualize_point_cloud(cam_pts, relevancy_pts):
    plot_pointcloud(cam_pts, relevancy_pts,show_plot=True,num_points=1000000)