from math import sqrt
import os
import random

import PIL
import cv2
import numpy as np
import openslide as op
from scipy.special import softmax
from skimage.filters import threshold_otsu
from scipy.interpolate import LinearNDInterpolator


from src.region import Region, RegionNonRect


def correct_itk_transform(itk_transform):
    # Proper way to convert an itk-compatible affine transform to an openslide-compatible affine transform
    itk_transform[1,2] = -itk_transform[1,2]
    itk_transform[0,1],itk_transform[1,0] =  -itk_transform[0,1], -itk_transform[1,0]
    return itk_transform


def get_patch(slide, r):
    if isinstance(r, Region):
        return slide.read_region((r.x,r.y), r.lvl,(r.size_x, r.size_y))
    elif isinstance(r, RegionNonRect):
        return read_non_rectangular_region(slide, r.pointA, r.pointB, r.pointC, r.pointD, r.lvl, (r.size_x, r.size_y))


def display_overlapping_patches(input_patch, output_patch, rotate):
    output_patch = output_patch.convert('RGB')
    if rotate == 'clockwise':
        output_patch = output_patch.transpose(method=PIL.Image.TRANSPOSE)
        output_patch = output_patch.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
    elif rotate == 'anticlockwise':
        output_patch = output_patch.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
        output_patch = output_patch.transpose(method=PIL.Image.TRANSPOSE)
    elif rotate == '180':
        output_patch = output_patch.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
        output_patch = output_patch.transpose(method=PIL.Image.FLIP_TOP_BOTTOM)
    
    output_patch = output_patch.resize(input_patch.size, PIL.Image.BILINEAR)        

    input_patch_gray = input_patch.convert('L')
    output_patch_gray = output_patch.convert('L')

    np_input_patch_gray = np.array(input_patch_gray).astype(np.uint8)
    np_output_patch_gray = np.array(output_patch_gray).astype(np.uint8)

    grayscale_overlap = np.stack((np_input_patch_gray, np_input_patch_gray, np_output_patch_gray), axis=-1)

    return(PIL.Image.fromarray(grayscale_overlap.astype(np.uint8)))


def define_engulfing_region(region: Region, slide_level_downsamples):
    input_region_downsample = slide_level_downsamples[region.lvl]
    region_center_x = region.x + input_region_downsample*region.size_x/2
    region_center_y = region.y + input_region_downsample*region.size_y/2

    output_region_downsample = slide_level_downsamples[region.lvl+1]
    new_x = region_center_x - output_region_downsample*region.size_x/2
    new_y = region_center_y - output_region_downsample*region.size_y/2

    bigger_region = Region(round(new_x), round(new_y), region.lvl+1, region.size_x, region.size_y)
    return bigger_region


def get_random_valid_points(slidepath, slide_dimensions, N=10):
    slide = op.open_slide(slidepath)
    thb = mask(slide)
    x,y = np.nonzero(thb)
    
    # select random indices
    range_of_indices = list(range(0, len(x)-1))
    sample_points = random.sample(range_of_indices, N)

    # transform all coords to lvl0 absolute coords
    _, max_coord = sorted(slide_dimensions)
    coef_norm = max_coord/1000   # openslide has rescaled the bigger dimension of the slide to 1000 pixels
    base_change = [[coef_norm,0,0],
                   [0,coef_norm,0],
                   [0,0,1]]
    base_change = np.array(base_change)
    coords = [(base_change @ (y[i],x[i],1)).astype(int)[:2] for i in sample_points]
    return coords


def square_zone_generator(square_size=25):
    for x_start in range(0, 1000-square_size, square_size):
        x_end = x_start + square_size
        for y_start in range(0, 1000-square_size, square_size):
            y_end = y_start + square_size
            yield (x_start, x_end, y_start, y_end)


def is_in_square(X,xs,xe,ys,ye):
    x,y = X
    if xs<=x and x<=xe and ys<=y and y<=ye:
        return True
    return False


def sample_coords_uniformly(coords, square_size, N):
    uniformly_sampled_coords = []
    # select one point per square zone on the image so that the resulting distribution of points is homogenous across the slide
    for (xs, xe, ys, ye) in square_zone_generator(square_size=square_size):
        filtered_coords = list(filter(lambda X: is_in_square(X,xs,xe,ys,ye), coords))
        if len(filtered_coords) == 0:
            continue
        if len(filtered_coords) <= N:
            sampled_points = filtered_coords
        if len(filtered_coords) > N:
            sampled_points = random.sample(filtered_coords, N)
        uniformly_sampled_coords.extend(sampled_points)
    return uniformly_sampled_coords



def mask(slide):
    """
    Use Otsu thresholding to return a binary image where each pixel represents tissue

    Args:
      slide: Slide object
    Returns
      thb: A 1000x1000 binary mask representing tissue on the slide
    """
    thb = slide.get_thumbnail((1000,1000))
    thb = thb.convert('L')
    thresh = threshold_otsu(np.array(thb))
    thb = thb.point(lambda p: 255 if p <= thresh else 0)
    thb = np.array(thb)
    
    # Remove edges because they tend to show dark artifacts
    thb[:25,:] = 0
    thb[-25:,:] = 0
    thb[:,:25] = 0
    thb[:,-25:] = 0
    return thb


def get_uniformly_distributed_valid_points(slide, N_landmarks=100):
    thb = mask(slide)
    x,y = np.nonzero(thb)
    coords = list(zip(x,y))

    # Will try to produce around N_landmarks coords
    initial = 50
    uniformly_sampled_coords = sample_coords_uniformly(coords, square_size=initial, N=1)
    ratio = len(uniformly_sampled_coords)/N_landmarks
    real_square_size = round(initial*sqrt(ratio))
    uniformly_sampled_coords = sample_coords_uniformly(coords, square_size=real_square_size, N=1)

    # transform all coords to lvl0 absolute coords    
    _, max_coord = sorted(slide.dimensions)
    coef_norm = max_coord/1000  # openslide has rescaled the bigger dimension of the slide to 500 pixels
    base_change = [[coef_norm,0,0],
                   [0,coef_norm,0],
                   [0,0,1]]
    base_change = np.array(base_change)
    coords = [(base_change @ (y,x,1)).astype(int)[:2] for (x,y) in uniformly_sampled_coords]
    return coords


def find_smooth_adjustment(x,y,input_pts, residues):
    """
    Find local adjustment (translation) at specific location (x,y) using a softmax interpolation of the inverse of the distance
    """
    distances = np.abs(input_pts - [x,y,1])  # compute distance between (x,y) and each input point to find nearest neighbor
    inverse_distances = np.reciprocal(distances)
    weights = softmax(inverse_distances)

    corr = np.dot(weights, residues)

    nn_correction = [[1,0,-corr[0]],
                        [0,1,-corr[1]],
                        [0,0,1]]
    return np.array(nn_correction)


def find_nn_adjustment(x,y,input_pts, residues):
    """
    Find local adjustment (translation) at specific location (x,y) using nearest-neighbor method
    """
    distances = np.abs(input_pts - [x,y,1])  # compute distance between (x,y) and each input point to find nearest neighbor
    nn_index = np.argmin(np.linalg.norm(distances, axis=1))
    nn_correction = [[1,0,-residues[nn_index][0]],
                        [0,1,-residues[nn_index][1]],
                        [0,0,1]]
    return np.array(nn_correction)


def create_linear_interpolator(input_pts, residues):
    # add 0,0 residue for each corner of the image to help scipy extrapolate outside the convex hull
    # corners = np.array([[0,0],[0,h],[w,0],[w,h]])
    inf = 2**32
    corners = np.array([[-inf,-inf],[-inf,inf],[inf,-inf],[inf,inf]])
    corner_residues = np.zeros_like(corners)
    full_input_pts = np.concatenate([input_pts[:,:2], corners])
    full_residues = np.concatenate([residues[:,:2], corner_residues])
    return LinearNDInterpolator(full_input_pts, full_residues, fill_value=0, rescale=True)


def find_linear_adjustment(x,y,interpolator):
    """
    Find local adjustment (translation) at specific location (x,y) using a LinearNDInterpolator
    """
    corr = interpolator([x,y])[0]
    correction = np.array([[1,0,-corr[0]],
                        [0,1,-corr[1]],
                        [0,0,1]])
    return correction


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    offset = 10
    
    imga = np.array(imga).astype(np.uint8)[:,:,:3]
    imgb = np.array(imgb).astype(np.uint8)[:,:,:3]
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb+offset
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa+offset:wa+wb+offset]=imgb
    return PIL.Image.fromarray(new_img.astype(np.uint8))


def read_non_rectangular_region(s: op.OpenSlide, pointA, pointB, pointC, pointD, lvl, output_img_size):
    """
    ABCD declared like so:
    A ------------- B
    |               |
    |               |
    D ------------- C
    """

    d = s.level_downsamples[lvl]
    x_list = [p[0] for p in [pointA, pointB, pointC, pointD]]
    y_list = [p[1] for p in [pointA, pointB, pointC, pointD]]
    xmin = np.amin(x_list)
    xmax = np.amax(x_list)
    ymin = np.amin(y_list)
    ymax = np.amax(y_list)
    big_enough_patch = s.read_region((round(xmin),round(ymin)),lvl,((round((xmax-xmin)/d),round((ymax-ymin)/d))))

    # Now find the coordinates within the big_enough_patch. Keep in mind that the downsample must be considered
    origin = np.array([xmin,ymin])
    pointA =  np.round((np.array(pointA) - origin)/d)
    pointB =  np.round((np.array(pointB) - origin)/d)
    pointC =  np.round((np.array(pointC) - origin)/d)
    pointD =  np.round((np.array(pointD) - origin)/d)

    # Find warp
    src_pts = np.array([pointA, pointB, pointC, pointD],np.float32)
    dst_pts = np.array([[0,0],[output_img_size[0],0],[output_img_size[0],output_img_size[1]],[0,output_img_size[1]]],np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    dst = cv2.warpPerspective(np.array(big_enough_patch).astype(np.float32), M, output_img_size)
    return PIL.Image.fromarray(dst.astype(np.uint8))


def find_lowest_matching_level(fixed_slide, moving_slide):
    """
    Returns lowest level of the fixed slide that matches with the moving slide's lvl0. 
    """
    fx = float(fixed_slide.properties["openslide.mpp-x"])
    fy = float(fixed_slide.properties["openslide.mpp-y"])

    mx = float(moving_slide.properties["openslide.mpp-x"])
    my = float(moving_slide.properties["openslide.mpp-y"])

    f = np.mean([fx,fy])
    m = np.mean([mx,my])

    desired_downsample = m/f
    if desired_downsample < 1:
        return 0
    else:
        return fixed_slide.get_best_level_for_downsample(desired_downsample)


def get_matrix_from_tp(tp):
    # Get itk affine matrix from the retrieved transform parameters
    m = [   [tp[3],tp[2],tp[5]],
            [tp[1],tp[0],tp[4]],
            [0,0,1]]
    m = np.array(m)
    return m