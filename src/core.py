import time
import PIL
import PIL.Image
import itk
import numpy as np
from tqdm import tqdm
import openslide as op

from src.data import correct_itk_transform, define_engulfing_region, display_overlapping_patches, find_lowest_matching_level, get_matrix_from_tp, get_patch, get_uniformly_distributed_valid_points
from src.linalg import get_equivalent_region, get_equivalent_region_non_rect
from src.metrics import nmi
from src.region import Region


def get_adjustment_for_region(r: Region, fixed_slide, moving_slide, full_transform, rotation, verbose=True, landmark_nbr=0, itr=0, reg_folder=None):
    r_out = get_equivalent_region_non_rect(r, fixed_slide, moving_slide, full_transform, rotation)

    input_patch= get_patch(fixed_slide, r)
    output_patch = get_patch(moving_slide, r_out)

    overlap = display_overlapping_patches(input_patch, output_patch, rotation)
    if reg_folder is not None and verbose:
        overlap.save(reg_folder / "all_overlaps" / f"{landmark_nbr}-{itr}_:_{r}_{r_out}.jpg")

    output_downsample = moving_slide.level_downsamples[r_out.lvl]

    if rotation == 'clockwise':
        # LEICA -> 3DHISTECH
        input_patch = input_patch.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
        input_patch = input_patch.transpose(method=PIL.Image.TRANSPOSE)
    elif rotation == 'anticlockwise':
        # 3DHISTECH -> LEICA
        input_patch = input_patch.transpose(method=PIL.Image.TRANSPOSE)
        input_patch = input_patch.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
    elif rotation == '180':
        input_patch = input_patch.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
        input_patch = input_patch.transpose(method=PIL.Image.FLIP_TOP_BOTTOM)
    # if rotation==identity, no need to change anything

    input_patch = input_patch.resize((r_out.size_x, r_out.size_y), PIL.Image.BILINEAR)
    input_patch_gray = input_patch.convert('L')
    output_patch_gray = output_patch.convert('L')

    itk_fixed = itk.GetImageFromArray(np.ascontiguousarray(input_patch_gray))
    itk_moving = itk.GetImageFromArray(np.ascontiguousarray(output_patch_gray))

    parameter_object = itk.ParameterObject.New()
    default_affine_parameter_map = parameter_object.GetDefaultParameterMap('affine')   # Affine
    default_affine_parameter_map['FinalBSplineInterpolationOrder'] = ['0']  # Always on
    parameter_object.AddParameterMap(default_affine_parameter_map)
    try:
        registered_image, params = itk.elastix_registration_method(itk_fixed, itk_moving, parameter_object=parameter_object, log_to_console=False)
    except RuntimeError as e:
        print(f"Error while running ITK registration:{e}\nWill return identity for this region's adjustment")
        return np.identity(3)
    
    parameter_map = params.GetParameterMap(0)
    tp = np.array(parameter_map['TransformParameters'], dtype=float)
    
    cx,cy = np.float32(parameter_map['CenterOfRotationPoint'])
    itk_affine_matrix = get_matrix_from_tp(tp)
    trnsl = [[1,0,cx],
            [0,1,cy],
            [0,0,1]]
    trnsl = np.array(trnsl)
    itk_transform = trnsl @ itk_affine_matrix @ np.linalg.inv(trnsl)
    b1_to_b0 = [[0,-output_downsample,r_out.pointA[0] ], 
                [output_downsample, 0,r_out.pointA[1] ],   
                [0,0,1]]
    b1_to_b0 = np.array(b1_to_b0)
    
    local_transform = correct_itk_transform(itk_transform.copy())
    adjustment_transform = b1_to_b0 @ local_transform @ np.linalg.inv(b1_to_b0)
    return adjustment_transform


def iterative_patch_reg(input_r: Region, fixed_slide, moving_slide, full_transform, rotation, verbose=False, landmark_nbr=0, reg_folder=None):
    highest_slide_lvl=len(fixed_slide.level_downsamples)

    # 1) Iteratively define all the regions of the input slide that we are going to go through
    regions_to_register = [input_r]
    # define_engulfing_region(input_r, fixed_slide.level_downsamples)

    while regions_to_register[-1].lvl != highest_slide_lvl-1:
        bigger_region = define_engulfing_region(regions_to_register[-1], fixed_slide.level_downsamples)
        regions_to_register.append(bigger_region)

    # 2) Now take the last region added (the biggest of all) and start the registration process, returning one adjustment transform.
    combined_adjustments = np.identity(3)
    for itr,r in enumerate(reversed(regions_to_register)):        
        adj = get_adjustment_for_region(
            r,
            fixed_slide,
            moving_slide,
            combined_adjustments @ full_transform,
            rotation,
            verbose=verbose,
            landmark_nbr=landmark_nbr,
            itr=itr,
            reg_folder=reg_folder,
        )

        combined_adjustments = adj @ combined_adjustments

    # 3) Finally compute the region with the full transform
    adjusted_equivalent_region = get_equivalent_region(input_r, fixed_slide, moving_slide, combined_adjustments @ full_transform)
    return adjusted_equivalent_region, combined_adjustments @ full_transform, 


def find_lstsq_solution(fixed_slide, moving_slide, full_transform, rotation, N_landmarks=100, quality_thresh=0.15, verbose=True, reg_folder=None):

    coords = get_uniformly_distributed_valid_points(fixed_slide, N_landmarks=N_landmarks)
    best_lvl = find_lowest_matching_level(fixed_slide, moving_slide)
    if verbose:
        print(f"best_lvl for the fixed slide: {best_lvl}")
    
    regions = []
    for x,y in coords:
        r = Region(x,y,best_lvl,512,512)
        regions.append(r)
    
    regions_in = []
    regions_out = []
    exact_transforms = []
    for i,r in tqdm(list(enumerate(regions))):
        try:
            r_out, exact_transform = iterative_patch_reg(r,fixed_slide, moving_slide, full_transform, rotation, verbose=verbose, landmark_nbr=i, reg_folder=reg_folder)
        except AssertionError as e:
            print(e)
            continue
        p_1 = get_patch(fixed_slide, r)
        p_2 = get_patch(moving_slide, r_out)
        
        overlap = display_overlapping_patches(p_1, p_2, rotation)
        quality = nmi(r, r_out, fixed_slide, moving_slide, rotation)
        if reg_folder is not None and verbose:
            overlap.save(reg_folder / "final_overlaps" / f"landmark{i}_nmi_{quality}.jpg")
        
        if quality<quality_thresh:
            continue    
        regions_in.append(r)
        regions_out.append(r_out)
        exact_transforms.append(exact_transform)
    
    print(f"{len(regions_in)} landmarks of sufficient quality were kept.")

    # Define the N pairs of corresponding points 
    input_pts = np.float32([[r.center(downsample=fixed_slide.level_downsamples[r.lvl])[0],r.center(downsample=fixed_slide.level_downsamples[r.lvl])[1],1] for r in regions_in])
    output_pts = np.float32([[r_out.center(downsample=moving_slide.level_downsamples[r_out.lvl])[0], r_out.center(downsample=moving_slide.level_downsamples[r_out.lvl])[1],1] for r_out in regions_out])  
    if verbose:
        print("input_pts")
        print(input_pts)
        print("output_pts")
        print(output_pts)
    lstsq_transform, res_sum, rank, s = np.linalg.lstsq(input_pts, output_pts, rcond=None)
    residues = (input_pts @ lstsq_transform) - output_pts
    if verbose:
        print("residues")
        print(residues)

    lstsq_transform = lstsq_transform.T
    initial_landmarks_nbr = len(regions)
    return lstsq_transform, input_pts, residues, initial_landmarks_nbr
