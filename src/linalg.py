import numpy as np

from src.region import Region, RegionNonRect


def get_equivalent_region(r_in, input_slide, output_slide, full_transform):

    downsample_for_this_level = input_slide.level_downsamples[r_in.lvl]
    absolute_patch_size_x = r_in.size_x*downsample_for_this_level
    absolute_patch_size_y = r_in.size_y*downsample_for_this_level
    
    # define the input region by its 4 points using lvl0 absolute coordinates
    pointA = (r_in.x,r_in.y,1)
    pointB = (r_in.x+absolute_patch_size_x,r_in.y,1)
    pointC = (r_in.x,r_in.y+absolute_patch_size_y,1)
    pointD = (r_in.x+absolute_patch_size_x,r_in.y+absolute_patch_size_y,1)

    # transform each of these 4 points to the moving slide lvl0 absolute referential
    transformed_pointA = full_transform @ pointA
    transformed_pointB = full_transform @ pointB
    transformed_pointC = full_transform @ pointC
    transformed_pointD = full_transform @ pointD

    # now average out the corresponding coordinates to create a rectangular region in case the result of the transform
    # is not exactly rectangular (this is most likely always the case)
    all_x_coords = [p[0] for p in (transformed_pointA, transformed_pointB, transformed_pointC, transformed_pointD)]
    all_y_coords = [p[1] for p in (transformed_pointA, transformed_pointB, transformed_pointC, transformed_pointD)]

    #The region we want is defined by the rectangle (avg_xmin, avg_xmax, avg_ymin, avg_ymax) in lvl0 absolute coords
    avg_xmin = round(np.mean(sorted(all_x_coords)[:2]))
    avg_xmax = round(np.mean(sorted(all_x_coords)[2:]))
    avg_ymin = round(np.mean(sorted(all_y_coords)[:2]))
    avg_ymax = round(np.mean(sorted(all_y_coords)[2:]))

    # now find the level and associated downsample of the moving slide that best fits our rectangle:
    resulting_absolute_patch_size_x = avg_xmax - avg_xmin
    resulting_absolute_patch_size_y = avg_ymax - avg_ymin
    
    desired_downsample_x = resulting_absolute_patch_size_x/r_in.size_x
    desired_downsample_y = resulting_absolute_patch_size_y/r_in.size_y
    
    best_lvl = output_slide.get_best_level_for_downsample(np.mean((desired_downsample_x, desired_downsample_y)))
    best_downsample = output_slide.level_downsamples[best_lvl]
    
    # Finally, deduce the patch size required to cover the requested area
    required_patch_size_x = round(resulting_absolute_patch_size_x/best_downsample)
    required_patch_size_y = round(resulting_absolute_patch_size_y/best_downsample)

    r_out = Region(avg_xmin, avg_ymin, best_lvl, required_patch_size_x, required_patch_size_y)
    return r_out


def get_equivalent_region_non_rect(r_in: Region, input_slide, output_slide, full_transform, rotation):

    downsample_for_this_level = input_slide.level_downsamples[r_in.lvl]
    absolute_patch_size_x = r_in.size_x*downsample_for_this_level
    absolute_patch_size_y = r_in.size_y*downsample_for_this_level
    
    # define the input region by its 4 points using lvl0 absolute coordinates
    pointA = (r_in.x,r_in.y,1)
    pointB = (r_in.x+absolute_patch_size_x,r_in.y,1)
    pointC = (r_in.x+absolute_patch_size_x,r_in.y+absolute_patch_size_y,1)
    pointD = (r_in.x,r_in.y+absolute_patch_size_y,1)

    # transform each of these 4 points to the moving slide lvl0 absolute referential
    transformed_pointA = (full_transform @ pointA)[:2]
    transformed_pointB = (full_transform @ pointB)[:2]
    transformed_pointC = (full_transform @ pointC)[:2]
    transformed_pointD = (full_transform @ pointD)[:2]

    transformed_pointA = np.round(transformed_pointA)
    transformed_pointB = np.round(transformed_pointB)
    transformed_pointC = np.round(transformed_pointC)
    transformed_pointD = np.round(transformed_pointD)

    # Find the smallest possible rectangle that contains the whole non-rectangular region
    all_x_coords = [p[0] for p in (transformed_pointA, transformed_pointB, transformed_pointC, transformed_pointD)]
    all_y_coords = [p[1] for p in (transformed_pointA, transformed_pointB, transformed_pointC, transformed_pointD)]

    # The region we want is defined by the bigger rectangle that contains it ((xmin, xmax), (ymin, ymax)) in lvl0 absolute coords
    xmin = np.amin(all_x_coords)
    xmax = np.amax(all_x_coords)
    ymin = np.amin(all_y_coords)
    ymax = np.amax(all_y_coords)

    # Find the level and associated downsample of the moving slide that best fits our region:
    resulting_absolute_patch_size_x = xmax - xmin
    resulting_absolute_patch_size_y = ymax - ymin
    
    desired_downsample_x = resulting_absolute_patch_size_x/r_in.size_x
    desired_downsample_y = resulting_absolute_patch_size_y/r_in.size_y
    
    best_lvl = output_slide.get_best_level_for_downsample(np.mean((desired_downsample_x, desired_downsample_y)))
    best_downsample = output_slide.level_downsamples[best_lvl]
    
    # Finally, deduce the patch size required to cover the requested area
    required_patch_size_x = round(resulting_absolute_patch_size_x/best_downsample)
    required_patch_size_y = round(resulting_absolute_patch_size_y/best_downsample)


    if rotation=='clockwise':
        # LEICA -> 3DHISTECH
        r_out = RegionNonRect(transformed_pointB, transformed_pointC, transformed_pointD, transformed_pointA, best_lvl, required_patch_size_x, required_patch_size_y)

    elif rotation=='anticlockwise':
        # 3DHISTECH -> LEICA
        r_out = RegionNonRect(transformed_pointD, transformed_pointA, transformed_pointB, transformed_pointC, best_lvl, required_patch_size_x, required_patch_size_y)

    elif rotation=='180':
        # 3DHISTECH -> ROCHE
        r_out = RegionNonRect(transformed_pointC, transformed_pointD, transformed_pointA, transformed_pointB, best_lvl, required_patch_size_x, required_patch_size_y)

    elif rotation=='identity':
        r_out = RegionNonRect(transformed_pointA, transformed_pointB, transformed_pointC, transformed_pointD, best_lvl, required_patch_size_x, required_patch_size_y)
    
    else:
        raise ValueError(f"Unknown rotation str: '{rotation}'")

    return r_out


def auto_detect_rotation(rough_transform):
    r = rough_transform[:2,:2]/np.amax(np.abs(rough_transform[:2,:2]))

    # create template rot matrices
    identity=np.eye(2)
    cw=np.array([[0,1],[-1,0]],dtype=float)
    acw=np.array([[0,-1],[1,0]],dtype=float)
    trsp=-np.eye(2)

    norms = np.array([np.linalg.norm(r-identity),np.linalg.norm(r-cw),np.linalg.norm(r-acw),np.linalg.norm(r-trsp)])
    a = np.argmin(norms)

    if a==0:
        rotation='identity'
    elif a==1:
        rotation='clockwise'
    elif a==2:
        rotation='anticlockwise'
    elif a==3:
        rotation='180'

    return rotation
