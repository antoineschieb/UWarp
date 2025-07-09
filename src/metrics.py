import itk
import skimage
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from PIL.Image import BILINEAR, TRANSPOSE, FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM

from src.data import display_overlapping_patches, get_patch
from src.region import Region

def nmi(r1: Region, r2: Region, fixed_slide, moving_slide, rotation, file=None):
    p1 = get_patch(fixed_slide, r1)
    p2 = get_patch(moving_slide, r2)
    
    overlap = display_overlapping_patches(p1, p2, rotation)
    if file:
        overlap.save(file)
        print(f'saving {file}')
    overlap = np.array(overlap).astype(np.float32)
    
    return(normalized_mutual_info_score(overlap[:,:,0].ravel(), overlap[:,:,2].ravel()))


def TRE_score(p1, p2, parameter_object, rotate):

    if rotate == 'clockwise':
        p2 = p2.transpose(method=TRANSPOSE)
        p2 = p2.transpose(method=FLIP_LEFT_RIGHT)
    elif rotate == 'anticlockwise':
        p2 = p2.transpose(method=FLIP_LEFT_RIGHT)
        p2 = p2.transpose(method=TRANSPOSE)
    elif rotate == '180':
        p2 = p2.transpose(method=FLIP_LEFT_RIGHT)
        p2 = p2.transpose(method=FLIP_TOP_BOTTOM)

    p2 = p2.resize(p1.size, BILINEAR)

    input_patch_gray = p1.convert('L')
    output_patch_gray = p2.convert('L')

    itk_fixed = itk.GetImageFromArray(np.ascontiguousarray(input_patch_gray))
    itk_moving = itk.GetImageFromArray(np.ascontiguousarray(output_patch_gray))
    registered_image, params = itk.elastix_registration_method(itk_fixed, itk_moving, parameter_object=parameter_object, log_to_console=False)
    
    parameter_map = params.GetParameterMap(0)
    tp = np.array(parameter_map['TransformParameters'], dtype=float)
    return np.linalg.norm([tp[4],tp[5]])


def TRE_score_phasediff(p1, p2, rotate):
    
    if np.sum(p1) == 0 or np.sum(p2) == 0:
        return None

    if rotate == 'clockwise':
        p2 = p2.transpose(method=TRANSPOSE)
        p2 = p2.transpose(method=FLIP_LEFT_RIGHT)
    elif rotate == 'anticlockwise':
        p2 = p2.transpose(method=FLIP_LEFT_RIGHT)
        p2 = p2.transpose(method=TRANSPOSE)
    elif rotate == '180':
        p2 = p2.transpose(method=FLIP_LEFT_RIGHT)
        p2 = p2.transpose(method=FLIP_TOP_BOTTOM)

    p2 = p2.resize(p1.size, BILINEAR)
    p1 = np.array(p1)[:,:,:3]
    p2 = np.array(p2)[:,:,:3]

    shift, err, phasediff = skimage.registration.phase_cross_correlation(p1, p2)
    return np.linalg.norm([shift[0],shift[1]])
