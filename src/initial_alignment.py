# Adapted from https://github.com/DeepMicroscopy/CrossScannerRegistration/blob/main/iterative_registration.py

from PIL import ImageOps
import cv2
import numpy as np


def inital_registration(slide_source, slide_target, downsampling_factor):
    '''
    :param slide_source: whole file scanned by scanner1
    :param slide_target: whole file scanned by the other scanner and that should registered to the reference file
    :param level_idx: the level on with the inital registration should be done, works best on level 7&8
    :return: transformation parameters: isotropic scale factor, rotation angle (in degrees), and translation vector.
    '''

    # Load images at the lowest level
    img_r = np.array(ImageOps.grayscale(slide_source.get_thumbnail([slide_source.dimensions[0] // downsampling_factor,
                                                                    slide_source.dimensions[1] // downsampling_factor])))
    img_t = np.array(ImageOps.grayscale(slide_target.get_thumbnail([slide_target.dimensions[0] // downsampling_factor,
                                                                    slide_target.dimensions[1] // downsampling_factor])))
    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    kpsA_ori, descsA = detector.detectAndCompute(img_r, None)
    kpsB_ori, descsB = detector.detectAndCompute(img_t, None)
    matches = matcher.knnMatch(descsA, descsB, k=2)
    mkp1, mkp2, good = [], [], []
    for match in matches:
        if len(match) < 2:
            break

        m, n = match
        if m.distance < n.distance * 0.6:
            good.append([m])
            mkp1.append(np.array(kpsA_ori[m.queryIdx].pt))
            mkp2.append(np.array(kpsB_ori[m.trainIdx].pt))
    ptsA, ptsB = [], []

    for ptA, ptB in zip(mkp1, mkp2):
        ptA = ptA * (slide_source.dimensions[0] / img_r.shape[1], slide_source.dimensions[1] / img_r.shape[0])
        ptB = ptB * (slide_target.dimensions[0]/img_t.shape[1], slide_target.dimensions[1]/img_t.shape[0])
        ptsA.append(ptA)
        ptsB.append(ptB)

    (E, status) = cv2.estimateAffine2D(np.float32(ptsB), np.float32(ptsA), method=cv2.RANSAC, ransacReprojThreshold=3,
                                       confidence=0.90)
    E = np.concatenate((E,[[0,0,1]]))
    return E
