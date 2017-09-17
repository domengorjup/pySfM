__author__ = "Domen Gorjup"

import os
import glob
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm, tqdm_notebook
import tools

# IZBIRA SLIK IN NAČINA REKONSTRUKCIJE ###############################################################
path = 'images/test_hisa'
path_calib = 'kalibracija/redmi'
# BUNDLE ADJUSTMWNT NASTAVITVE: ######################################################################
#   "P" ... Pairwise
#   "S" ... Sequential
#   "F" ... Final
#   "X" ... point coordinates (X) only
#   or any combination of the above 
BA = 'SX'
# NASTAVITVE OBDELAVE ################################################################################
SCALE = 0.25        # image resampling scale
MATCH_RATIO = 0.65  # more -> LESS matches rejected
FTOL = 1e-5         # cost function change tolerance for BA
NFEV = 5000         # maximum number of least_squares iterations
MODE = 'ORB'        # feature extraction algorithm
WEIGHT_CUTOFF = 150  # px
d_max = 2           # (RANSAC) max distance from epipolar line above which point is an outlier
alpha = 0.995       # (RANSAC) desired probability, that the estimated F matrix is correct
max_outlier_sd = 2  # for outlier rejection after bundle adjustment
SAVENAME = ''       # appendix to results save name
DEBUG = False       # if True, show analysis progress and render intermediate scenes
######################################################################################################

this_dir = os.path.dirname(os.path.realpath(__file__))
images = glob.glob(os.path.join(this_dir, path, '*.JPG'))
calibration_images = glob.glob(os.path.join(this_dir, path_calib, '*.JPG'))

if 'X' in BA:
    X_only = True
else:
    X_only = False


beginning_time = time.perf_counter()

# Camera calibration
K, distCoeffs = tools.calibrate_camera(calibration_images, (6, 4), (40, 40), scale=0.2)
N = len(images)

after_calibration = time.perf_counter()

# Main loop
for i in tqdm(range(N - 1)):
    
    if i == 0:
        last_Rt = None
        im1 = tools.read_bw(images[i])
        im2 = tools.read_bw(images[i+1])
       
        Rt1, Rt2, p1, p2 = tools.get_camera_matrices(im1, im2, K, last_Rt, optimize=True, d_max=d_max, alpha=alpha,
                                           scale=SCALE, match_ratio=MATCH_RATIO, mode=MODE)
    else:
        im1 = im2
        im2 = tools.read_bw(images[i+1])
        Rt1, Rt2, p1, p2 = tools.get_camera_matrices_PnP(im1, im2, K, last_Rt, cloud, i, distCoeffs, 
                                            d_max=d_max, alpha=alpha, scale=SCALE, match_ratio=MATCH_RATIO, mode=MODE)

    last_Rt = Rt2
    P1 = K.dot(Rt1)
    P2 = K.dot(Rt2)
    
    points3D = []
    for x1, x2 in zip(p1, p2):
        X = tools.triangulate_lm(tools.to_homogenous(x1), tools.to_homogenous(x2), P1, P2)
        points3D.append(tools.Point3D(X, x1, x2, i, i+1))
    
    if i == 0:
        cloud = tools.Scene([Rt1, Rt2], points3D, K, im1.shape, debug=DEBUG)
        if 'P' in BA:
            cloud.bundle_adjustment(ftol=FTOL, max_nfev=NFEV, max_sd_dist=max_outlier_sd, 
                                    weight_cutoff=WEIGHT_CUTOFF, X_only=X_only)
    else:
        pair = tools.Scene([Rt1, Rt2], points3D, K, im1.shape, frame_offset=i, debug=DEBUG)
        if 'P' in BA:
            pair.bundle_adjustment(ftol=FTOL, max_nfev=NFEV, max_sd_dist=max_outlier_sd, 
                                weight_cutoff=WEIGHT_CUTOFF, X_only=X_only)
        cloud.add_next_pair(pair)

        # Sequential bundle adjustment
        if 'S' in BA:
            cloud.bundle_adjustment(ftol=FTOL, max_nfev=NFEV, max_sd_dist=max_outlier_sd, X_only=X_only)

main_loop_time = time.perf_counter()

# Final BA:
if 'F' in BA:
    cloud.bundle_adjustment(ftol=1e-3, max_nfev=NFEV, X_only=True, debug=DEBUG)

end_time = time.perf_counter()

# Timing:
total = end_time - beginning_time 
calibration = after_calibration - beginning_time
loop = main_loop_time - after_calibration
ba = end_time - main_loop_time
if DEBUG:
    print('total time:\t{:.0f} min {:.1f} s'.format(total // 60, total % 60))
    print('calib. time:\t{:.0f} min {:.1f} s'.format(loop // 60, calibration % 60))
    print('loop time:\t{:.0f} min {:.1f} s'.format(loop // 60, loop % 60))
    print('ba time:\t{:.0f} min {:.1f} s'.format(ba // 60, ba % 60))

# Save results
name = path.replace('/', '_').replace('\\', 'n')
name += SAVENAME
cloud.pickle_save(name=name, path=os.path.join(this_dir, 'results'), mode='scene')

# Visualization (Matplotlib)
X, Y, Z = cloud.render(cameras=True)
plt.show()