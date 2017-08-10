__author__ = "Domen Gorjup"

import os
import glob
import cv2
import warnings
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import sys


class Scene:
    """ A class containing the identified cameras and 3D points. """
    
    def __init__(self, cameras, points3D, K, shape=None, frame_offset=0, debug=False):
        self.cameras = cameras # [R|t] matrices
        self.K = K # P = K[R|t]
        self.N_frames = len(self.cameras)
        # to easily search dict keys for matching features
        self.points3D = dict([ (p.features[-1], p) for p in points3D ])
        # to offset for camera matrix index when not all frames in scene
        self.frame_offset = frame_offset
        self.shape = shape
        self.debug = debug
            
    def add_next_pair(self, new_pair):
        """ Merge a new view pair into this scene. """
        self.cameras.append(new_pair.cameras[-1])
        self.N_frames += 1
        
        # check existing 3D points for matches
        for key, p in new_pair.points3D.items():
            # search for features from first image in pair
            # (only searches for first feature of new point - for ordered image sequences!)
            query = p.features[0] 
            if query in self.points3D.keys():
                self.points3D[key] = self.points3D.pop(query)
                self.points3D[key].update(p)
            # if there are no matches, add a new point
            else:
                self.points3D[key] = p
    
    def get_3d_points(self):
        """ Return X, Y, Z coordinated of triangulated points. """
        XYZ = np.array([ from_homogenous(point3D.X) for key, point3D in self.points3D.items() ])
        return XYZ.T

    def render(self, N_points=0, cameras=False, openGL=False, show=True):
        """ Draw the scene (point cloud and camera positions). """
        XYZ = np.array([ from_homogenous(point3D.X) for key, point3D in self.points3D.items() ])
        X, Y, Z = XYZ.T
        Xc, Yc, Zc = np.array([pos_from_P(P)[:-1, -1] for P in self.cameras]).T 
        
        if show:
            if openGL:
                if cameras:
                    pg_render(XYZ, np.column_stack((Xc, Yc, Zc)))
                else:
                    pg_render(XYZ)
                
            else:
                if N_points != 0:
                    increment = len(X) // N_points
                else:
                    increment = 1
                    N_points = len(X)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X[::increment], Y[::increment], Z[::increment], 
                        depthshade=False, s=0.1, c='k', antialiased=False)
                ax.set_title('3D point cloud')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')

                # plot camera positions
                if cameras:     
                    colors = ['C2'] + ['C0'] * (len(Xc) - 1)
                    ax.scatter(Xc, Yc, Zc, c=colors, depthshade=False, 
                            antialiased=False, label='camera positions')
                    ax.legend()
                
                    Xa = np.hstack([X, Xc])
                    Ya = np.hstack([Y, Yc])
                    Za = np.hstack([Z, Zc])
                else:
                    Xa, Ya, Za = X, Y, Z
                
                # equal axis aspect ratio bounding box
                max_range = np.array([Xa.max()-Xa.min(), Ya.max()-Ya.min(), Za.max()-Za.min()]).max()
                Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(Xa.max()+Xa.min())
                Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Ya.max()+Ya.min())
                Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Za.max()+Za.min())
                for xb, yb, zb in zip(Xb, Yb, Zb):
                    ax.plot([xb], [yb], [zb], 'w')
                
                plt.show()                       
                print('{:d} identified points ({:d} shown).'.format(len(X), N_points))
       
        return X, Y, Z
                
    def pack(self, X_only=False):
        """ Pack the scene for optimization. """
        cams = []
        for Rt in self.cameras:
            rvec = cv2.Rodrigues(Rt[:, :3])[0].ravel()
            tvec = Rt[:, -1]
            cams.append(np.hstack((rvec, tvec)))
        cams = np.hstack(cams)
        
        # coordinates, visibility and 2d feature coordinates of all 3d points
        X = []
        key_order = []

        visibility = np.zeros((len(self.cameras), len(self.points3D)), dtype=bool)
        points2D = np.zeros((len(self.cameras), len(self.points3D), 3))
        for i, (key, point) in enumerate(self.points3D.items()):
            # offset frame indices for frames not yet in scene
            offset_frames = np.array(point.frames) - self.frame_offset
            X.append(from_homogenous(point.X))
            key_order.append(key)
            # <point.frame> rows of visibility matrix at column, corresponding to <i> point: 
            visibility[offset_frames, i] = 1
            # <point.frame> rows of 2d-feature matrix at column, corresponding to <i> point:
            points2D[offset_frames, i, :] = np.vstack([to_homogenous(f[1]) for f in point.features])                                                                    
        
        points = np.hstack(X)
        x = np.hstack((cams, points))
        self.key_order = key_order

        if X_only:
            return points, self.cameras, visibility, points2D, self.N_frames
        
        return x, visibility, points2D, self.N_frames
    
    def unpack(self, x, X_only=False, remove_outliers=True, max_sd_dist=2):
        """ Unpack bundle adjustment results to update the scene. """

        if not X_only:
            # construct [R|t] matrices
            n_camera_params = 6*self.N_frames
            camera_parameters = x[:n_camera_params].reshape(-1,6)
            cameras = []
            for camera_i in camera_parameters:
                rvec = camera_i[:3]
                tvec = camera_i[3:]
                Rt = np.column_stack((cv2.Rodrigues(rvec)[0], tvec))
                cameras.append(Rt)
            self.cameras = cameras

            new_X = x[n_camera_params:].reshape(-1, 3)

        else:
            new_X = x.reshape(-1, 3)
            
        # update point3D objects        
        if remove_outliers:
            distances = np.linalg.norm(new_X, axis=1)
            dist_sd = np.std(distances)
            dist_mean = np.mean(distances)
            
            #print('-'*30 + '> ', max_sd_dist * dist_sd)

        for X_, key in zip(new_X, self.key_order):
            if remove_outliers:
                d = np.abs(np.linalg.norm(X_) - dist_mean)
                if d > max_sd_dist * dist_sd:
                    self.points3D.pop(key)
                else:
                    self.points3D[key].X = to_homogenous(X_)
                    
            else:
                self.points3D[key].X = to_homogenous(X_)
                        
    def bundle_adjustment(self, ftol=1e-6, max_nfev=5000, remove_outliers=True, max_sd_dist=2, X_only=False):
        """ Perform bundle adjustment to optimize cameras and points in the scene. """
        packed = self.pack(X_only=X_only)
        x0 = packed[0]
        args = packed[1:] + (self.K, 150)
        if X_only:
            cost_f = reprojection_error_X_only
        else:
            cost_f = reprojection_error

        if self.debug:  
            print('--- BUNDLE ADJUSTMENT ---')
            print('parameters:\t{:d} '.format(len(x0)))
            print('cost before BA:\t{:.2f}'.format(np.sum(cost_f(x0, *args, debug=True, title='before', shape=self.shape) ** 2)))
        
        # LS optimization
        try:
            optimization_results = least_squares(cost_f, 
                                                 x0=x0, 
                                                 method='lm', 
                                                 ftol=ftol,
                                                 xtol=1e-9,
                                                 gtol=1e-12,
                                                 max_nfev=max_nfev,
                                                 verbose=0,
                                                 args=args)
            new_x = optimization_results.x
            cost = optimization_results.cost

            # debugging
            if self.debug:
                print('optimization summary:\n\tcost {:.5f}\n\tnfev {:.5f}\n\tstatus {:d}\n\tmessage {:s}'.format(optimization_results.cost, 
                                                                                                                optimization_results.nfev, 
                                                                                                                optimization_results.status, 
                                                                                                                optimization_results.message))
                print('cost after BA:\t{:.2f}'.format(np.sum(cost_f(new_x, *args, debug=True, title='after', shape=self.shape) ** 2)), '\n')
                plt.show()
        
        except Exception as e:
            print('Warning! ({:s}) Omitting optimization step.'.format(str(e)))
            new_x = x0
            cost = -1
        
        # update the scene
        self.unpack(new_x, X_only=X_only, remove_outliers=remove_outliers, max_sd_dist=max_sd_dist)
        return cost     

    def build_frames(self):
        """ Build a dictionary of frames and corresponding points in the scene. """
        x, cams, visibility, points2D, N_frames = self.pack(X_only=True)
        X = x.reshape(-1, 3)

        X_ = np.vstack((X.T, np.ones(X.shape[0])))
        XYZc = np.array([pos_from_P(P)[:-1, -1] for P in self.cameras])

        frames = {}
        for i in range(N_frames):
            P = self.K.dot(self.cameras[i])
            reprojected_points = np.dot(P, X_[:, visibility[i]])[:2].T
            frames[i] = {'3D': X[visibility[i]],
                         '2D': points2D[i][visibility[i]][:, :2],
                         're': reprojected_points,
                         'ca': XYZc[i]}
        
        return frames
    
    def pickle_save(self, name='point_cloud', path=os.path.dirname(os.path.realpath(__file__)), mode='points'):
        """ Save scene to a pickle file. """
        timestamp = datetime.datetime.now().strftime(("%d-%m-%H-%M"))
        
        if mode == 'points':
            filename = os.path.join(path, '{:s}_{:s}.pkl'.format(name, timestamp))
            X = []
            for i, (key, point) in enumerate(self.points3D.items()):
                # offset frame indices for frames not yet in scene
                X.append(from_homogenous(point.X))
            XYZ = np.vstack(X).T
            XYZc = np.array([pos_from_P(P)[:-1, -1] for P in self.cameras]).T 
            pickle.dump({'points': XYZ, 'cameras': XYZc}, open(filename, 'wb'))

        elif mode == 'scene':
            filename = os.path.join(path, 'scene_{:s}_{:s}.pkl'.format(name, timestamp))
            pickle.dump(self, open(filename, 'wb'))
        
        print('Saved {:s} to: {:s}'.format(mode, filename))


class Point3D:
    """
    Triangulated 3D point with feature correspondence and source image information.
    """
    def __init__(self, X, p1, p2, i_im1, i_im2):
        self.X = np.array(X, dtype=np.float32)
        self.features = [(i_im1, tuple(p1)), (i_im2, tuple(p2))]
        self.frames = [i_im1, i_im2] # frames, in which the point is visible
        
    def update(self, new_point):
        """
        Append the new 3D coordinates and update the last feature reference.
        To be used when this point has a match in the next frame.
        """
        # update 3D coordinates to mean of individual points ?
        self.X = np.mean(np.vstack((self.X, new_point.X)), axis=0)
        self.features.append(new_point.features[-1])
        self.frames.append(new_point.frames[-1])


def read_bw(file, flip=False):
    """ Read an image in grayscale, as 2D 8-bit numpy.ndarray. """
    im = cv2.imread(file, 0)
    if flip:
        im = np.flipud(im.T)
    return im


def to_homogenous(x):
    """ Convert a point into homogenous coordinates. """
    return np.append(x, 1)

def from_homogenous(xh):
    """ Convert a point from homogenous coordiantes. """
    return xh[:-1] / xh[-1]

def triangulate_lsq(x1, x2, P1, P2):
    """
    Triangulate the 3D point by solving 
        x = PX
    in a least-squares sense.
    """
    A = np.vstack((P1, P2))
    b = np.hstack([x1, x2]).T
    X = np.linalg.lstsq(A, b)[0]
    return X / X[-1]

def cross_matrix(a):
    """ The cross-product matrix of a length 3 vector a (H&Z, p.581). """
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]], dtype=float)

def null_vector(A):
    """ Find the (right) null vector of matrix A. """
    u, s, v = np.linalg.svd(A)
    return v[-1] / v[-1, -1]

def reprojection_error(x, visibility, points2D, N_frames, K, weight_cutoff=15, debug=False, shape=None, title=''):
    """
    Reprojection error, minimized in bundle adustment
    S = sum(d(P_i X_j, x_j)) (H&Z, p. 434)
    """    
    # unpack camera matrices  
    n_camera_params = 6*N_frames
    camera_parameters = x[:n_camera_params].reshape(-1,6)
    cameras = []

    for camera_i in camera_parameters:
        rvec = camera_i[:3]
        tvec = camera_i[3:]
        Rt = np.column_stack((cv2.Rodrigues(rvec)[0], tvec))
        cameras.append(K.dot(Rt))

    # unpack 3D points coordiantes, convert to homogenous
    X_ = x[n_camera_params:].reshape(-1, 3).T
    X = np.vstack((X_, np.ones(X_.shape[1])))
    
    if debug:
        fig, ax = plt.subplots(1, len(cameras), figsize=(4*len(cameras), 3))

    # reprojection error
    residuals = []
    for i, P in enumerate(cameras):
        reprojected_points = np.dot(P, X[:, visibility[i]])
        image_points = points2D[i, visibility[i]].T

        #debugging
        if debug:
            #fig = plt.figure()
            plt.axis('equal')
            plt.suptitle(title)
            ax[i].set_title('camera {:d}'.format(i))
            ax[i].text(0.01, 0.92, '{:d} points'.format(image_points.shape[1]), 
                        transform=ax[i].transAxes, fontsize=10)
            if shape is not None:
                ax[i].set_ylim(0, shape[0])
                ax[i].set_xlim(0, shape[1])
            ax[i].scatter(image_points[0], image_points[1], c='C0', alpha=0.75, label='feature points')
            ax[i].scatter(reprojected_points[0], reprojected_points[1], c='C3', alpha=0.5, label='reprojected')
            ax[i].legend(loc=3)
            #plt.show()
        
        diff = reprojected_points[:2] - image_points[:2]
        # dist = np.sqrt(np.sum(diff**2, axis=0))
        
        # # Tukey weights (https://github.com/dfridovi/SimpleSFM)
        # weight_cutoff = np.sort(dist, axis=0)[int(np.ceil(0.98*dist.shape[0]))]#np.mean(dist) + 5 * np.std(dist) # potrebno? 
        # weights = (1 - (dist / weight_cutoff) ** 2) ** 2
        # weights[dist > weight_cutoff] = 0

        # residuals.append(np.tile(weights, 2) * diff.ravel())
        residuals.append(diff.ravel())

    S = np.hstack(residuals)
    
    if S.shape[0] < x.shape[0]:
        raise Exception('Too few matches! ({:d} of {:d})'.format(S.shape[0], x.shape[0]))

    return S


def reprojection_error_X_only(x, Rt, visibility, points2D, N_frames, K, weight_cutoff=15, debug=False, shape=None, title=''):
    """
    Reprojection error, minimized in bundle adustment
    S = sum(d(P_i X_j, x_j)) (H&Z, p. 434)
    """    
   
    # unpack 3D points coordiantes, convert to homogenous
    X_ = x.reshape(-1, 3).T
    X = np.vstack((X_, np.ones(X_.shape[1])))

    # camera matrices
    cameras = [K.dot(_) for _ in Rt]

    if debug:
        fig, ax = plt.subplots(1, len(cameras), figsize=(4*len(cameras), 3))
    
    # reprojection error
    residuals = []
    for i, P in enumerate(cameras):
        reprojected_points = np.dot(P, X[:, visibility[i]])
        image_points = points2D[i, visibility[i]].T

        #debugging
        if debug:
            #fig = plt.figure()
            plt.axis('equal')
            plt.suptitle(title)
            ax[i].set_title('camera {:d}'.format(i))
            ax[i].text(0.01, 0.92, '{:d} points'.format(image_points.shape[1]), 
                        transform=ax[i].transAxes, fontsize=10)
            if shape is not None:
                ax[i].set_ylim(0, shape[0])
                ax[i].set_xlim(0, shape[1])
            ax[i].scatter(image_points[0], image_points[1], c='C0', alpha=0.75, label='feature points')
            ax[i].scatter(reprojected_points[0], reprojected_points[1], c='C3', alpha=0.5, label='reprojected')
            ax[i].legend(loc=3)
            #plt.show()
        
        diff = reprojected_points[:2] - image_points[:2]
        # dist = np.sqrt(np.sum(diff**2, axis=0))

        # # Tukey weights (https://github.com/dfridovi/SimpleSFM)
        # weight_cutoff = np.mean(dist) + 5 * np.std(dist) # potrebno? 
        # weights = (1 - (dist / weight_cutoff) ** 2) ** 2
        # weights[dist > weight_cutoff] = 0
        
        #residuals.append(np.tile(weights, 2) * diff.ravel())
        residuals.append(diff.ravel())
    
    S = np.hstack(residuals)

    if S.shape[0] < x.shape[0]:
        raise Exception('Too few matches! ({:d} of {:d})'.format(S.shape[0], x.shape[0]))

    return S


def calibrate_camera(files, pattern_size, d_pattern, scale=0.25):
    """
    Given the images of a calibration pattern of specified 
    dimensions and block size find the camera calibration matrix
    """
    
    views = [read_bw(file) for file in files]
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    object_points = []
    image_points = []
    object_p = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    object_p[:,:2] = np.mgrid[0:pattern_size[0]*d_pattern[0]:d_pattern[0], 
                              0:pattern_size[1]*d_pattern[1]:d_pattern[1]].T.reshape(-1,2)

    for i, im in enumerate(views):
        img = cv2.resize(im, dsize=(0, 0), fx=scale, fy=scale)
        found, corners = cv2.findChessboardCorners(img, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH)
        if found:
            image_points.append(cv2.cornerSubPix(img, corners,(5,5), (-1,-1), criteria))
            object_points.append(object_p)
            
    scaled_im_points = np.array(image_points) / scale
    scaled_im_shape = views[0].shape[::-1]
    ret, K, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(np.array(object_points), scaled_im_points, 
                                                     scaled_im_shape, None, None)
    return K, distCoeffs

def match_features(im1, im2, scale, match_ratio, mode='ORB'):
    """
    Detect and matcch features in two grayscale images, return corresponding points.
    """
    descriptors = {'ORB': [cv2.ORB_create, cv2.NORM_HAMMING],
                   'SIFT': [cv2.xfeatures2d.SIFT_create, cv2.NORM_L2],
                   'SURF': [cv2.xfeatures2d.SURF_create, cv2.NORM_L2]}

    im1 = cv2.resize(im1, dsize=(0,0), fx=scale, fy=scale) # query  
    im2 = cv2.resize(im2, dsize=(0,0), fx=scale, fy=scale) # train
    
    if mode in descriptors.keys():
        descriptor = descriptors[mode][0](5000)
        matcher = cv2.BFMatcher(descriptors[mode][1], crossCheck=False)
    else:
        print('Invalid feature matching mode, using ORB.')
        descriptor = descriptors['ORB'][0](5000)
        matcher = cv2.BFMatcher(descriptors['ORB'][1], crossCheck=False)

    # feature detection
    kp1, des1 = descriptor.detectAndCompute(im1, mask=None)
    kp2, des2 = descriptor.detectAndCompute(im2, mask=None)
    
    # brute-force feature matching:
    matches = matcher.knnMatch(des1, des2, k=2) # k matches for each feature for ratio test
    
    good = []
    for m, n in matches:
        if m.distance < match_ratio * n.distance:
            good.append(m)
            
    p1 = np.array([kp1[m.queryIdx].pt for m in good]) / scale
    p2 = np.array([kp2[m.trainIdx].pt for m in good]) / scale
    return p1, p2

def get_fundamental_matrix(p1, p2, K, d_max=1, alpha=0.995):
    """
    Given corresponding point pairs, calculate the fundamental matrix using RANSAC.
    """    
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, d_max, alpha)
    
    if F is None:
        print(len(p1))
        raise Exception('F problem ({:d} matches)'.format(len(p1)))
    
    p1 = p1[mask.ravel().astype(bool)]
    p2 = p2[mask.ravel().astype(bool)]

    inv_K = np.linalg.inv(K)
    p1_ = np.column_stack((p1, np.ones(p1.shape[0]))).T
    p2_ = np.column_stack((p2, np.ones(p2.shape[0]))).T
    p1_ = np.dot(inv_K, p1_)[:2].T
    p2_ = np.dot(inv_K, p2_)[:2].T
    F, mask = cv2.findFundamentalMat(p1_, p2_, cv2.FM_8POINT)
    return np.dot(inv_K.T, F).dot(inv_K), p1, p2

def get_camera_matrices(im1, im2, K, P1=None, d_max=1, alpha=0.995, **kwargs):
    """
    Given two images, calculate point matches and camera matrices.
    P = [R|t] ... normalized camera matrix; P1 = [I|0] (H&Z, p. 257)
    """
    
    if P1 is None:
        P1 = np.column_stack((np.eye(3), np.zeros(3)))
    
    # feature matching
    p1, p2 = match_features(im1, im2, **kwargs)
    
    # fundamental matrix and reduced points
    F, p1, p2 = get_fundamental_matrix(p1, p2, K, d_max, alpha)

    # essential matrix (H&Z, p. 257)
    E = K.T.dot(F).dot(K)
    
    U, D, VT = np.linalg.svd(E) # np.linalg.svd -> U, D, V.T
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]], dtype=float)
    
    u3 = U[:, -1]
    R1 = U.dot(W).dot(VT)
    R2 = U.dot(W.T).dot(VT)
    #positive determinants
    R1 = R1 * np.sign(np.linalg.det(R1))
    R2 = R2 * np.sign(np.linalg.det(R2)) 
    
    # four possibilities for the second camera matrix P2 (H&Z, p. 259)
    P1_4 = np.vstack((P1, np.array([0, 0, 0, 1])))
    P2_1 = np.column_stack((R1, u3)).dot(P1_4)
    P2_2 = np.column_stack((R1, -u3)).dot(P1_4)
    P2_3 = np.column_stack((R2, u3)).dot(P1_4)
    P2_4 = np.column_stack((R2, -u3)).dot(P1_4)
    P2_list = [P2_1, P2_2, P2_3, P2_4]
    
    # test points to determine if projections are in front of both cameras
    tally = np.zeros(4)
    for x1, x2 in zip(p1, p2):
        for i, P2 in enumerate(P2_list):
            X = triangulate_lsq(to_homogenous(x1), to_homogenous(x2), P1, P2)
            if in_front(X, P2) and in_front(X, P1):
                tally[i] += 1
    P2 = P2_list[np.argmax(tally)]
    
    return P1, P2, p1, p2

def get_camera_matrices_PnP(im1, im2, K, P1, scene, frame_i, distCoeffs, d_max=1, alpha=0.995, **kwargs):
    """
    Given a pair of images and known 3D points, calculate P2 using PnPRANSAC.
    P = [R|t] ... normalized camera matrix; P1 = [I|0] (H&Z, p. 257)
    """
  
    # feature matching
    p1, p2 = match_features(im1, im2, **kwargs)

    # searching the scene for existing matching points from previous frame
    feature_keys = [(frame_i, tuple(_)) for _ in p1]
    matching_p2 = []
    matching_X = []
    for i, key in enumerate(feature_keys):
        if key in scene.points3D.keys():
            matching_p2.append(p2[i])
            matching_X.append(from_homogenous(scene.points3D[key].X))

    if len(matching_X) < 50:
        P1, P2, p1, p2 = get_camera_matrices(im1, im2, K, P1, d_max=d_max, alpha=alpha, **kwargs)
        print('Too few ({:d}) matches for PnPRANSAC, calculating Fundamental matrix.'.format(len(matching_X)))
    else:
        print('PnPRANSAC ', len(matching_X))
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(matching_X), 
                                                 np.array(matching_p2), 
                                                 cameraMatrix=K,
                                                 distCoeffs=distCoeffs)

        # outlier rejection through RANSAC
        F, p1, p2 = get_fundamental_matrix(p1, p2, K, d_max=d_max, alpha=alpha)

        R2, jac = cv2.Rodrigues(rvec)
        R2 = R2 * np.sign(np.linalg.det(R2))
        P2 = np.column_stack((R2, tvec))

    return P1, P2, p1, p2


def in_front(X, Rt):
    """
    True if X is in front of the camera center C (if positive z coordinate).
    C = -R.T t (H&Z, p. 156)
    """
    R = Rt[:, :3]
    t = Rt[:, -1]
    C = np.dot(-R.T, t)

    if np.dot(R[2], X[:3] - C) > 0:
        return True
    return False

def pos_from_P(P):
    """ Convert a camera projection matrix to its pose. """
    R = P[:, :-1]
    t = P[:, -1] 
    pos = np.eye(4)
    pos[:-1, :-1] = R.T
    pos[:-1, -1] = np.dot(np.linalg.inv(R), -t)
    return pos

# Reading the Middlebury dataset
def K_from_text(line):
    vals = np.array([float(_) for _ in line.split(' ')[1:]])
    return vals[:9].reshape(3,3)

def Rt_from_text(line):
    vals = np.array([float(_) for _ in line.split(' ')[1:]])
    R = vals[9:-3].reshape(3,3)
    t = vals[-3:]
    return np.column_stack((R, t))

def P_from_text(line):
    """ P = K[R|t] (H&Z, p. 156) """
    K = K_from_text(line)
    Rt = Rt_from_text(line)
    return np.dot(K, Rt)

def camera_matrices_from_txt(file):
    with open(file, 'r') as f:
        lines = f.readlines()[1:]
        Rts = np.array([ Rt_from_text(line) for line in lines ])
        K = K_from_text(lines[0])
    return Rts, K


# pyqtgraph render point cloud
def pg_render(XYZ, cXYZ=None):
    """ Draw the scene using pyqtgraph. """
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 1
    w.show()
    w.setWindowTitle('3d point cloud')

    gx = gl.GLGridItem()
    w.addItem(gx)

    plot = gl.GLScatterPlotItem(pos=XYZ, color=np.ones((XYZ.shape[0], 3)), size=2, pxMode=True)
    w.addItem(plot)

    if cXYZ is not None:
        #c_colors = np.tile(np.array([0., 1., 0.]), (cXYZ.shape[0], 1))
        N_cams = cXYZ.shape[0]
        r = np.linspace(0, 1., N_cams)
        g = np.zeros(N_cams)
        b = r[::-1]
        c_colors = np.column_stack((r, g, b))
        
        camera_plot = gl.GLScatterPlotItem(pos=cXYZ, color=c_colors, size=20, pxMode=True)
        w.addItem(camera_plot)
    
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(app.exec_())