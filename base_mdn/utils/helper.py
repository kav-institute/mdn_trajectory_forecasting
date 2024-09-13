import argparse
import numpy as np
import random


def config_parser():
    """parse command line arguments
    """
    
    p = argparse.ArgumentParser()
    p.add_argument('-t', '--target', type=str, default='imptc')
    p.add_argument('-c', '--configs', type=str, default='default_peds_imptc.json')
    p.add_argument('-l', '--log', action='store_true')
    p.add_argument('-p', '--print', action='store_true')
    p.add_argument('-g', '--gpu', type=str, default='0')
    p.set_defaults(log=True)
    p.set_defaults(print=True)
    
    return p


def build_R(rotation_angle):
    """build rotation matrix
    
    Args:
        rotation_angle (np.array): stack of rotation angles
        
    Returns:
        np.array: stack of rotation matrixes
    """
    
    # get identity matrices
    R = np.zeros((*rotation_angle.shape, 2, 2))
    R[..., 0, 0] = np.cos(rotation_angle)
    R[..., 0, 1] = np.sin(rotation_angle)
    R[..., 1, 0] = -np.sin(rotation_angle)
    R[..., 1, 1] = np.cos(rotation_angle)
    
    return R


def build_R_inv(rotation_angle):
    """build inverse of rotation matrix
    
    Args:
        rotation_angle (np.array): stack of rotation angles
        
    Returns:
        np.array: stack of inverted rotation matrixes
    """
    
    # get identity 
    rotation_angle = -rotation_angle
    R_inv = np.zeros((*rotation_angle.shape, 2, 2))
    R_inv[..., 0, 0] = np.cos(rotation_angle)
    R_inv[..., 0, 1] = np.sin(rotation_angle)
    R_inv[..., 1, 0] = -np.sin(rotation_angle)
    R_inv[..., 1, 1] = np.cos(rotation_angle)
    
    return R_inv


def bat_mat_vec_mult(a, M):
    """simple helper for batched matrix multiplication (with einsum)
    
    Args:
        a (np.array): stack of track data
        M rotation_angle (np.array): stack of rotation matrixes
    """
    
    # batched matrix multiplication
    return np.einsum('...ij,...j->...i', M[...,np.newaxis,:,:], a, optimize=True)


def calc_velocities(t, s):
    """calc velocities for ego shifted position data
    
    Args:
        t (np.array): sample data with [x,y,vx,vy]
        s (int): sample rate
        
    Returns:
        np.array: track data with velocities
    """
    
    # isolate x and y positions, skip last one
    x0 = t[:,:-1,0]
    y0 = t[:,:-1,1]
    
    # isolate x and y positions as shift of one time-step, skip first one
    x1 = t[:,1:,0]
    y1 = t[:,1:,1]
    
    # re-calculate vx and vy in ego coords
    t[:,1:,2] = (x1 - x0) / (1/s)
    t[:,1:,3] = (y1 - y0) / (1/s)
    
    # copy velocities from t+1 to t+0
    t[:,0,2] = t[:,1,2]
    t[:,0,3] = t[:,1,3]
    
    return t


def crop_trajectory(data, win_size, fh, shift=1, dims=2):
    """high level method to partition a large trajectory with a sliding window
    
    Args:
        a (np.array): full track data
        n_in (int): size of the input window
        s (int): shift size between two sliding window crops
        n_out (int): size of the ground truth window
        dims (int): the dimensions of the trajectory to be used
        
    Raises:
        ValueError: dimensionally error
        
    Returns:
        np.array: the cropped input trajectory parts as stack
    """
    
    if dims > data.shape[-1]:
        
        raise ValueError('The trajectory only has %d dimensions' % data.shape[-1])
    
    T = rolling_window(data, shape=(win_size, dims), shift=shift)
    X = T[:,0:-fh,:]
    y = T[:,-fh:,:]
    
    return X, y, T.shape[0]


def rolling_window(a, shape, shift):
    """sliding window over a trajectory
    
    Args:
        a (np.array): input trajectory (e.g. 2d trajectory got the shape (n_samples,2))
        shape (tuple): shape for the sliding window (e.g. for 50 samples of a 2d trajectory use (50,2))
        shift (int): shift size between to sliding window extractions
        
    Returns:
        np.array: rolling window (e.g. for 50 samples of a 2d trajectory, the result has the shape (n_samples,50,2))
    """
    
    s = (a.shape[-2] - shape[-2] + 1,) + (a.shape[-1] - shape[-1] + 1,) + shape
    strides = a.strides + a.strides
    astrided_array = np.lib.stride_tricks.as_strided(a, shape=s, strides=strides).squeeze()
    
    if astrided_array.ndim < 3:
        
        astrided_array = astrided_array[None,...]
        
    elif astrided_array.ndim > 3:
        
        print ("Error too many dimensions in rolling window result...")
    
    # apply shift size and return
    return astrided_array[::shift]


def estimate_ego_transform(X):
    """estimate a vrus moving direction 
    
    Args:
        X (np.array): stack of track data
        
    Returns:
        np.array: rotation angles
        np.array: reference positions
    """
    
    # apply small amount of random noise (sub-mm area) to data, to prevent zero divisions for standing objects
    X = X + np.random.normal(0, 0.001, X.shape)
    
    # get last point as reference position
    reference_positions = X[:,-1:,0:2]
    X_translated = X[:,:,0:2] - reference_positions
    X_normed = X_translated / np.linalg.norm(X_translated)
    direction = np.sum(X_normed, axis=1)
    
    # determine angle between the orientation and the y-axis
    rotation_angles = np.arctan2(direction[:,0], -direction[:,1])
    
    # check for broken rotation angles and fix them
    for idx, _ in enumerate(rotation_angles):
        
        if np.isnan(rotation_angles[idx]):
            
            rotation_angles[idx] = np.random.uniform(low=1.50, high=1.64, size=1)
            
    return rotation_angles, reference_positions


def world2ego(X, rotation_angle, translation, sample_rate):
    """transform from world coordinates to ego coordinates
    
    Args:
        X (np.array): stack of track data
        rotation_angle (np.array): stack of rotation angles
        translation (np.array): stack of translation vectors
        
    Returns:
        np.array: stack of track data in ego coords
    """
    
    R = build_R(rotation_angle)
    X_t = X[:,:,0:2] - translation
    X_Rt = bat_mat_vec_mult(X_t,R)
    ego = calc_velocities(t=np.concatenate((X_Rt, X[:,:,0:2]), axis=2), s=sample_rate)
    
    return ego


def ego2world(X, rotation_angle, translation):
    """transform from ego coordinates to world coordinates
    
    Args:
        X (np.array): stack of track data
        rotation_angle (np.array): stack of rotation angles
        translation (np.array): stack of translation vectors
        
    Returns:
        np.array: stack of track data in world coords
    """
    
    R_inv = build_R_inv(rotation_angle)
    X_R = bat_mat_vec_mult(X,R_inv)
    X_tR = X_R + translation
    return X_tR


def count_model_parameters(model):
    """pytorch method to count a models parameter size
    """
    
    return sum(p.numel() for p in model.parameters())


def generate_unique_randoms(count, min_value, max_value):
    """generate unique random numbers within a given range
    """
    
    # create a set to store the generated random numbers
    generated_numbers = set()
    
    while len(generated_numbers) < count:
        
        # generate a new random number in the specified range
        new_number = random.randint(min_value, max_value)
        
        # if the number is not already in the set, add it to the set and continue generating numbers
        if new_number not in generated_numbers:
            
            generated_numbers.add(new_number)
            
    return list(generated_numbers)