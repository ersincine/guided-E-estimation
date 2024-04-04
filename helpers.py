from typing import Union

import cv2 as cv
import numpy as np
import torch
from kornia.geometry.conversions import rotation_matrix_to_quaternion

from utils.vision.opencv.epipolar_geometry import angle_between_vectors


def recover_pose(E, pts1, pts2, K):
    retval, R, t, mask = cv.recoverPose(E, pts1, pts2, K)
    # Important: Mask is not 0 or 1, but 0 or 255.

    assert np.isclose(np.linalg.det(R), 1, atol=1e-3)
    num_inliers_after_recover_pose = np.sum(mask != 0)
    assert num_inliers_after_recover_pose == retval

    return R, t, mask


def convert_to_quaternion(R: np.ndarray, as_numpy=True) -> Union[np.ndarray, torch.Tensor]:
    q = rotation_matrix_to_quaternion(torch.tensor(R).reshape(1, 3, 3)).reshape(4)
    if as_numpy:
        q = q.numpy()
    return q


def angle_between_quaternions(q1: np.ndarray, q2: np.ndarray, eps: float=1e-15) -> float:
    assert isinstance(q1, np.ndarray)
    assert isinstance(q2, np.ndarray)
    assert len(q1.shape) == 1 and q1.shape[0] == 4
    assert len(q2.shape) == 1 and q2.shape[0] == 4
    
    """
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    return 2 * np.arccos(np.abs(np.dot(q1, q2)))  # ?
    """
    
    q1 = q1 / (np.linalg.norm(q1) + eps)
    q2 = q2 / (np.linalg.norm(q2) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q1 * q2)**2))
    return np.arccos(1 - 2 * loss_q)


def get_F_from_K_E(K, E):
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    return F


def calculate_q_t_error(q_true: np.ndarray, t_true: np.ndarray, q_estimated: np.ndarray, t_estimated: np.ndarray,
                        verbose: bool=False) -> tuple[float, float]:
    
    assert isinstance(q_true, np.ndarray)
    assert isinstance(t_true, np.ndarray)
    assert isinstance(q_estimated, np.ndarray)
    assert isinstance(t_estimated, np.ndarray)
    
    t_estimated = t_estimated.flatten()
    t_true = t_true.flatten()
    
    assert len(q_true.shape) == 1 and q_true.shape[0] == 4
    assert len(t_true.shape) == 1 and t_true.shape[0] == 3
    assert len(q_estimated.shape) == 1 and q_estimated.shape[0] == 4
    assert len(t_estimated.shape) == 1 and t_estimated.shape[0] == 3
    
    if verbose:
        print(f'{q_true=}')
        print(f'{t_true=}')
        print(f'{q_estimated=}')
        print(f'{t_estimated=}')
    
    err_q = angle_between_quaternions(q_true, q_estimated)
    err_t = angle_between_vectors(t_true, t_estimated)

    assert not (np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t))), 'This should never happen! Debug here'

    return err_q, err_t
