import os
import cv2 as cv
import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.feature.adalam import AdalamFilter
from kornia.geometry.conversions import (axis_angle_to_quaternion,
                                         quaternion_to_rotation_matrix)
from kornia.geometry.epipolar import essential_from_Rt

from helpers import angle_between_quaternions, calculate_q_t_error, convert_to_quaternion, get_F_from_K_E, recover_pose
from utils.vision.opencv.estimation_utils import estimate_essential_matrix
from utils.vision.opencv.interactive import fundamental_explorer


def _match_pts(descriptor, pt_idx_list, des2, ratio, norm):
    # print(len(pt_idx_list), end=" ")

    # descriptor is numpy array of shape (K,)
    # des2 is numpy array of shape (N, K)
    # pt_idx_list is a list of indices of des2.
    # Returns the index of the best match in pt_idx_list.
    distance_closest = np.inf
    distance_second_closest = np.inf
    idx_closest = None
    for idx in pt_idx_list:
        # distance = np.linalg.norm(descriptor - des2[idx])  # TODO: Önceki kod buydu. Gerekirse buna döndür.
        # Get hamming distance between descriptor and des2[idx]. Use OpenCV
        distance = cv.norm(descriptor, des2[idx], norm) # type: ignore
        if distance < distance_closest:
            distance_second_closest = distance_closest
            distance_closest = distance
            idx_closest = idx
        elif distance < distance_second_closest:
            distance_second_closest = distance

    if distance_closest / distance_second_closest <= ratio:
        return idx_closest, distance_closest
    else:
        return None, None


def perform_guided_matching_for_essential_matrix_estimation(kp1, kp2, des1, des2, E_coarse, K, max_distance_in_pixels=20, is_adaptive=True, ratio=0.8, scale_tolerance=None, norm=cv.NORM_L2):
    assert E_coarse is not None  # None ise guide edecek bir şey yok, gerçi sıfırdan guided matching yapabiliriz belki. Performans kazandırır.
    assert not is_adaptive  # TODO Implement this. (?)
    assert scale_tolerance is None  # TODO Implement this. (?)

    pts1 = np.array([kp.pt for kp in kp1])
    #pts2 = np.array([kp.pt for kp in kp2])
    pts2 = np.array([(*kp.pt, 1) for kp in kp2])

    F_coarse = np.linalg.inv(K).T @ E_coarse @ np.linalg.inv(K)
    pts1_transformed_as_lines = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F_coarse).reshape(-1, 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        points_tensor = torch.tensor(
            pts2,
            device=device,
        )
        lines_tensor = torch.tensor(
            pts1_transformed_as_lines,
            device=device,
        )

        distances = torch.abs(torch.matmul(lines_tensor, points_tensor.T)) / torch.norm(
            lines_tensor[:, :2], dim=1, keepdim=True
        )


        filtered_points = (distances <= max_distance_in_pixels).detach().cpu().numpy()

    pt_idx_list_list = [np.where(filtered_points[i])[0] for i in range(filtered_points.shape[0])]


    # TODO Make this efficient!

    # pt_idx_list_list = []
    # for pt1_transformed_as_line in pts1_transformed_as_lines:
    #     a, b, c = pt1_transformed_as_line
    #     pt_idx_list = []
    #     for idx, pt2 in enumerate(pts2):
    #         x, y = pt2
    #         distance = np.abs(a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)
    #         if distance <= max_distance_in_pixels:
    #             pt_idx_list.append(idx)
    #     pt_idx_list_list.append(pt_idx_list)

    matches = []
    for idx, pt_idx_list in enumerate(pt_idx_list_list):
        if len(pt_idx_list) == 0:
            continue

        idx_closest, distance_closest = _match_pts(des1[idx], pt_idx_list, des2, ratio, norm)
        if idx_closest is None:
            continue
        match = cv.DMatch(idx, idx_closest, distance_closest) # type: ignore
        matches.append(match)

    return matches


def _read_image_pair_and_K_q_t(dataset, category, img_pair_name, dataset_dir='datasets'):
    img0_path = f"{dataset_dir}/{dataset}/{category}/{img_pair_name}/0.jpg"
    img1_path = f"{dataset_dir}/{dataset}/{category}/{img_pair_name}/1.jpg"
    K_path = f"{dataset_dir}/{dataset}/{category}/{img_pair_name}/K.txt"
    q_path = f"{dataset_dir}/{dataset}/{category}/{img_pair_name}/q.txt"
    t_path = f"{dataset_dir}/{dataset}/{category}/{img_pair_name}/t.txt"

    assert os.path.exists(img0_path), f"{img0_path} does not exist."
    assert os.path.exists(img1_path), f"{img1_path} does not exist."
    assert os.path.exists(K_path), f"{K_path} does not exist."

    img0 = cv.imread(img0_path, cv.IMREAD_GRAYSCALE)
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    K = np.loadtxt(K_path)

    if os.path.exists(q_path):
        q = np.loadtxt(q_path)
    else: 
        q = None
    
    if os.path.exists(t_path):
        t = np.loadtxt(t_path)
    else:
        t = None

    return img0, img1, K, q, t


def _read_image_pair_and_K_E(dataset, category, img_pair_name, dataset_dir='datasets'):
    img0, img1, K, q, t = _read_image_pair_and_K_q_t(dataset, category, img_pair_name, dataset_dir)

    if q is None:  # Unknown
        assert t is None  # Unknown
        E = None
    
    else:
        R2 = quaternion_to_rotation_matrix(axis_angle_to_quaternion(torch.tensor([0.0, 0.0, 0.0]))).reshape(1, 3, 3)
        R1 = quaternion_to_rotation_matrix(torch.tensor(q, dtype=R2.dtype)).reshape(1, 3, 3)

        t2 = torch.zeros(3, dtype=R1.dtype).reshape(1, 3, 1)
        t1 = torch.tensor(t, dtype=R2.dtype).reshape(1, 3, 1)

        # Important: I had to reorder R2 and R1, also t2 and t1. References are R2 and t2.

        E = essential_from_Rt(R1, t1, R2, t2).reshape(3, 3).numpy()
    return img0, img1, K, E


def read_image_pair_and_F(dataset, category, img_pair_name, dataset_dir='datasets'):
    img0, img1, K, E = _read_image_pair_and_K_E(dataset, category, img_pair_name, dataset_dir)
    if E is None:
        F = None
    else:
        F = get_F_from_K_E(K, E)
    return img0, img1, F
    

def match_with_kornia(img0_path, img1_path, K_matrix, feature_count):
    device = K.utils.get_cuda_or_mps_device_if_available()
    print(device)

    fname1 = img0_path
    fname2 = img1_path

    img1 = K.io.load_image(fname1, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    img2 = K.io.load_image(fname2, K.io.ImageLoadType.RGB32, device=device)[None, ...]

    feature = KF.SIFTFeature(feature_count).eval().to(device)

    input_dict = {
        "image0": K.color.rgb_to_grayscale(img1),  # LofTR works on grayscale images only
        "image1": K.color.rgb_to_grayscale(img2),
    }

    hw1 = torch.tensor(img1.shape[2:])
    hw2 = torch.tensor(img1.shape[2:])

    adalam_config = {"device": device}

    with torch.inference_mode():
        lafs1, resps1, descs1 = feature(K.color.rgb_to_grayscale(img1))
        lafs2, resps2, descs2 = feature(K.color.rgb_to_grayscale(img2))
        
        dists, idxs = KF.match_adalam(
            descs1.squeeze(0),
            descs2.squeeze(0),
            lafs1,
            lafs2,  # Adalam takes into account also geometric information
            config=adalam_config,
            hw1=hw1,
            hw2=hw2,  # Adalam also benefits from knowing image size
        )

    print(f"{idxs.shape[0]} tentative matches with AdaLAM")


    def get_matching_keypoints(lafs1, lafs2, idxs):
        mkpts1 = KF.get_laf_center(lafs1).squeeze()[idxs[:, 0]].detach().cpu().numpy()
        mkpts2 = KF.get_laf_center(lafs2).squeeze()[idxs[:, 1]].detach().cpu().numpy()
        return mkpts1, mkpts2


    mkpts1, mkpts2 = get_matching_keypoints(lafs1, lafs2, idxs)

    E, mask = cv2.findEssentialMat(mkpts1, mkpts2, cameraMatrix=K_matrix, method=cv2.USAC_MAGSAC, prob=0.999, threshold=0.75)
    print(f"{(mask > 0).sum()} inliers with AdaLAM")

    inlier_pts1 = mkpts1[mask.ravel() == 1]
    inlier_pts2 = mkpts2[mask.ravel() == 1]

    R_estimated, t_estimated, final_mask = recover_pose(E, inlier_pts1, inlier_pts2, K_matrix)
    final_mask = final_mask.ravel()

    inlier_count = np.count_nonzero(final_mask)

    q_estimated = convert_to_quaternion(R_estimated)
    assert isinstance(q_estimated, np.ndarray)

    q_estimated = q_estimated
    t_estimated = t_estimated

    reference_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # (0, 0, 0) as axis-angle 
    angle_estimated = angle_between_quaternions(q_estimated, reference_quaternion)
    angle_estimated = np.degrees(angle_estimated)  # angle_translation diye bir şey hesaplanamıyor...

    # calculate F
    F = get_F_from_K_E(K_matrix, E)

    return E, F, angle_estimated, q_estimated, t_estimated, inlier_count, mkpts1, mkpts2, descs1, descs2


def guided_matching(kp1, kp2, des1, des2, E_coarse, max_distance_in_pixels, is_adaptive, ratio, scale_tolerance, norm):

    matches = perform_guided_matching_for_essential_matrix_estimation(kp1, kp2, des1, des2, E_coarse, K_matrix, max_distance_in_pixels, is_adaptive, ratio, scale_tolerance, norm)

    pts1 = np.int32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.int32([kp2[m.trainIdx].pt for m in matches])

    method = cv.USAC_ACCURATE
    prob = 0.999
    threshold = 1.0
    E_final, inlier_pts1, inlier_pts2, mask = estimate_essential_matrix(pts1, pts2, K_matrix, method, prob, threshold)

    R_estimated, t_estimated, final_mask = recover_pose(E_final, inlier_pts1, inlier_pts2, K_matrix)
    final_mask = final_mask.ravel()

    inlier_count = np.count_nonzero(final_mask)

    q_estimated = convert_to_quaternion(R_estimated)
    assert isinstance(q_estimated, np.ndarray)

    q_estimated = q_estimated
    t_estimated = t_estimated

    reference_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # (0, 0, 0) as axis-angle 
    angle_estimated = angle_between_quaternions(q_estimated, reference_quaternion)
    angle_estimated = np.degrees(angle_estimated)  # angle_translation diye bir şey hesaplanamıyor...

    F_final = get_F_from_K_E(K_matrix, E_final)

    return E_final, F_final, angle_estimated, q_estimated, t_estimated, inlier_count


dataset = "ersin"
category = "meetingroom"
img_pair_name = "210-215"

img_pair_path = "datasets/ersin/meetingroom/210-215"
img0_path = f"{img_pair_path}/0.jpg"
img1_path = f"{img_pair_path}/1.jpg"
K_path = f"{img_pair_path}/K.txt"
q_path = f"{img_pair_path}/q.txt"
t_path = f"{img_pair_path}/t.txt"

K_matrix = np.loadtxt(K_path)
q_gt = np.loadtxt(q_path)
t_gt = np.loadtxt(t_path)

feature_counts = [500, 1000]
results = {}

for feature_count in feature_counts:
    kornia_result_path = f'{dataset}_{category}_{img_pair_name}_F_{feature_count}.txt'
    if os.path.exists(kornia_result_path):
        E = np.loadtxt(f'{dataset}_{category}_{img_pair_name}_E_{feature_count}.txt')
        F = np.loadtxt(kornia_result_path)
        angle_estimated = np.loadtxt(f'{dataset}_{category}_{img_pair_name}_angle_{feature_count}.txt')
        q_estimated = np.loadtxt(f'{dataset}_{category}_{img_pair_name}_q_{feature_count}.txt')
        t_estimated = np.loadtxt(f'{dataset}_{category}_{img_pair_name}_t_{feature_count}.txt')
        inlier_count = np.loadtxt(f'{dataset}_{category}_{img_pair_name}_inlier_count_{feature_count}.txt')

    else:
        E, F, angle_estimated, q_estimated, t_estimated, inlier_count, mkpts1, mkpts2, descs1, descs2 = match_with_kornia(
            img0_path,
            img1_path,
            K_matrix, feature_count)
        # np.savetxt(f'{dataset}_{category}_{img_pair_name}_E_{feature_count}.txt', E)
        # np.savetxt(kornia_result_path, F)
        # np.savetxt(f'{dataset}_{category}_{img_pair_name}_angle_{feature_count}.txt', np.array([angle_estimated]))
        # np.savetxt(f'{dataset}_{category}_{img_pair_name}_q_{feature_count}.txt', q_estimated)
        # np.savetxt(f'{dataset}_{category}_{img_pair_name}_t_{feature_count}.txt', t_estimated)
        # np.savetxt(f'{dataset}_{category}_{img_pair_name}_inlier_count_{feature_count}.txt', np.array([inlier_count]))

    if q_gt is not None:
        assert t_gt is not None
        q_err, t_err = calculate_q_t_error(q_gt, t_gt, q_estimated, t_estimated, verbose=False)
    else:
        q_err = float('nan')
        t_err = float('nan')
    
    results[feature_count] = (E, F, angle_estimated, q_estimated, t_estimated, inlier_count, q_err, t_err, mkpts1, mkpts2, descs1, descs2)


feature_count_to_perform_guided_matching = 500
E = results[feature_count_to_perform_guided_matching][0]
kp1 = [cv.KeyPoint(pt[0], pt[1], 1) for pt in results[feature_count_to_perform_guided_matching][8]]
kp2 = [cv.KeyPoint(pt[0], pt[1], 1) for pt in results[feature_count_to_perform_guided_matching][9]]
des1 = results[feature_count_to_perform_guided_matching][10].squeeze().detach().cpu().numpy()
des2 = results[feature_count_to_perform_guided_matching][11].squeeze().detach().cpu().numpy()

E_guided1, F_guided1, angle_estimated_guided1, q_estimated_guided1, t_estimated_guided1, inlier_count_guided1 = \
    guided_matching(kp1, kp2, des1, des2, E, 5, False, 1.0, None, cv.NORM_L2)

# TODO Bizde Hamming!!!

q_err_guided1, t_err_guided1 = calculate_q_t_error(q_gt, t_gt, q_estimated_guided1, t_estimated_guided1, verbose=False) 


E_guided2, F_guided2, angle_estimated_guided2, q_estimated_guided2, t_estimated_guided2, inlier_count_guided2 = \
    guided_matching(kp1, kp2, des1, des2, E, 3, False, 1.0, None, cv.NORM_L2)

# TODO Bizde Hamming!!!

q_err_guided2, t_err_guided2 = calculate_q_t_error(q_gt, t_gt, q_estimated_guided2, t_estimated_guided2, verbose=False) 


E_guided3, F_guided3, angle_estimated_guided3, q_estimated_guided3, t_estimated_guided3, inlier_count_guided3 = \
    guided_matching(kp1, kp2, des1, des2, E, 5, False, 0.9, None, cv.NORM_L2)  # 30 kötü oldu. 5 deneyelim.

# TODO Bizde Hamming!!!

q_err_guided3, t_err_guided3 = calculate_q_t_error(q_gt, t_gt, q_estimated_guided3, t_estimated_guided3, verbose=False) 



reference_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # (0, 0, 0) as axis-angle 
correct_angle = np.degrees(angle_between_quaternions(q_gt, reference_quaternion))

img0, img1, F_gt = read_image_pair_and_F(dataset, category, img_pair_name)


fundamental_explorer.explore_transformation(
    img0, 
    img1, 
    [results[feature_count][1] for feature_count in feature_counts] + [F_guided1, F_guided2, F_guided3],
    [f"kornia {feature_count} (angle:{results[feature_count][2]:.2f}, q_err:{results[feature_count][6]:.2f}, t_err:{results[feature_count][7]:.2f}, inliers:{results[feature_count][5]})" for feature_count in feature_counts]
    +
    [
     f"guided matching 1 {feature_count_to_perform_guided_matching} (angle:{angle_estimated_guided1:.2f}, q_err:{q_err_guided1:.2f}, t_err:{t_err_guided1:.2f}, inliers:{inlier_count_guided1})",
     f"guided matching 2 {feature_count_to_perform_guided_matching} (angle:{angle_estimated_guided2:.2f}, q_err:{q_err_guided2:.2f}, t_err:{t_err_guided2:.2f}, inliers:{inlier_count_guided2})",
     f"guided matching 3 {feature_count_to_perform_guided_matching} (angle:{angle_estimated_guided3:.2f}, q_err:{q_err_guided3:.2f}, t_err:{t_err_guided3:.2f}, inliers:{inlier_count_guided3})"
    ],
    [(255, 155, 155), (255, 200, 200), (0, 255, 0), (0, 0, 255), (255, 0, 0)],  # (255, 0, 0), 
    None,  # F_gt
    f"Ground truth (angle: {correct_angle:.2f})"
)
