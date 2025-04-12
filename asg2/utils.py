import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def compute_SIFT_kps_and_descriptors(image):
    """
    A function that computes the SIFT keypoints and descriptors
    for a given image.
    :param image: The input image.
    :return: A tuple containing the keypoints and descriptors.
    """
    gray_img = RGB2GRAY(image)
    sift = cv.SIFT_create()
    kps, des = sift.detectAndCompute(gray_img, None)
    return kps, des

def get_top_correspondences(des1, des2, count=None):
    """
    A function that takes two sets of descriptors and computes
    the top correspondences between them.
    :param des1: The first set of descriptors.
    :param des2: The second set of descriptors.
    :param count: The number of top correspondences to return.
    :return: A list of matches.
    """
    bf = cv.BFMatcher()
    frame_matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good = []
    for m, n in frame_matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])
    
    if count is not None:
        # Pick the best matches
        good = sorted(good, key=lambda x: x[0].distance)[:count]
    return good


import numpy as np
import cv2 as cv

def normalize_points(pts):
    """Normalize points so that the centroid is at the origin and the mean distance is sqrt(2)."""
    pts = np.array(pts, dtype=np.float32)
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0).mean()
    if std == 0:
        std = 1
    # Scale so that mean distance is sqrt(2)
    scale = np.sqrt(2) / std
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0,    0,              1]])
    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_hom.T).T
    return pts_norm[:, :2], T

def compute_homography(matches, kp, kp_video):
    """
    A function that takes a set of matches and computes
    the associated 3x3 homography matrix H using the normalized DLT algorithm.
    
    :param kp: list of keypoints in the first image.
    :param kp_video: list of keypoints in the second image.
    :param matches: list of opencv matches.
    :return: Homography matrix.
    """
    if len(matches) < 4:
        raise ValueError("At least four matches are required to compute homography.")

    # Extract the points for both views
    src_pts = np.float32([kp[m[0].queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp_video[m[0].trainIdx].pt for m in matches]).reshape(-1, 2)
    
    # Normalize points (just for numerical stability)
    src_pts_norm, T_src = normalize_points(src_pts)
    dst_pts_norm, T_dst = normalize_points(dst_pts)
    
    # Construct matrix A for normalized coordinates
    N = src_pts_norm.shape[0]
    A = []
    for i in range(N):
        x, y = src_pts_norm[i]
        _x, _y = dst_pts_norm[i]
        A.append([-x, -y, -1,   0,   0,  0, x*_x, y*_x, _x])
        A.append([ 0,   0,  0, -x, -y, -1, x*_y, y*_y, _y])
    A = np.array(A)
    
    # Solve using SVD
    U, S, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)
    
    # Denormalize: H = T_dst^-1 * H_norm * T_src
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    H /= H[2, 2]  # Normalize such that H[2,2] is 1
    
    return H


def show_image(image, title=None):
    plt.imshow(image, cmap="gray" if len(image.shape) == 2 else None)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

def show_diff(image1, image2, title1=None, title2=None):
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(image1, cmap="gray" if len(image1.shape) == 2 else None)
    ax[0].axis("off")
    ax[0].set_title(title1)
    ax[1].imshow(image2, cmap="gray" if len(image2.shape) == 2 else None)
    ax[1].axis("off")
    ax[1].set_title(title2)
    plt.show()

def read_image(path):
    img = cv.imread(path)
    return img

def read_image_gray(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return img

def read_image_rgb(path):
    img = cv.imread(path)
    img = BGR2RGB(img)
    return img

def read_video(path):
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(BGR2RGB(frame))
    cap.release()
    return frames

def RGB2GRAY(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray

def BGR2RGB(image):
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return rgb