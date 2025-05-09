import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import random


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


def compute_H(correspondences):
    """
    A function that computes the homography matrix H
    from the given correspondences.
    :param correspondences: A list of correspondences.
    :return: The computed homography matrix H.
    """
    A = []
    for p, p_dash in correspondences:
        A.append(
            [-p[0], -p[1], -1, 0, 0, 0, p[0] * p_dash[0], p[1] * p_dash[0], p_dash[0]]
        )
        A.append(
            [0, 0, 0, -p[0], -p[1], -1, p[0] * p_dash[1], p[1] * p_dash[1], p_dash[1]]
        )

    A = np.array(A)
    U, D, V_transpose = np.linalg.svd(A)
    H = np.reshape(V_transpose[8], (3, 3))
    H /= H[2, 2]

    return H


def RANSAC(correspondences):
    """
    A function that implements the RANSAC algorithm to find the best
    homography matrix H from the given correspondences.
    :param correspondences: A list of correspondences.
    :return: The best homography matrix H.
    """
    max_inliers = []

    for _ in range(500):

        random_indices = random.sample(range(0, len(correspondences)), 4)
        random_correspondences = [correspondences[i] for i in random_indices]

        H = compute_H(random_correspondences)

        curr_inliers = []
        for corr in correspondences:
            P = corr[0]
            mapped_P = tuple(
                map(
                    int,
                    (
                        np.dot(H, np.transpose([P[0], P[1], 1]))
                        / (np.dot(H, np.transpose([P[0], P[1], 1]))[2])
                    ).astype(int)[:2],
                )
            )
            e = np.linalg.norm(np.asarray(corr[1]) - np.asarray(mapped_P))
            if e < 5:
                curr_inliers.append(corr)

        if len(curr_inliers) > len(max_inliers):
            max_inliers = curr_inliers

    return compute_H(max_inliers)


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
        rgb_frame = BGR2RGB(frame)
        frames.append(rgb_frame)
    cap.release()
    return frames


def RGB2GRAY(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray


def BGR2RGB(image):
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return rgb
