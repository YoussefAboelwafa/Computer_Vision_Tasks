import matplotlib.pyplot as plt
import cv2 as cv


def show_image(image, title=None):
    plt.imshow(image, cmap="gray" if len(image.shape) == 2 else None)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def show_diff_image(image1, image2, title1=None, title2=None):
    fig, ax = plt.subplots(1, 2)
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


def BGR2RGB(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)
