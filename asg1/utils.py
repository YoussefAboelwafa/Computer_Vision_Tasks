import matplotlib.pyplot as plt
import cv2 as cv


def show_image(image, title=None):
    plt.imshow(image, cmap="gray" if len(image.shape) == 2 else None)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def read_image(path):
    img = cv.imread(path)
    return img


def BGR2RGB(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)
