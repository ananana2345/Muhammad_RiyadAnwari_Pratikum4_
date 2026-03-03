import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD IMAGES
# =========================
image1 = cv2.imread("image1.jpeg")
image2 = cv2.imread("image2.jpeg")
image3 = cv2.imread("image3.jpeg")

images = {
    "Underexposed": image1,
    "Overexposed": image2,
    "Uneven Illumination": image3
}

# =========================
# FUNCTIONS
# =========================

def negative(img):
    return 255 - img

def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    return np.array(c * np.log(1 + img), dtype=np.uint8)

def gamma_transform(img, gamma):
    table = np.array([(i / 255.0) ** gamma * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def contrast_stretch_auto(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

def hist_equalization(img):
    return cv2.equalizeHist(img)

def clahe(img):
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe_obj.apply(img)

# =========================
# PROCESS
# =========================

for name, img in images.items():

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = [
        ("Original", gray),
        ("Negative", negative(gray)),
        ("Log", log_transform(gray)),
        ("Gamma 0.5", gamma_transform(gray, 0.5)),
        ("Gamma 1.5", gamma_transform(gray, 1.5)),
        ("Gamma 2.5", gamma_transform(gray, 2.5)),
        ("Stretch Auto", contrast_stretch_auto(gray)),
        ("Hist Eq", hist_equalization(gray)),
        ("CLAHE", clahe(gray))
    ]

    # =============================
    # DISPLAY IMAGES (RAPI)
    # =============================

    plt.figure(figsize=(18, 12))
    plt.suptitle("Enhancement Results - " + name, fontsize=18)

    for i, (title, image) in enumerate(results):
        plt.subplot(3, 3, i+1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis("off")

    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()

    # =============================
    # DISPLAY HISTOGRAMS (TERPISAH)
    # =============================

    plt.figure(figsize=(18, 12))
    plt.suptitle("Histogram Results - " + name, fontsize=18)

    for i, (title, image) in enumerate(results):
        plt.subplot(3, 3, i+1)
        plt.hist(image.ravel(), 256, [0,256])
        plt.title(title)

    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()