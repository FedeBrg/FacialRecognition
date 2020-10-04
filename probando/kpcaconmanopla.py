import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from PIL import Image
import time

"""It helps visualising the portraits from the dataset."""


def plot_portraits(images, titles, h, w, n_row, n_col):
    plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
    for i in range(n_row * n_col -1):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
    plt.show()


dir = 'lfwcrop_grey/nosotros'
celebrity_photos = os.listdir(dir)[0:16]
celebrity_images = [dir + '/' + photo for photo in celebrity_photos]
images = np.array([cv2.resize(plt.imread(image), (64, 64)) for image in celebrity_images], dtype=np.float64)
celebrity_names = [name[:name.find('0') - 1].replace("_", " ") for name in celebrity_photos]
n_samples, h, w = images.shape
#plot_portraits(images, celebrity_names, h, w, n_row=4, n_col=4)


def pca(X, n_pc):
    n_samples, n_features = X.shape
    mean = np.mean(X, axis=0)
    centered_data = X - mean
    U, S, V = np.linalg.svd(centered_data)
    components = V[:n_pc]
    projected = U[:, :n_pc] * S[:n_pc]
    return projected, components, mean, centered_data

def kpca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.
    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_examples, n_features]
    gamma: float
        Tuning parameter of the RBF kernel
    n_components: int
        Number of principal components to return
    Returns
    ------------
    X_pc: {NumPy ndarray}, shape = [n_examples, k_features]
        Projected dataset
    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')
    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)
    # Compute the symmetric kernel matrix.
    K = np.exp(-gamma * mat_sq_dists)
    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    # Collect the top k eigenvectors (projected examples)
    X_pc = np.column_stack([eigvecs[:, i]
                           for i in range(n_components)])
    X=K
    mean = np.mean(X, axis=0)
    centered_data = X - mean
    U, S, V = np.linalg.svd(centered_data)
    components = V[:n_components]
    projected = U[:, :n_components] * S[:n_components]
    return X_pc, components, mean, centered_data


n_components = 10
X = images.reshape(n_samples, h * w)
P, C, M, Y = kpca(X, 10, n_components)
# print(str(C1.size) + " " + str(C1[0].size))
# P, C, M, Y = pca(X, n_components)
# print(str(C.size) + " " + str(C[0].size))
# eigenfaces = C.reshape((n_components, h, w))
# eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
#plot_portraits(eigenfaces, eigenface_titles, h, w, 4, 4)

# plt.imshow(M.reshape(h, w), cmap=plt.cm.gray)


def reconstruction(Y, C, M, h, w, image_index):
    n_samples, n_features = Y.shape
    weights = np.dot(Y, C.T)
    centered_vector = np.dot(weights[image_index, :], C)
    recovered_image = (M + centered_vector).reshape(h, w)
    return recovered_image


# photo = 'lfwcrop_grey/faces/Arnold_Schwarzenegger_0008.pgm'
# image = np.array([plt.imread(photo)], dtype=np.float64)
# plot_portraits(image, ["Test"], h, w, 1, 1)
# image = image.reshape(1, h * w) - np.mean(X, axis=0)
# print("Looking for Arnold Schwarzenegger\n")
# Pa = np.dot(image, C.T)
#
# distance = 100000000  # better would be : +infinity
# closest = None
# idx = -1
# for i in range(P.shape[0]):
#     delta = sum((P[i] - Pa[0]) ** 2)
#     if delta < distance:
#         distance = delta
#         closest = P[i]
#         idx = i
#
# print("Found:" + celebrity_names[idx])
# plot_portraits([images[idx]], ["Recon"], h, w, 1, 1)

def recon_face(image):
    imga = image.reshape(1, 64 * 64) - np.mean(X, axis=0)
    pa = np.dot(imga, C.T)
    distance = 100000000  # better would be : +infinity
    idx = -1
    for i in range(P.shape[0]):
        delta = sum((P[i] - pa[0]) ** 2)
        if delta < distance:
            distance = delta
            idx = i
    return idx


faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # he = y+h-25
        # if he <= y+50:
        #     continue
        # wi = x+w-40
        # if wi <= x+25:
        #     continue
        roi_gray = gray[y:y + h, x:x + w]
        # roi_gray = gray[y+50:y + h-25, x+25:x + w-40]
        roi_color = img[y:y + h, x:x + w]
        cv2.putText(img, celebrity_names[recon_face(cv2.resize(roi_gray, (64, 64)))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # plot_portraits([cv2.resize(roi_gray, (64, 64))], ["Recon"], 64, 64, 1, 1)
        # photo = 'lfwcrop_grey/faces/Abbas_Kiarostami_0001.pgm'
        # image = np.array([plt.imread(photo)], dtype=np.float64)
        # cv2.putText(img, celebrity_names[recon_face(image)], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
