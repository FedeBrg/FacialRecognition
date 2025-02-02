import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from scipy.spatial.distance import pdist, squareform
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


dir = 'lfwcrop_grey/our_faces'
celebrity_photos = os.listdir(dir)
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
    # Calculating the squared Euclidean distances for every pair of points
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Converting the pairwise distances into a symmetric MxM matrix.
    mat_sq_dists = squareform(sq_dists)

    # Computing the MxM kernel matrix.
    K = np.exp(-gamma * mat_sq_dists)

    # Centering the symmetric NxN kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K_norm = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    # eigvals, eigvecs = eigh(K_norm)
    U, S, V = np.linalg.svd(K_norm)

    # Obtaining the i eigenvectors (alphas) that corresponds to the i highest eigenvalues (lambdas).
    # alphas2 = np.column_stack((eigvecs[:,-i] for i in range(1, n_components + 1)))
    # lambdas2 = [eigvals[-i] for i in range(1, n_components+1)]

    for i in range(0, len(U)):
        U[i] = np.flip(U[i])

    alphas = np.column_stack((U[:, -i] for i in range(1, n_components + 1)))
    lambdas = [S[i] for i in range(1, n_components + 1)]


    X_pc = np.column_stack((U[:, -i] for i in range(1, n_components + 1)))

    return X_pc, alphas, lambdas


def calculategamma(X):
    min_distance = np.empty(len(X))
    filler = float("inf")
    index = np.arange(min_distance.size)
    np.put(min_distance, index, filler)
    for i in range(0, len(X)):
        for j in range(0, len(X)):
            if j != i:
                accum = 0
                for k in range(0, len(X[i])):
                    accum = accum + ((X[i][k]-X[j][k]) ** 2) ** (1/2)
                if min_distance[i] > accum:
                    min_distance[i] = accum
    return 5 * np.mean(min_distance)



n_components = 10
X = images.reshape(n_samples, h * w)
# P, C, M, Y = pca(X, n_pc=n_components)
# eigenfaces = C.reshape((n_components, h, w))
# eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
# plot_portraits(eigenfaces, eigenface_titles, h, w, 4, 4)
# plt.imshow(M.reshape(h, w), cmap=plt.cm.gray)

# gamma = 1/(2*22546**2)
# print(gamma)
gamma = 1/(2*(calculategamma(X)**2))
# print  (gamma)
X_pc, alphas, lambdas = kpca(X, gamma=gamma, n_components=n_components)

# def reconstruction(Y, C, M, h, w, image_index):
#     n_samples, n_features = Y.shape
#     weights = np.dot(Y, C.T)
#     centered_vector = np.dot(weights[image_index, :], C)
#     recovered_image = (M + centered_vector).reshape(h, w)
#     return recovered_image


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

# def recon_face(image):
#     imga = image.reshape(1, 64 * 64) - np.mean(X, axis=0)
#     pa = np.dot(imga, C.T)
#     distance = 100000000  # better would be : +infinity
#     idx = -1
#     for i in range(P.shape[0]):
#         delta = sum((P[i] - pa[0]) ** 2)
#         if delta < distance:
#             distance = delta
#             idx = i
#     return idx

def project_x(x_new, X, gamma, alphas, lambdas):
    x_new = x_new.reshape(1, 64 * 64)
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

def recon_face_kpca(x_new, X, gamma, alphas, lambdas):
    x_new = project_x(x_new, X, gamma, alphas, lambdas)
    distance = float("inf")  # better would be : +infinity
    idx = -1
    for i in range(X.shape[0]):
        delta = sum((X_pc[i] - x_new) ** 2)
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
        # cv2.putText(img, celebrity_names[recon_face(cv2.resize(roi_gray, (64, 64)))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        x_new = cv2.resize(roi_gray, (64, 64))
        cv2.putText(img, celebrity_names[recon_face_kpca(x_new, X, gamma=gamma, alphas=alphas, lambdas=lambdas)], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
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


# xdata = np.empty(len(X_pc), dtype=float)
# ydata = np.empty(len(X_pc), dtype=float)
# zdata = np.empty(len(X_pc), dtype=float)
#
# ax = plt.axes(projection='3d')
# for i in range(0, len(X_pc)):
#     xdata[i] = X_pc[i][0]
#     ydata[i] = X_pc[i][1]
#     zdata[i] = X_pc[i][2]
# ax.scatter3D(xdata, ydata, zdata, c=zdata)
# for i in range(0, len(X_pc)):
#     ax.text(xdata[i], ydata[i], zdata[i], celebrity_names[i], size=7, zorder=1, color='k')
# plt.show()

