import matplotlib.pyplot as plt
import numpy as np
import os

"""It helps visualising the portraits from the dataset."""


def plot_portraits(images, titles, h, w, n_row, n_col):
    plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
    plt.show()


dir = 'lfwcrop_grey/faces'
celebrity_photos = os.listdir(dir)[1:1001]
celebrity_images = [dir + '/' + photo for photo in celebrity_photos]
images = np.array([plt.imread(image) for image in celebrity_images], dtype=np.float64)
celebrity_names = [name[:name.find('0') - 1].replace("_", " ") for name in celebrity_photos]
n_samples, h, w = images.shape
plot_portraits(images, celebrity_names, h, w, n_row=4, n_col=4)


def pca(X, n_pc):
    n_samples, n_features = X.shape
    mean = np.mean(X, axis=0)
    centered_data = X - mean
    U, S, V = np.linalg.svd(centered_data)
    components = V[:n_pc]
    projected = U[:, :n_pc] * S[:n_pc]
    return projected, components, mean, centered_data


n_components = 300
X = images.reshape(n_samples, h * w)
P, C, M, Y = pca(X, n_pc=n_components)
eigenfaces = C.reshape((n_components, h, w))
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_portraits(eigenfaces, eigenface_titles, h, w, 4, 4)

plt.imshow(M.reshape(h, w), cmap=plt.cm.gray)


def reconstruction(Y, C, M, h, w, image_index):
    n_samples, n_features = Y.shape
    weights = np.dot(Y, C.T)
    centered_vector = np.dot(weights[image_index, :], C)
    recovered_image = (M + centered_vector).reshape(h, w)
    return recovered_image


photo = 'lfwcrop_grey/faces/Arnold_Schwarzenegger_0008.pgm'
image = np.array([plt.imread(photo)], dtype=np.float64)
plot_portraits(image, ["Test"], h, w, 1, 1)
image = image.reshape(1, h * w) - np.mean(X, axis=0)
print("Looking for Arnold Schwarzenegger\n")
Pa = np.dot(image, C.T)

distance = 100000000  # better would be : +infinity
closest = None
idx = -1
for i in range(P.shape[0]):
    delta = sum((P[i] - Pa[0]) ** 2)
    if delta < distance:
        distance = delta
        closest = P[i]
        idx = i

print("Found:" + celebrity_names[idx])
plot_portraits([images[idx]], ["Recon"], h, w, 1, 1)
