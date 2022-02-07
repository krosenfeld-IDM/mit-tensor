import scipy as sp
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from itertools import permutations

def nmf(X, r, niter=30):
    m, n = X.shape
    U = np.random.rand(m, r)
    V = np.random.rand(n, r)
    for i in range(niter):
        # print('{}/{}'.format(i, niter))
        # iterate over V
        for iv in range(V.shape[0]):
            V[iv, :], _ = sp.optimize.nnls(U, X[:, iv])
        # now repeat for U
        for iu in range(U.shape[0]):
            U[iu, :], _ = sp.optimize.nnls(V, X[iu, :])
    return U, V

#############################################

# X = io.loadmat('tensordata.mat')['X']
X = np.loadtxt('nmfdata.txt')
print('Data shape:', X.shape)

# Scree plot
r = [2, 5, 10, 20]
num_samples = 5
nmf_model_err = np.zeros(shape=(num_samples, len(r)))
for ir, r_ in enumerate(r):
    for isamp in range(num_samples):
        U, V = nmf(X, r_)
        nmf_model_err[isamp, ir] = np.sqrt(np.mean(np.abs(X - np.matmul(U, V.T)) ** 2))

plt.figure()
plt.plot(r, nmf_model_err.T, 'b.')

# Truncated SVD/PCA
svd_model_err = np.zeros(shape=(num_samples, len(r)))
for ir, r_ in enumerate(r):
    fit = TruncatedSVD(n_components=r_, random_state=isamp).fit(X)
    X_est = fit.inverse_transform(fit.transform(X))
    svd_model_err[isamp, ir] = np.sqrt(np.mean((X - X_est)**2))

plt.plot(r, svd_model_err.T, 'g.')

plt.plot([], [], 'b.', label='NMF')
plt.plot([], [], 'g.', label='SVD')

plt.legend(frameon=False)
plt.ylim(0, None)
plt.xlabel('rank (r)')
plt.ylabel('Model RMSE')

#############################################

# Similarity plot
def matrix_similarity(U, Up):
    # n x r matrix

    # normalize U and Up
    U = U/np.sqrt(np.sum(U**2, axis=0)[np.newaxis, :])
    Up = Up/np.sqrt(np.sum(Up**2, axis=0)[np.newaxis, :])
    assert(np.all(np.sum(U**2)) == 1)
    assert(np.all(np.sum(Up**2)) == 1)

    # number of components
    r = U.shape[1]

    score = 0
    # iterate over all permutations of Up column order
    for p in permutations(range(r)):
        score = max(score, np.trace(np.matmul(U.T, Up[:, list(p)])))
    return score/r

# Similarity plot
r = [2, 3, 4]
num_samples = 5
U_similarity = np.zeros(shape=(num_samples, len(r)))
V_similarity = np.zeros(shape=(num_samples, len(r)))

for ir, r_ in enumerate(r):
    for isamp in range(num_samples):
        U0, V0 = nmf(X, r_)
        U1, V1 = nmf(X, r_)

        U_similarity[isamp, ir] = matrix_similarity(U0, U1)
        V_similarity[isamp, ir] = matrix_similarity(V0, V1)

plt.figure()
plt.plot(r, np.mean(U_similarity, axis=0), '-ko', label='U')
plt.plot(r, np.mean(V_similarity, axis=0), '-bo', label='V')
plt.ylabel('Matrix similarity')
plt.xlabel('Number of components')
plt.legend(frameon=False)

#############################################

# Application
r = 5
# NMF model
U, V = nmf(X, r)
# X \simeq U*Transpose(V)
print(U.shape, V.shape)
print(X.shape)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Raw data [Samples x Genes]
ax = axes[0]
im = ax.imshow(X)
plt.colorbar(im, ax=ax)
ax.set_title('Raw data')

# Cluster columns (e.g. Genes)
# identify max
cluster_id = np.argmax(V.T, axis=0)
print('Cluster ID shape:', cluster_id.shape)
# sort me
sort_idx = np.argsort(cluster_id)
# plot
ax = axes[1]
im = ax.imshow(X[:, sort_idx])
plt.colorbar(im, ax=ax)
ax.set_title('Clustered columns')

# Cluster rows (e.g. samples)
# identify max
cluster_id = np.argmax(U, axis=1)
print('Cluster ID shape:', cluster_id.shape)
# sort me
sort_idx = np.argsort(cluster_id)
# plot
ax = axes[2]
im = ax.imshow(X[sort_idx, :])
plt.colorbar(im, ax=ax)
ax.set_title('Clustered rows')

plt.show()
