import numpy as np

def factorize(V, W, H, epsilon=0.001, max_iter=200):
    H_old = H + 42
    W_old = W + 42

    i = 0
    while ((np.sum(np.abs(H - H_old)) > epsilon) or (np.sum(np.abs(W - W_old)) > epsilon)) and i < max_iter:
        H_old = H
        H = H * ((W.transpose().dot(V)) / (W.transpose().dot(W).dot(H) + epsilon))
        W_old = W
        W = W * ((V.dot(H.transpose())) / (W.dot(H).dot(H.transpose()) + epsilon))
        i += 1

    return (W, H)
