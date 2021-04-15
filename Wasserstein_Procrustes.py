import numpy as np
import torch
import ot

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from tqdm import tqdm
import time
import argparse

import warnings

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Wasserstein Procrustes for Embedding Alignment"
    )
    parser.add_argument("--emb_src", type=str, help="Path to source word embeddings")
    parser.add_argument("--emb_tgt", type=str, help="Path to target word embeddings")
    parser.add_argument("--label_src", type=str, help="Path to source word embeddings")
    parser.add_argument("--label_tgt", type=str, help="Path to target word embeddings")

    parser.add_argument(
        "--seed", default=1111, type=int, help="Random number generator seed"
    )
    parser.add_argument("--nepoch", default=15, type=int, help="Number of epochs")
    parser.add_argument(
        "--niter", default=1024, type=int, help="Initial number of iterations"
    )
    parser.add_argument("--bsz", default=50, type=int, help="Initial batch size")
    parser.add_argument("--kmeans", action="store_true", help="apply kmeans first")
    parser.add_argument(
        "--kmeans_num_cluster", default=-1, type=int, help="Number of cluster"
    )
    parser.add_argument("--lr", default=10, type=float, help="Learning rate")
    parser.add_argument(
        "--nmax",
        default=-1,
        type=int,
        help="Max number of alignment points used",
    )
    parser.add_argument(
        "--reg", default=0.05, type=float, help="Regularization parameter for sinkhorn"
    )
    args = parser.parse_args()
    return args


def objective(X, Y, R, n=1000):
    if n > len(X):
        n = len(X)
    Xn, Yn = X[:n], Y[:n]
    C = -np.dot(np.dot(Xn, R), Yn.T)
    P = ot.sinkhorn(np.ones(n), np.ones(n), C, 0.025, stopThr=1e-3)
    return 1000 * np.linalg.norm(np.dot(Xn, R) - np.dot(P, Yn)) / n


def sqrt_eig(x):
    U, s, VT = np.linalg.svd(x, full_matrices=False)
    return np.dot(U, np.dot(np.diag(np.sqrt(s)), VT))


def align(
    X,
    Y,
    R,
    lr=10.0,
    bsz=200,
    nepoch=5,
    niter=1000,
    corres=None,
    nmax=10000,
    reg=0.05,
    verbose=True,
    last_iter=False,
):
    t0 = time.time()

    for epoch in range(1, nepoch + 1):
        for _it in (
            tqdm(range(1, niter + 1), desc="alignment n°" + str(epoch))
            if verbose
            else range(1, niter + 1)
        ):
            # sample mini-batch
            xt = X[np.random.permutation(nmax)[:bsz], :]
            yt = Y[np.random.permutation(nmax)[:bsz], :]
            # compute OT on minibatch
            C = -np.dot(np.dot(xt, R), yt.T)
            P = ot.sinkhorn(np.ones(bsz), np.ones(bsz), C, reg, stopThr=1e-3)
            # compute gradient
            G = -np.dot(xt.T, np.dot(P, yt))
            R -= lr / bsz * G
            # project on orthogonal matrices
            U, s, VT = np.linalg.svd(R)
            R = np.dot(U, VT)

        bsz *= 2
        bsz = min(bsz, min(len(X), len(Y)))
        niter //= 2
        # niter = max(niter,50)
        if verbose:
            t = int(time.time() - t0)
            print(
                "epoch: %d  obj: %.3f  time: %d %d \t"
                % (epoch, objective(X, Y, R), t // 60, t % 60),
                bsz,
                niter,
            )
            print(
                np.mean(
                    [
                        np.linalg.norm(Emb_L[i] - np.dot(Emb_U[corres[i]], R))
                        for i in range(len(Emb_U))
                    ]
                )
            )
        if niter == 0 or ((not last_iter) and bsz >= min(len(X), len(Y))):
            break
    if verbose:
        print("Alignment Done")
    return R


def convex_init(X, Y, niter=100, reg=0.05, apply_sqrt=False):
    n, d = X.shape
    if apply_sqrt:
        X, Y = sqrt_eig(X), sqrt_eig(Y)
    K_X, K_Y = np.dot(X, X.T), np.dot(Y, Y.T)
    K_Y *= np.linalg.norm(K_X) / np.linalg.norm(K_Y)
    K2_X, K2_Y = np.dot(K_X, K_X), np.dot(K_Y, K_Y)
    P = np.ones([n, n]) / float(n)
    for it in range(1, niter + 1):
        G = np.dot(P, K2_X) + np.dot(K2_Y, P) - 2 * np.dot(K_Y, np.dot(P, K_X))
        q = ot.sinkhorn(np.ones(n), np.ones(n), G, reg, stopThr=1e-3)
        alpha = 2.0 / float(2.0 + it)
        P = alpha * q + (1.0 - alpha) * P
    #  obj = np.linalg.norm(np.dot(P, K_X) - np.dot(K_Y, P))
    # print(obj)
    return procrustes(np.dot(P, X), Y).T


def procrustes(X_src, Y_tgt):
    U, s, V = np.linalg.svd(np.dot(Y_tgt.T, X_src))
    return np.dot(U, V)


def compute_nn_accuracy(X, Y, R, Ux, Uy):
    size = min(len(X), len(Y))
    if len(X) != size or len(Y) != size:
        Ux, Uy = Ux[:size], Uy[:size]
        if len(X) != size:
            X = X[:size]
        if len(Y) != size:
            Y = Y[:size]

    C = -np.dot(np.dot(X, R), Y.T)
    P = ot.sinkhorn(np.ones(size), np.ones(size), C, 0.025, stopThr=1e-3)

    Xn, Yn = np.dot(X, R), np.dot(P, Y)

    compute_unit = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_emb = len(Xn)
    L = np.zeros(n_emb).astype(int)
    for i in range(n_emb):
        distances = (
            torch.sum(
                (
                    torch.FloatTensor(Xn[i])
                    .to(compute_unit)
                    .unsqueeze(0)
                    .repeat(n_emb, 1)
                    - torch.FloatTensor(Yn).to(compute_unit)
                )
                ** 2,
                dim=1,
            )
            .cpu()
            .numpy()
        )
        L[i] = np.argmin(distances)
    acc_U, acc_F = np.sum(Uy[L] == Ux), np.sum(L == np.arange(len(L)))

    return acc_U / size, acc_F / size, P, np.array(L).astype(int)


def Stiefel_Manifold(R):
    U, S, Vt = svd(R)
    return np.dot(U, Vt)


def KMeans_reshape(Emb, User, K):
    lab = KMeans(n_clusters=K, init=Emb[np.arange(K)]).fit(Emb).labels_
    nE, nU = np.zeros((K, Emb.shape[1])), np.zeros(K)
    u = 0
    for i in range(K):
        args = np.argwhere(lab == i)[:, 0]
        # print(args)
        if len(args) == 1:
            nE[i] = Emb[args]
            nU[i] = User[args]
            u += 1
        else:
            nE[i] = np.mean(Emb[args], axis=0)
            nU[i] = User[args][0]
    print(u, K, len(Emb))
    return nE, nU


def Wasserstein_Procrustes_Alignment(
    args,
    verbose=False,
    last_iter=False,
    limited=0,
):

    User_U = np.load("exp/enroll_train_uv/User_U.npy")
    User_L = np.load("exp/enroll_train_uv/User_L.npy")
    Emb_U_ = np.load("exp/enroll_train_uv/Emb_U.npy")
    Emb_L_ = np.load("exp/enroll_train_uv/Emb_L.npy")

    # DO normalize
    Emb_U = normalize(Emb_U_)
    Emb_L = normalize(Emb_L_)

    # DO LDA
    Emb_U = LDA().fit_transform(Emb_U_, User_U)
    Emb_L = LDA().fit_transform(Emb_L_, User_L)
    print("Computed LDA", Emb_U.shape, Emb_L.shape)

    # DO PCA
    d = 30
    print("Computing PCA,", d, "dimensions")
    pca = PCA(n_components=d).fit(Emb_U_)
    Emb_U = pca.transform(Emb_U_)
    print(
        Emb_U.shape,
        "total explained variance ratio :",
        np.sum(pca.explained_variance_ratio_),
    )
    pca = PCA(n_components=d).fit(Emb_L_)
    Emb_L = pca.transform(Emb_L_)
    print(
        Emb_L.shape,
        "total explained variance ratio :",
        np.sum(pca.explained_variance_ratio_),
    )
    # END PCA

    idx = np.arange(min(len(User_L), len(User_U)))
    corres = idx

    if corres is None:
        corres = np.arange(User_U)
    if limited != 0:
        User_U, Emb_U = User_U[:limited], Emb_U[:limited]
        User_L, Emb_L = User_L[:limited], Emb_L[:limited]

    if args.kmeans and args.kmeans_num_cluster == -1:
        if len(Emb_U) < len(Emb_L):
            Emb_L, User_L = KMeans_reshape(Emb_L, User_L, len(Emb_U))
        elif len(Emb_U) > len(Emb_L):
            Emb_U, User_U = KMeans_reshape(Emb_U, User_U, len(Emb_L))
    if args.kmeans and args.kmeans_num_cluster != -1:
        Emb_L, User_L = KMeans_reshape(Emb_L, User_L, args.kmeans_num_cluster)
        Emb_U, User_U = KMeans_reshape(Emb_U, User_U, args.kmeans_num_cluster)

    ninit = min(len(User_U), 1000)
    if args.nmax != -1:
        N_pts_used = args.nmax
    else:
        N_pts_used = min(len(Emb_L), len(Emb_U))

    np.random.seed(args.seed)

    x_src = Emb_U
    x_tgt = Emb_L
    R0 = convex_init(
        x_src[np.random.permutation(len(x_src))[:ninit], :],
        x_tgt[np.random.permutation(len(x_tgt))[:ninit], :],
        reg=args.reg,
        apply_sqrt=True,
    )
    # print(np.mean([np.linalg.norm(Emb_L[i] - np.dot(Emb_U[corres[i]],R0)) for i in range(len(Emb_U))]))
    R = align(
        x_src,
        x_tgt,
        R0.copy(),
        bsz=args.bsz,
        lr=args.lr,
        niter=args.niter,
        corres=corres,
        nepoch=args.nepoch,
        reg=args.reg,
        nmax=args.nmax,
        verbose=verbose,
        last_iter=last_iter,
    )

    # comparison bewteen X.R et P.Y
    acc1, accf, P, L = compute_nn_accuracy(
        x_src[corres], x_tgt, R, User_U[corres], User_L
    )
    if verbose:
        print(
            "\nPrecision Users : %.3f Same segments : %.3f\n" % (100 * acc1, 100 * accf)
        )

    R_final = procrustes(x_src[:N_pts_used], (x_tgt[:N_pts_used])[L]).T
    # return acc1, accf
    return Stiefel_Manifold(R_final), acc1, accf


if __name__ == "__main__":

    for niter in [1024]:
        for lr in [50]:
            for bsz in [40]:

                print("niter :", niter, "lr :", lr, "bsz :", bsz)
                WP_R, acc1, acc2 = Wasserstein_Procrustes_Alignment(
                    User_U,
                    Emb_U,
                    User_L,
                    Emb_L,
                    verbose=False,
                    niter=niter,
                    lr=lr,
                    bsz=bsz,
                )
                print("Accuracy :", acc1, acc2)
        # print(acc1)

    Emb_L = np.dot(Emb_L, WP_R)
