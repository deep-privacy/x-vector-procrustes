import numpy as np
import torch
import ot

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from tqdm import tqdm
import argparse

import warnings

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Wasserstein Procrustes for Embedding Alignment"
    )
    parser.add_argument("--emb_src", type=str, help="Path to source embeddings")
    parser.add_argument("--emb_tgt", type=str, help="Path to target embeddings")
    parser.add_argument("--label_src", type=str, help="Path to source labels")
    parser.add_argument("--label_tgt", type=str, help="Path to target labels")
    parser.add_argument("--rotation", type=str, help="Path to WP rotation")

    parser.add_argument(
        "--seed", default=1111, type=int, help="Random number generator seed"
    )
    parser.add_argument("--nepoch", default=15, type=int, help="Number of epochs")
    parser.add_argument(
        "--niter", default=1024, type=int, help="Initial number of iterations"
    )
    parser.add_argument("--bsz", default=40, type=int, help="Initial batch size")
    parser.add_argument(
        "--lda", action="store_true", help="apply LDA first and normalize"
    )
    parser.add_argument(
        "--pca", action="store_true", help="apply PCA first and normalize"
    )
    parser.add_argument(
        "--test", action="store_true", help="testing mode"
    )
    parser.add_argument(
        "--kmeans", action="store_true", help="apply KMeans first otherwise normalize"
    )
    parser.add_argument(
        "--kmeans_num_cluster", default=-1, type=int, help="Number of KMeans cluster"
    )
    parser.add_argument("--lr", default=50, type=float, help="Learning rate")
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
    lr,
    bsz,
    nepoch,
    niter,
    corres,
    nmax,
    reg,
    verbose,
    last_iter,
    ):
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

        if verbose:
            print(
                "epoch: %d\t batchSize: %d\t niter: %d\t Wass_dist: %.3f\t distance: %.4f"
                % (
                    epoch,
                    bsz,
                    niter,
                    objective(X, Y, R),
                    np.mean(
                        [
                            np.linalg.norm(X[i] - np.dot(Y[corres[i]], R))
                            for i in range(len(Y))
                        ]
                    ),
                ),
            )
        if niter == 0 or ((not last_iter) and bsz >= min(len(X), len(Y))):
            print("Stopping alignment batchSize %d > total labels" % bsz)
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



def frontend(args, Emb_U_, User_U, Emb_L_, User_L):
    # DO LDA
    if args.lda:
        Emb_U = LDA().fit_transform(Emb_U_, User_U)
        Emb_L = LDA().fit_transform(Emb_L_, User_L)
        print("Computed LDA", Emb_U.shape, Emb_L.shape)
        return Emb_U, User_U, Emb_L, User_L

    # DO PCA
    if args.pca:
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
        return Emb_U, User_U, Emb_L, User_L

    # DO Kmeans
    if args.kmeans and args.kmeans_num_cluster == -1:
        if len(Emb_U) < len(Emb_L):
            Emb_L, User_L = KMeans_reshape(Emb_L, User_L, len(Emb_U))
        elif len(Emb_U) > len(Emb_L):
            Emb_U, User_U = KMeans_reshape(Emb_U, User_U, len(Emb_L))
        return Emb_U, User_U, Emb_L, User_L

    if args.kmeans and args.kmeans_num_cluster != -1:
        Emb_L, User_L = KMeans_reshape(Emb_L, User_L, args.kmeans_num_cluster)
        Emb_U, User_U = KMeans_reshape(Emb_U, User_U, args.kmeans_num_cluster)
        return Emb_U, User_U, Emb_L, User_L

    return Emb_U_, User_U, Emb_L_, User_L


def Wasserstein_Procrustes_Alignment(
    args,
    verbose=False,
    last_iter=False,
    ):

    User_U = np.load(args.label_src)
    User_L = np.load(args.label_tgt)
    Emb_U_ = np.load(args.emb_src)
    Emb_L_ = np.load(args.emb_tgt)

    # DO normalize DEFAULT
    Emb_U = normalize(Emb_U_)
    Emb_L = normalize(Emb_L_)

    # Overwrite normalize and apply frontend instead
    Emb_U_, User_U, Emb_L_, User_L = frontend(args, Emb_U_, User_U, Emb_L_, User_L)

    idx = np.arange(min(len(User_L), len(User_U)))
    corres = idx

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
        nmax=N_pts_used,
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

    return Stiefel_Manifold(R_final), acc1, accf

    

if __name__ == "__main__":

    args = parse_arguments()
    if not args.test:
        WP_R, acc1, acc2 = Wasserstein_Procrustes_Alignment(
            args,
            verbose=True,
        )
        print("Accuracy :", acc1, acc2, "for args:", args)
        np.save(args.rotation, WP_R)

    else:
        User_A = np.load(args.label_src)
        User_B = np.load(args.label_tgt)
        Emb_A_ = np.load(args.emb_src)
        Emb_B_ = np.load(args.emb_tgt)
        WP_R = np.load(args.rotation)

        Emb_A = normalize(Emb_A_)
        Emb_B = normalize(Emb_B_)
        Xn, Yn = Emb_A, np.dot(Emb_B, WP_R)
        Ux, Uy = User_A, User_B
        compute_unit = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        n_emb = len(Xn)
        L = np.zeros(n_emb).astype(int)
        for i in range(n_emb):
            distances = torch.sum((torch.FloatTensor(Xn[i]).to(compute_unit).unsqueeze(0).repeat(n_emb,1)-torch.FloatTensor(Yn).to(compute_unit))**2, dim=1).cpu().numpy()
            L[i] = np.argmin(distances)
        acc_U, acc_F = np.sum(Uy[L]==Ux), np.sum(L==np.arange(len(L)))
        print(acc_U/len(Uy), acc_F/len(Ux))
