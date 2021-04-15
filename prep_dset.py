import argparse  # https://stackoverflow.com/a/30493366
import sys
import os
from kaldiio import ReadHelper

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser(description="Align original/anon scp files")

# Required positional argument
parser.add_argument("x_vector_u", type=str, help="The 'U' scp file to x-vectors")

# Required positional argument
parser.add_argument("x_vector_l", type=str, help="The 'L' scp file to x-vectors")

# Required positional argument
parser.add_argument("x_vector_u_out", type=str, help="Output of 'U' data prep x-vector")

# Required positional argument
parser.add_argument(
    "x_vector_u_out_label", type=str, help="Output of 'U' data prep label"
)

# Required positional argument
parser.add_argument("x_vector_l_out", type=str, help="Output of 'U' data prep x-vector")

# Required positional argument
parser.add_argument(
    "x_vector_l_out_label", type=str, help="Output of 'U' data prep label"
)

# Optional argument
parser.add_argument(
    "--test", action="store_true", help="Test if the scp files can be read"
)

# Optional argument
parser.add_argument(
    "--noplot", default=True, action="store_false", help="Don't display the plot"
)

# Optional argument
parser.add_argument(
    "--filter_scp_trials_enrolls",
    action="store_true",
    help="All 'Unlabeled' x-vector speaker are present in the 'Label' scp",
)

args = parser.parse_args()

x_vector_u = {}
with ReadHelper(f"scp:{args.x_vector_u}") as reader:
    for key, numpy_array in reader:
        x_vector_u[key] = numpy_array

x_vector_l = {}
with ReadHelper(f"scp:{args.x_vector_l}") as reader:
    for key, numpy_array in reader:
        x_vector_l[key] = numpy_array

if args.test:
    sys.exit(0)


if args.filter_scp_trials_enrolls:
    with open(
        os.path.dirname(args.x_vector_l)
        + os.path.sep
        + "meta"
        + os.path.sep
        + "spk2gender"
    ) as f:
        spklist = f.read().splitlines()
        spklist = list(map(lambda x: x.split(" ")[0], spklist))

    print("spklist:", spklist)

    print("x_vector_l samples:", len(x_vector_l))
    print("x_vector_u samples:", len(x_vector_u))
    for k, m in x_vector_u.copy().items():
        if k.split("-")[0] not in spklist:
            x_vector_u.pop(k)
    print("x_vector_u samples after filtering:", len(x_vector_u))

# Convert from dictionaries to numpy arrays
u_out, u_out_label = (
    np.array([x_vector_u[i] for i in x_vector_u]),
    np.array([i.split("-")[0] for i in x_vector_u]).astype(int),
)
print(u_out.shape, u_out_label.shape, set(u_out_label))
l_out, l_out_label = (
    np.array([x_vector_l[i] for i in x_vector_l]),
    np.array([i.split("-")[0] for i in x_vector_l]).astype(int),
)
print(l_out.shape, l_out_label.shape, set(l_out_label))
# Convert users id to 0-N numbers
id_tr, id_en = {x: i for i, x in enumerate(list(set(l_out_label)))}, {
    x: i for i, x in enumerate(list(set(u_out_label)))
}
u_out_label, l_out_label = (
    np.array([id_en[i] for i in u_out_label]).astype(int),
    np.array([id_tr[i] for i in l_out_label]).astype(int),
)

np.save(args.x_vector_u_out, u_out)
np.save(args.x_vector_l_out, l_out)
np.save(args.x_vector_u_out_label, u_out_label)
np.save(args.x_vector_l_out_label, l_out_label)

if args.noplot:
    print("Plotting using PCA")
    pca = PCA(n_components=30).fit(u_out)
    new_u_out = pca.transform(u_out)
    new_u_out = normalize(new_u_out)
    print(
        new_u_out.shape,
        "total explained variance ratio :",
        np.sum(pca.explained_variance_ratio_),
    )
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    for u in set(u_out_label):
        ax1.scatter(new_u_out[u_out_label == u, 1], new_u_out[u_out_label == u, 2])

    pca = PCA(n_components=30).fit(l_out)
    new_l_out = pca.transform(l_out)
    new_l_out = normalize(new_l_out)
    print(
        new_l_out.shape,
        "total explained variance ratio :",
        np.sum(pca.explained_variance_ratio_),
    )
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    for u in set(l_out_label):
        ax2.scatter(new_l_out[l_out_label == u, 1], new_l_out[l_out_label == u, 2])
    plt.show()
