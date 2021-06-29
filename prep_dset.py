import argparse  # https://stackoverflow.com/a/30493366
import sys
import os
from kaldiio import ReadHelper
import itertools
from tqdm.contrib import tzip

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
    "--test-remove-train-spk", default=False, action="store_true", help="remove spk that are in train and test"
)

# Optional argument
parser.add_argument(
    "--test-only-train-spk", default=False, action="store_true", help="remove spk that are not in train and test"
)

# Optional argument
parser.add_argument(
    "--filter_scp_trials_enrolls",
    action="store_true",
    help="All 'Unlabeled' x-vector speaker are present in the 'Label' scp",
)

# Optional argument
parser.add_argument(
    "--spk_utt_all_combinations",
    action="store_true",
    help="Get all possible combinations of a speaker to segments pairs (instead of utt -> utt_anon)",
)

# Optional argument
parser.add_argument(
    "--filter_gender",
    type=str,
    nargs='?',
    default="",
    help="Filter by gender 'f' or 'm'",
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

if args.filter_gender != "" and args.filter_gender:
    with open(
        os.path.dirname(args.x_vector_l)
        + os.path.sep
        + "meta"
        + os.path.sep
        + "spk2gender"
    ) as f:
        spk2gender = f.read().splitlines()
        spk2gender = {spk:gender for spk, gender in list(map(lambda x: x.split(" "), spk2gender))}
    print(f"Filtering by gender {args.filter_gender}")
    print("x_vector_l samples:", len(x_vector_l))
    print("x_vector_u samples:", len(x_vector_u))
    for k, m in x_vector_u.copy().items():
        spk = k.split("-")[0]
        if spk2gender[spk] != args.filter_gender:
            x_vector_u.pop(k)
    for k, m in x_vector_l.copy().items():
        spk = k.split("-")[0]
        if spk2gender[spk] != args.filter_gender:
            x_vector_l.pop(k)
    print("x_vector_l samples after filtering:", len(x_vector_l))
    print("x_vector_u samples after filtering:", len(x_vector_u))


# Convert from dictionaries to numpy arrays
u_out, u_out_label = (
    np.array([x_vector_u[i] for i in x_vector_u]),
    np.array([i.split("-")[0] for i in x_vector_u]).astype(int),
)
#  print(u_out.shape, u_out_label.shape, set(u_out_label))
l_out, l_out_label = (
    np.array([x_vector_l[i] for i in x_vector_l]),
    np.array([i.split("-")[0] for i in x_vector_l]).astype(int),
)

#  print(l_out.shape, l_out_label.shape, set(l_out_label))
# Convert users id to 0-N numbers
id_tr, id_en = {x: i for i, x in enumerate(list(set(l_out_label)))}, {
    x: i for i, x in enumerate(list(set(u_out_label)))
}

if args.test_remove_train_spk or args.test_only_train_spk: # remove/filter spk that are in train and test
    filename = "./data/x_vector_vpc__crossgender=false__f0transformation=false__diffpseudospeaker/xvect_libri_test_enrolls/meta/spk2gender"
    with open(filename) as f:
        content = f.read().splitlines()
    filter_remove_spk = [a.split(" ")[0] for a in content]

    u_out_t = []
    u_out_label_t = np.array([])
    for i in range(len(u_out_label)):
        label = u_out_label[i]
        data = u_out[i]
        if args.test_remove_train_spk and str(label) not in filter_remove_spk or \
        args.test_only_train_spk and str(label) in filter_remove_spk:
            u_out_label_t = np.append(u_out_label_t, label)
            u_out_t.append(data)

    u_out_t = np.array(u_out_t)
    u_out_label = u_out_label_t
    u_out = u_out_t

    l_out_t = []
    l_out_label_t = np.array([])
    for i in range(len(l_out_label)):
        label = l_out_label[i]
        data = l_out[i]
        if args.test_remove_train_spk and str(label) not in filter_remove_spk or \
        args.test_only_train_spk and str(label) in filter_remove_spk:
            l_out_label_t = np.append(l_out_label_t, label)
            l_out_t.append(data)

    l_out_t = np.array(l_out_t)
    l_out_label = l_out_label_t
    l_out = l_out_t



u_out_label, l_out_label = (
    np.array([id_en[i] for i in u_out_label]).astype(int),
    np.array([id_tr[i] for i in l_out_label]).astype(int),
)


if args.spk_utt_all_combinations:

    all_combination = list(itertools.product(u_out, l_out))
    all_combination_label = list(itertools.product(u_out_label, l_out_label))

    u_out = []
    l_out = []

    u_out_label = np.array([])
    l_out_label = np.array([])

    for (u, l),(u_label, l_label) in tzip(all_combination,all_combination_label):
        if u_label != l_label:
            continue
        if len(u_out) == 0:
            u_out = np.array([u])
            l_out = np.array([l])
        else:
            u_out = np.append(u_out, [u], axis=0)
            l_out = np.append(l_out, [l], axis=0)

        u_out_label = np.append(u_out_label, u_label)
        l_out_label = np.append(l_out_label, l_label)

    print("x_vector_l samples after all_combination:", len(l_out))
    print("x_vector_u samples after all_combination:", len(u_out))


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
