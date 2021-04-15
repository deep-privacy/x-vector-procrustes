import argparse # https://stackoverflow.com/a/30493366
import sys
import os
from kaldiio import ReadHelper

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='Align original/anon scp files')

# Required positional argument
parser.add_argument('x_vector_enroll', type=str,
                    help='The scp file to read original x-vectors')

# Required positional argument
parser.add_argument('x_vector_trials', type=str,
                    help='The scp file to read anonymized x-vectors')

# Optional argument
parser.add_argument('--test', action='store_true',
                    help='Test if the scp files can be read')

# Optional argument
parser.add_argument('--filter_scp_trials_enrolls', action='store_true',
                    help='Test if the scp files can be read')

args = parser.parse_args()

x_vector_enroll = {}
with ReadHelper(f'scp:{args.x_vector_enroll}') as reader:
    for key, numpy_array in reader:
        x_vector_enroll[key] = numpy_array

x_vector_trials = {}
with ReadHelper(f'scp:{args.x_vector_trials}') as reader:
    for key, numpy_array in reader:
        x_vector_trials[key] = numpy_array

if args.test:
    print("Scp files could be read")
    sys.exit(0)


if args.filter_scp_trials_enrolls:
    with open(os.path.dirname(args.x_vector_trials) + os.path.sep + "meta" + os.path.sep + "spk2gender") as f:
        spklist = f.read().splitlines()
        spklist = list(map(lambda x: x.split(" ")[0], spklist))

    print("spklist:", spklist)

    print("x_vector_trials samples:", len(x_vector_trials))
    print("x_vector_enroll samples:", len(x_vector_enroll))
    for k, m in x_vector_enroll.copy().items():
        if k.split("-")[0] not in spklist:
            x_vector_enroll.pop(k)
    print("x_vector_enroll samples after filtering:", len(x_vector_enroll))

#Convert from dictionnaries to numpy arrays
Emb_enroll, User_enroll = np.array([x_vector_enroll[i] for i in x_vector_enroll]), np.array([i.split('-')[0] for i in x_vector_enroll]).astype(int)
print(Emb_enroll.shape, User_enroll.shape, set(User_enroll))
Emb_trials, User_trials = np.array([x_vector_trials[i] for i in x_vector_trials]), np.array([i.split('-')[0] for i in x_vector_trials]).astype(int)
print(Emb_trials.shape, User_trials.shape, set(User_trials))
#Convert users id to 0-N numbers
id_tr, id_en = {x:i for i,x in enumerate(list(set(User_trials)))}, {x:i for i,x in enumerate(list(set(User_enroll)))}
User_enroll, User_trials = np.array([id_en[i] for i in User_enroll]).astype(int), np.array([id_tr[i] for i in User_trials]).astype(int) 

pca = PCA(n_components = 30).fit(Emb_enroll)
new_Emb_enroll = pca.transform(Emb_enroll)
print(new_Emb_enroll.shape, "total explained variance ratio :", np.sum(pca.explained_variance_ratio_))
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
for u in set(User_enroll):
    ax1.scatter(new_Emb_enroll[User_enroll==u,0], new_Emb_enroll[User_enroll==u,1])

pca = PCA(n_components = 30).fit(Emb_trials)
new_Emb_trials = pca.transform(Emb_trials)
print(new_Emb_trials.shape, "total explained variance ratio :", np.sum(pca.explained_variance_ratio_))
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

for u in set(User_trials):
    ax2.scatter(new_Emb_trials[User_trials==u,0], new_Emb_trials[User_trials==u,1])
plt.show()

np.save("numpy_arrays/Emb_A",Emb_enroll)
np.save("numpy_arrays/Emb_B",Emb_trials)
np.save("numpy_arrays/User_A",User_enroll)
np.save("numpy_arrays/User_B",User_trials)