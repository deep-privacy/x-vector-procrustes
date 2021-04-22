import argparse
from Wasserstein_Procrustes import frontend
from sklearn.preprocessing import normalize
import kaldiio
import numpy as np

def main(args):
    with kaldiio.ReadHelper(f'scp:{args.emb_src}') as reader:
        x_vector_u = {utt:embd for utt, embd in reader}

    R = np.load(args.rotation)

    # Convert from dictionaries to numpy arrays
    u_out, u_out_label = (
        np.array([x_vector_u[i] for i in x_vector_u]),
        np.array([i for i in x_vector_u]),
    )
    emb, _, _, _ = frontend(args, u_out, u_out_label, np.zeros((512,512)), np.zeros((512,)))

    emb = normalize(emb)

    R_emb = np.dot(emb, R)

    scp_data = {u_out_label[i]: embd for i, embd in enumerate(R_emb)}

    kaldiio.save_ark(f'{args.emb_out}/transformed_xvector.ark', scp_data, scp=f'{args.emb_out}/transformed_xvector.scp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Apply Procrustes on scp x-vector')

    parser.add_argument("--emb_src", type=str, help="Path to source embeddings")
    parser.add_argument("--emb_out", type=str, help="Output Path transformed embeddings")
    parser.add_argument("--rotation", type=str, help="Path to WP rotation")

    # frontend args
    parser.add_argument(
        "--lda", action="store_true", help="apply LDA first and normalize"
    )
    parser.add_argument(
        "--pca", action="store_true", help="apply PCA first and normalize"
    )
    parser.add_argument(
        "--pca_n_dim", default=10, type=int, help="Number of components of the PCA"
    )
    parser.add_argument("--pca_load_path", type=str, help="PCA pickle")
    parser.add_argument("--test", action="store_true", help="testing mode")
    parser.add_argument(
        "--kmeans", action="store_true", help="apply KMeans first otherwise normalize"
    )
    parser.add_argument(
        "--kmeans_num_cluster", default=-1, type=int, help="Number of KMeans cluster"
    )
    parser.add_argument(
        "--seed", default=1111, type=int, help="Random number generator seed"
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    main(args)
