import argparse
from get_align_procrustes import frontend, parse_arguments
import kaldiio
import numpy as np

def main(args):
    with kaldiio.ReadHelper(f'scp:{args.emb_in}') as reader:
        x_vector_u = {utt:embd for utt, embd in reader}

    R = np.load(args.rotation)

    # Convert from dictionaries to numpy arrays
    u_out, u_out_label = (
        np.array([x_vector_u[i] for i in x_vector_u]),
        np.array([i for i in x_vector_u]),
    )
    _, _, emb, emb_label = frontend(args, np.zeros((512,512)), np.zeros((512,)), u_out, u_out_label)

    R_emb = np.dot(emb, R)

    scp_data = {utt:embd for utt, embd in zip(emb_label, R_emb)}

    kaldiio.save_ark(f'{args.emb_out}/transformed_xvector.ark', scp_data, scp=f'{args.emb_out}/transformed_xvector.scp')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Apply Procrustes on scp x-vector')

    parser.add_argument("--emb_in", type=str, help="Path to source embeddings")
    parser.add_argument("--emb_out", type=str, help="Output Path transformed embeddings")
    #  parser.add_argument("--rotation", type=str, help="Path to WP rotation")

    args = parse_arguments(parser)

    np.random.seed(args.seed)

    main(args)
