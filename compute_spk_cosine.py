import os
import argparse
import kaldiio
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.preprocessing import normalize


def compute_eer(y_pred, y):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[idx] + fnr[idx]) / 2
    thre = threshold[idx]
    return eer, thre


def cosine_scoring(embd1s, embd2s):
    scores = []
    for embd1, embd2 in zip(embd1s, embd2s):
        # Multiplying by -1 to ensure compatibility with affinity
        # Now lower value will indicate less affinity as compared
        # to original cosine distance
        score = 1 - cosine(embd1, embd2)
        scores.append(score)
    return scores


def main(args):
    trials = [x.split() for x in open(args.trials)]
    utt1s = [x[0] for x in trials]
    utt2s = [x[1] for x in trials]
    if len(trials[0]) == 3:
        tar2int = {'nontarget':0, 'target':1}
        target = [tar2int[x[2]] for x in trials]
    else:
        target = None

    with kaldiio.ReadHelper(f'scp:{args.enroll_scp_dir}/{args.enroll_scp}') as reader:
        utt2embd1s = {utt:embd for utt, embd in reader}

    with kaldiio.ReadHelper(f'scp:{args.trial_scp_dir}/{args.trial_scp}') as reader:
        utt2embd2s = {utt:embd for utt, embd in reader}

    utt2embd1s = [utt2embd1s[utt] for utt in utt1s]
    utt2embd2s = [utt2embd2s[utt] for utt in utt2s]

    scores = cosine_scoring(utt2embd1s, utt2embd2s)
    np.savetxt(args.output, scores, fmt='%.4f')

    if target is not None:
        eer, threshold = compute_eer(scores, target)
        print("EER: {:.2f}%".format(eer * 100))
        #  print("Threshold: {:.2f}".format(threshold))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speaker Verification Trials Validation.')

    # Kaldi trials files
    parser.add_argument('trials')

    #  Average the utterance-level xvectors to get speaker-level xvectors.
    #  https://github.com/kaldi-asr/kaldi/blob/5caf2c0ae46f908e2d97b5b905fd8240ca5ccc9f/egs/sre08/v1/sid/nnet3/xvector/extract_xvectors.sh#L99-L105
    parser.add_argument('trial_scp_dir')
    parser.add_argument('enroll_scp_dir')

    parser.add_argument('output')
    parser.add_argument(
        "--enroll-scp", default="spk_xvector.scp", type=str
    )
    parser.add_argument(
        "--trial-scp", default="xvector.scp", type=str
    )

    args = parser.parse_args()

    assert os.path.isfile(args.trials), "NO SUCH FILE: %s" % args.trials
    assert os.path.isdir(args.enroll_scp_dir), "NO SUCH DIRECTORY: %s" % args.enroll_scp_dir
    assert os.path.isdir(args.trial_scp_dir), "NO SUCH DIRECTORY: %s" % args.trial_scp_dir
    main(args)

