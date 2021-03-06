import os
import argparse
import kaldiio
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve
import numpy as np

from get_align_procrustes import frontend, parse_arguments, top1


def compute_eer(y_pred, y):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[idx] + fnr[idx]) / 2
    thre = threshold[idx]
    if eer > 0.50:
        eer = 0.50 - (eer - 0.500)
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
        utt2embd_enroll = {utt:embd for utt, embd in reader}

        l_out, l_out_label = (
            np.array([utt2embd_enroll[i] for i in utt2embd_enroll]),
            np.array([i for i in utt2embd_enroll]),
        )

        l_out, l_out_label, _, _  = frontend(args, l_out, l_out_label, np.zeros((512,512)), np.zeros((512,)))

        utt2embd_enroll = {utt:embd for utt, embd in zip(l_out_label, l_out)}


    with kaldiio.ReadHelper(f'scp:{args.trial_scp_dir}/{args.trial_scp}') as reader:
        utt2embd_trial = {utt:embd for utt, embd in reader}

        u_out, u_out_label = (
            np.array([utt2embd_trial[i] for i in utt2embd_trial]),
            np.array([i for i in utt2embd_trial]),
        )

        utt2embd_trial = {utt:embd for utt, embd in zip(u_out_label, u_out)}


    utt2embd_enroll = [utt2embd_enroll[utt] for utt in utt1s]
    utt2embd_trial = [utt2embd_trial[utt] for utt in utt2s]

    scores = cosine_scoring(utt2embd_enroll, utt2embd_trial)
    score_file_kaldi = []
    for enroll, trial, score in zip(utt1s, utt2s, scores):
        score_file_kaldi.append([enroll, trial, str(score)])

    with open(args.output, "w") as txt_file:
        for line in score_file_kaldi:
            txt_file.write(" ".join(line) + "\n") # works with any number of elements in a line


    if target is not None:
        eer, threshold = compute_eer(scores, target)
        print("ROC_EER: {:.2f}".format(eer * 100))
        #  print("Threshold: {:.2f}".format(threshold))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speaker Verification Trials Validation.')

    # Kaldi trials files
    parser.add_argument('trials')

    #  Average the utterance-level xvectors to get speaker-level xvectors.
    #  https://github.com/kaldi-asr/kaldi/blob/5caf2c0ae46f908e2d97b5b905fd8240ca5ccc9f/egs/sre08/v1/sid/nnet3/xvector/extract_xvectors.sh#L99-L105
    parser.add_argument('trial_scp_dir') # de-anonymized x-vector
    parser.add_argument('enroll_scp_dir') # original x-vector

    parser.add_argument('output')
    parser.add_argument(
        "--enroll-scp", default="spk_xvector.scp", type=str
    )
    parser.add_argument(
        "--trial-scp", default="xvector.scp", type=str
    )

    args = parse_arguments(parser)

    assert os.path.isfile(args.trials), "NO SUCH FILE: %s" % args.trials
    assert os.path.isdir(args.enroll_scp_dir), "NO SUCH DIRECTORY: %s" % args.enroll_scp_dir
    assert os.path.isdir(args.trial_scp_dir), "NO SUCH DIRECTORY: %s" % args.trial_scp_dir
    main(args)

