import argparse # https://stackoverflow.com/a/30493366
import sys
import os
from kaldiio import ReadHelper

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
