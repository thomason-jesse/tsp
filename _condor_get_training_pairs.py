#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import os
import pickle
import time


def main():

    # Load parameters from command line.
    parser_infile = FLAGS_parser_infile
    pairs_infile = FLAGS_pairs_infile
    outfile = FLAGS_outfile

    # Launch jobs.
    with open(pairs_infile, 'rb') as f:
        d = pickle.load(f)
    jobs_remaining = []
    for idx in range(len(d)):
        cmd = ("condorify_gpu_email " +
               "python _condor_get_parse_pair.py" +
               " --parser_infile " + parser_infile +
               " --pairs_infile " + pairs_infile +
               " --pair_idx " + str(idx) +
               " --outfile temp.pair." + str(idx) + ".pickle" +
               " temp.pair." + str(idx) + ".log")
        os.system(cmd)
        jobs_remaining.append(idx)

    # Collect jobs.
    t = []
    num_trainable = 0
    num_matches = 0
    num_fails = 0
    num_genlex_only = 0
    while len(jobs_remaining) > 0:
        time.sleep(10)  # poll every 60 seconds for finished parses

        newly_completed = []
        for idx in jobs_remaining:
            fn = "temp.pair." + str(idx) + ".pickle"
            try:
                with open(fn, 'rb') as f:
                    result = pickle.load(f)

                    num_trainable += result['num_trainable']
                    num_matches += result['num_matches']
                    num_fails += result['num_fails']
                    num_genlex_only += result['num_genlex_only']
                    if result['num_fails'] == 0:
                        t.append([result['x'], result['chosen_parse'], result['correct_parse'], result['chosen_new_lexicon_entries'],
                                  result['correct_new_lexicon_entries'], result['chosen_skipped_surface_forms'],
                                  result['correct_skipped_surface_forms']])

                    newly_completed.append(idx)

                os.system("rm " + fn)  # remove output file
                os.system("rm *temp.pair." + str(idx) + ".log")  # remove err and log files

            except (IOError, ValueError):
                pass

        now_remaining = [idx for idx in jobs_remaining if idx not in newly_completed]
        if len(newly_completed) > 0:
            print ("_condor_get_training_pairs: completed " + str(len(newly_completed)) +
                   " more jobs (" + str(len(d) - len(now_remaining)) + "/" + str(len(d)) + ")")
        jobs_remaining = now_remaining[:]

    # Output results.
    with open(outfile, 'wb') as f:
        pickle.dump([t, num_trainable, num_matches, num_fails, num_genlex_only], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parser_infile', type=str, required=True,
                        help="the parser pickle")
    parser.add_argument('--pairs_infile', type=str, required=True,
                        help="the pairs pickle")
    parser.add_argument('--outfile', type=str, required=True,
                        help="where to dump the pairs and epoch data")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
