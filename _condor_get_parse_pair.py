#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import pickle


def main():

    # Load parameters from command line.
    parser_infile = FLAGS_parser_infile
    pairs_infile = FLAGS_pairs_infile
    pair_idx = FLAGS_pair_idx
    outfile = FLAGS_outfile
    x = FLAGS_x
    y = FLAGS_y

    # Load the parser and prepare the pair.
    with open(parser_infile, 'rb') as f:
        p = pickle.load(f)
    if x is not None and y is not None:
        ccg_str, form_str = y.split(" : ")
        ccg = p.lexicon.read_category_from_str(ccg_str)
        y = p.lexicon.read_semantic_form_from_str(form_str, None, None, [])
        y.category = ccg
    else:
        with open(pairs_infile, 'rb') as f:
            pairs = pickle.load(f)
        x, y = pairs[pair_idx]

    # Parse.
    num_trainable = 0
    num_matches = 0
    num_fails = 0
    num_genlex_only = 0

    correct_parse = None
    correct_new_lexicon_entries = []
    cky_parse_generator = p.most_likely_cky_parse(x, reranker_beam=1, known_root=y,
                                                  reverse_fa_beam=p.training_reverse_fa_beam,
                                                  debug=False)
    chosen_parse, chosen_score, chosen_new_lexicon_entries, chosen_skipped_surface_forms = \
        next(cky_parse_generator)
    current_parse = chosen_parse
    correct_score = chosen_score
    current_new_lexicon_entries = chosen_new_lexicon_entries
    current_skipped_surface_forms = chosen_skipped_surface_forms
    correct_skipped_surface_forms = None
    match = False
    first = True
    if chosen_parse is None:
        print "WARNING: could not find valid parse for '" + x + "' during training"  # DEBUG
        num_fails += 1
    else:
        while correct_parse is None and current_parse is not None:
            if y.equal_allowing_commutativity(current_parse.node, p.ontology):
                correct_parse = current_parse
                correct_new_lexicon_entries = current_new_lexicon_entries
                correct_skipped_surface_forms = current_skipped_surface_forms
                if first:
                    match = True
                    num_matches += 1
                else:
                    num_trainable += 1
                break
            first = False
            current_parse, correct_score, current_new_lexicon_entries, current_skipped_surface_forms = \
                next(cky_parse_generator)
        if correct_parse is None:
            print "WARNING: could not find correct parse for '"+str(x)+"' during training"
            num_fails += 1
        else:
            print "\tx: "+str(x)  # DEBUG
            print "\t\tchosen_parse: "+p.print_parse(chosen_parse.node, show_category=True)  # DEBUG
            print "\t\tchosen_score: "+str(chosen_score)  # DEBUG
            print "\t\tchosen_skips: "+str(chosen_skipped_surface_forms)  # DEBUG
            if len(chosen_new_lexicon_entries) > 0:  # DEBUG
                print "\t\tchosen_new_lexicon_entries: "  # DEBUG
                for sf, sem in chosen_new_lexicon_entries:  # DEBUG
                    print "\t\t\t'"+sf+"' :- "+p.print_parse(sem, show_category=True)  # DEBUG
            if not match or len(correct_new_lexicon_entries) > 0:
                if len(correct_new_lexicon_entries) > 0:
                    num_genlex_only += 1
                print "\t\ttraining example generated:"  # DEBUG
                print "\t\t\tcorrect_parse: "+p.print_parse(correct_parse.node, show_category=True)  # DEBUG
                print "\t\t\tcorrect_score: "+str(correct_score)  # DEBUG
                print "\t\t\tcorrect_skips: " + str(correct_skipped_surface_forms)  # DEBUG
                if len(correct_new_lexicon_entries) > 0:  # DEBUG
                    print "\t\t\tcorrect_new_lexicon_entries: "  # DEBUG
                    for sf, sem in correct_new_lexicon_entries:  # DEBUG
                        print "\t\t\t\t'"+sf+"' :- "+p.print_parse(sem, show_category=True)  # DEBUG
                print "\t\t\ty: "+p.print_parse(y, show_category=True)  # DEBUG

    # Output relevant data in results structure.
    result = {'num_trainable': num_trainable, 'num_matches': num_matches, 'num_fails': num_fails, 'num_genlex_only': num_genlex_only,
              'x': x, 'chosen_parse': chosen_parse, 'correct_parse': correct_parse, 'chosen_new_lexicon_entries': chosen_new_lexicon_entries,
              'correct_new_lexicon_entries': correct_new_lexicon_entries, 'chosen_skipped_surface_forms': chosen_skipped_surface_forms,
              'correct_skipped_surface_forms': correct_skipped_surface_forms}
    with open(outfile, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parser_infile', type=str, required=True,
                        help="the parser pickle")
    parser.add_argument('--pairs_infile', type=str, required=True,
                        help="the pairs pickle")
    parser.add_argument('--pair_idx', type=int, required=True,
                        help="the pair idx in the pairs_infile pickle this thread will process")
    parser.add_argument('--outfile', type=str, required=True,
                        help="where to dump new pair information")
    parser.add_argument('--x', type=str, required=False,
                        help="an input utterance")
    parser.add_argument('--y', type=str, required=False,
                        help="a target form")

    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
