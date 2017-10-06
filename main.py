#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import pickle
import CKYParser
import Lexicon
import Ontology


def main():

    # Load parameters from command line.
    ontology_fn = FLAGS_ontology_fn
    lexicon_fn = FLAGS_lexicon_fn
    train_pairs_fn = FLAGS_train_pairs_fn
    model_fn = FLAGS_model_fn
    validation_pairs_fn = FLAGS_validation_pairs_fn
    lexicon_embeddings = FLAGS_lexicon_embeddings
    max_epochs = FLAGS_max_epochs
    epochs_between_validations = FLAGS_epochs_between_validations
    allow_merge = True if FLAGS_allow_merge == 1 else False
    perform_type_raising = True if FLAGS_perform_type_raising == 1 else False
    verbose = FLAGS_verbose
    assert validation_pairs_fn is None or max_epochs >= epochs_between_validations

    o = Ontology.Ontology(ontology_fn)
    l = Lexicon.Lexicon(o, lexicon_fn, word_embeddings_fn=lexicon_embeddings,)
    p = CKYParser.CKYParser(o, l, allow_merge=allow_merge,
                            lexicon_weight=1.0, perform_type_raising=perform_type_raising)

    # hyperparameter adjustments
    p.max_multiword_expression = 1
    p.max_missing_words_to_try = 0  # basically disallows polysemy that isn't already present in lexicon

    # Train the parser one epoch at a time, examining validation performance between each epoch.
    train_data = p.read_in_paired_utterance_semantics(train_pairs_fn)
    val_data = p.read_in_paired_utterance_semantics(validation_pairs_fn) if validation_pairs_fn is not None else None
    print "finished instantiating parser; beginning training"
    for epoch in range(0, max_epochs, epochs_between_validations):
        if val_data is not None:
            acc_at_1 = get_performance_on_pairs(p, val_data)
            print "validation accuracy at 1 for epoch " + str(epoch) + ": " + str(acc_at_1)
        converged = p.train_learner_on_semantic_forms(train_data, epochs=epochs_between_validations,
                                                      epoch_offset=epoch, reranker_beam=10,
                                                      verbose=verbose)
        if converged:
            print "training converged after epoch " + str(epoch)
            break
    if val_data is not None:
        acc_at_1 = get_performance_on_pairs(p, val_data)
        print "validation accuracy at 1 at training stop: " + str(acc_at_1)

    # Write the parser to file.
    print "writing trained parser to file..."
    with open(model_fn, 'wb') as f:
        pickle.dump(p, f)
    print "... done"


def get_performance_on_pairs(p, d):
    num_correct = 0
    for [x, y] in d:
        parse_generator = p.most_likely_cky_parse(x, reranker_beam=10)
        best, score, _, _ = next(parse_generator)
        if best is not None and y.equal_allowing_commutativity(best.node, p.commutative_idxs, ontology=p.ontology):
            num_correct += 1
    return float(num_correct) / float(len(d))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ontology_fn', type=str, required=True,
                        help="the parser ontology text file")
    parser.add_argument('--lexicon_fn', type=str, required=True,
                        help="the parser lexicon text file")
    parser.add_argument('--train_pairs_fn', type=str, required=True,
                        help="pairs of sentence and gold semantic forms on which to train parser")
    parser.add_argument('--model_fn', type=str, required=True,
                        help="output filename for the trained parser model")
    parser.add_argument('--validation_pairs_fn', type=str, required=False,
                        help="pairs of sentence and gold semantic forms on which to test parser")
    parser.add_argument('--lexicon_embeddings', type=str, required=False,
                        help="word embeddings to look up novel words against those in the lexicon")
    parser.add_argument('--max_epochs', type=int, required=False, default=10,
                        help="maximum epochs to iterate over the training data")
    parser.add_argument('--epochs_between_validations', type=int, required=False, default=1,
                        help="how many epochs to run between validation tests")
    parser.add_argument('--allow_merge', type=int, required=False, default=1,
                        help="whether to allow the parser to use the merge operation")
    parser.add_argument('--perform_type_raising', type=int, required=False, default=1,
                        help="whether to type-raise bare nouns (requires <e,t> types)")
    parser.add_argument('--verbose', type=int, required=False, default=1,
                        help="the verbosity level during training in 0, 1, 2")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
