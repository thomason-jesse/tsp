__author__ = 'jesse'

import copy
import math
import numpy as np
import operator
import random
import sys
import ParseNode
import SemanticNode

neg_inf = float('-inf')
random.seed(4)


class Parameters:
    def __init__(self, ont, lex, allow_merge, use_language_model=False, lexicon_weight=1.0):
        debug = False

        self.ontology = ont
        self.lexicon = lex
        self.use_language_model = use_language_model

        # get initial count data structures
        self._token_given_token_counts = {}
        self._CCG_given_token_counts = self.init_ccg_given_token(lexicon_weight)
        self._CCG_production_counts = self.init_ccg_production(lexicon_weight, allow_merge)
        self._CCG_root_counts = {}
        self._lexicon_entry_given_token_counts = self.init_lexicon_entry(lexicon_weight)
        self._semantic_counts = {}
        self._skipwords_given_surface_form = self.init_skipwords_given_surface_form(lexicon_weight)

        if debug:
            print "_CCG_given_token_counts: "+str(self._CCG_given_token_counts)  # DEBUG
            print "_CCG_production_counts: "+str(self._CCG_production_counts)  # DEBUG
            print "_lexicon_entry_counts: "+str(self._lexicon_entry_given_token_counts)  # DEBUG
            print "_semantic_counts: "+str(self._semantic_counts)  # DEBUG

        # calculate probability tables from counts
        # note that this needs to happen every time counts are updated
        self.token_given_token = {}
        self.CCG_given_token = {}
        self.CCG_production = {}
        self.CCG_root = {}
        self.lexicon_entry_given_token = {}
        self.semantic = {}
        self.skipwords_given_surface_form = {}

        # update probabilities given new counts
        self.update_probabilities()

        if debug:
            print "CCG_given_token: "+str(self.CCG_given_token)  # DEBUG
            print "CCG_production: "+str(self.CCG_production)  # DEBUG
            print "lexicon_entry: "+str(self.lexicon_entry_given_token)  # DEBUG
            print "semantic: "+str(self.semantic)  # DEBUG

    # update the probability tables given counts
    def update_probabilities(self):
        missing_entry_mass = 0

        # skipwords just become heuristically less likely the more times they're used as skipwords correctly
        # fixed penalty 0.5 for skipping word, adjusted based on count of skip/not skip based on
        # p(skip_w | skip_w_count) = sigmoid(skip_w_count)
        for sf_idx in self._skipwords_given_surface_form.keys():
            self.skipwords_given_surface_form[sf_idx] = math.log(1 / (1 +
                                                                      math.exp(
                                                                          -self._skipwords_given_surface_form[sf_idx])))

        language_model_surface_forms = range(-1, len(self.lexicon.surface_forms))
        if self.use_language_model:
            # get log probabilities for token_given_token ( P(sf|sf), P(sf=s|S)=1 )
            # use special indices -2 for start, -3 for end, -1 for unknown token
            language_model_surface_forms.extend([-2, -3])
            for sf_idx in language_model_surface_forms:
                for next_idx in language_model_surface_forms:
                    if (sf_idx, next_idx) not in self._token_given_token_counts:
                        self._token_given_token_counts[(sf_idx, next_idx)] = missing_entry_mass
                nums = [self._token_given_token_counts[(sf_idx, next_idx)]
                        for next_idx in language_model_surface_forms]
                num_min = 0
                mass = 0
                if len(nums) > 0:
                    num_min = float(min(nums))
                    mass = float(sum(nums)) - num_min*len(nums) + len(nums)
                for next_idx in language_model_surface_forms:
                    key = (sf_idx, next_idx)
                    self.token_given_token[key] = math.log((1+self._token_given_token_counts[key]-num_min) / mass) \
                        if mass > 0 else math.log(1.0 / len(nums))

        # get log probabilities for CCG_root p(root)
        self._CCG_root_counts[-1] = missing_entry_mass  # hallucinate unseen roots score
        nums = [self._CCG_root_counts[cat_idx] for cat_idx in self._CCG_root_counts]
        num_min = 0
        mass = 0
        if len(nums) > 0:
            num_min = float(min(nums))
            mass = float(sum(nums)) - num_min*len(nums) + len(nums)
        for key in self._CCG_root_counts:
            self.CCG_root[key] = (math.log((1+self._CCG_root_counts[key]-num_min) / mass)
                                  if mass > 0 else math.log(1.0 / len(nums)))

        for cat_idx in range(0, len(self.lexicon.categories)):

            # get log probabilities for CCG_given_token ( P(ccg|surface), P(ccg=c|S)=1 )
            self._CCG_given_token_counts[(cat_idx, -1)] = missing_entry_mass  # hallucinate missing entries
            nums = [self._CCG_given_token_counts[(cat_idx, sf_idx)]
                    for sf_idx in range(-1, len(self.lexicon.surface_forms))
                    if (cat_idx, sf_idx) in self._CCG_given_token_counts]
            num_min = 0
            mass = 0
            if len(nums) > 0:
                num_min = float(min(nums))
                mass = float(sum(nums)) - num_min*len(nums)
            for sf_idx in range(-1, len(self.lexicon.surface_forms)):
                key = (cat_idx, sf_idx)
                if key in self._CCG_given_token_counts:
                    self.CCG_given_token[key] = math.log((1+self._CCG_given_token_counts[key]-num_min) / mass) \
                        if mass > 0 else math.log(1.0 / len(nums))

                # get log probabilities for lexicon_entry_given_token
                # ( P(sem|surface), P(sem=s|S)=1 )
                for sem_idx in range(0, len(self.lexicon.semantic_forms)):
                    if (sem_idx, sf_idx) not in self._lexicon_entry_given_token_counts:
                        self._lexicon_entry_given_token_counts[(sem_idx, sf_idx)] = missing_entry_mass  # hallucinate
                entry_nums = [self._lexicon_entry_given_token_counts[(sem_idx, sf_idx)]
                              for sem_idx in (self.lexicon.entries[sf_idx]
                              if sf_idx > -1 else range(0, len(self.lexicon.semantic_forms)))]
                entry_num_min = 0
                entry_mass = 0
                if len(entry_nums) > 0:
                    entry_num_min = float(min(entry_nums))
                    entry_mass = float(sum(entry_nums)) - entry_num_min*len(entry_nums) + len(entry_nums)
                for sem_idx in (self.lexicon.entries[sf_idx]
                                if sf_idx > -1 else range(0, len(self.lexicon.semantic_forms))):
                    key = (sem_idx, sf_idx)
                    self.lexicon_entry_given_token[key] = math.log(
                        (1+self._lexicon_entry_given_token_counts[key]-entry_num_min) / entry_mass) \
                        if entry_mass > 0 else math.log(1.0 / len(entry_nums))

            # get log probabilities for CCG_production ( P(ccg|(leftcgg, rightccg)), P(ccg=c|LR)=1 )
            nums = [self._CCG_production_counts[(cat_idx, l_idx, r_idx)]
                    for l_idx in range(0, len(self.lexicon.categories))
                    for r_idx in range(0, len(self.lexicon.categories))
                    if (cat_idx, l_idx, r_idx) in self._CCG_production_counts]
            if len(nums) == 0:
                continue
            num_min = float(min(nums))
            mass = float(sum(nums)) - num_min*len(nums) + len(nums)
            for l_idx in range(0, len(self.lexicon.categories)):
                for r_idx in range(0, len(self.lexicon.categories)):
                    key = (cat_idx, l_idx, r_idx)
                    if key in self._CCG_production_counts:
                        self.CCG_production[key] = math.log((1+self._CCG_production_counts[key]-num_min) / mass) \
                            if mass > 0 else math.log(1.0 / len(nums))

        # get log probabilities for semantic_args ( P(arg|(pred,pos)), P(arg=a|(P, PS))=1
        for arg_idx in range(0, len(self.ontology.preds)):
            nums = [self._semantic_counts[(pred_idx, arg_idx, pos)]
                    for pred_idx in range(0, len(self.ontology.preds))
                    for pos in range(0, self.ontology.num_args[pred_idx])
                    if (pred_idx, arg_idx, pos) in self._semantic_counts]
            if len(nums) == 0:
                continue
            num_min = float(min(nums))
            mass = float(sum(nums)) - num_min*len(nums) + len(nums)
            for pred_idx in range(0, len(self.ontology.preds)):
                for pos in range(0, self.ontology.num_args[pred_idx]):
                    key = (pred_idx, arg_idx, pos)
                    if key in self._semantic_counts:
                        self.semantic[key] = math.log((1+self._semantic_counts[key]-num_min) / mass) \
                            if mass > 0 else math.log(1.0 / len(nums))

    # indexed by surface_forms_idx, value parameter weight
    def init_skipwords_given_surface_form(self, lexicon_weight):
        # initialize with negative lexicon weight because these words should -not- be rewarded for skipping
        return {sf_idx: -lexicon_weight for sf_idx in range(len(self.lexicon.surface_forms))}

    # indexed by (categories idx, surface forms idx), value parameter weight
    def init_ccg_given_token(self, lexicon_weight):
        return {(self.lexicon.semantic_forms[sem_idx].category, sf_idx): lexicon_weight
                for sf_idx in range(0, len(self.lexicon.entries))
                for sem_idx in self.lexicon.entries[sf_idx]}

    # indexed by categories idxs (production, left, right), value parameter weight
    def init_ccg_production(self, lexicon_weight, allow_merge):
        ccg_production = {}
        for cat_idx in range(0, len(self.lexicon.categories)):

            # add production rules of form Z -> X Y for X(Y)=Z or Y(X)=Z (function application)
            consumables = self.lexicon.find_consumables_for_cat(cat_idx)
            for d, child in consumables:
                if d == 0:  # consumes to the left
                    l = child
                    r = self.lexicon.categories.index([cat_idx, d, child])
                else:  # consumes to the right
                    r = child
                    l = self.lexicon.categories.index([cat_idx, d, child])
                ccg_production[(cat_idx, l, r)] = lexicon_weight

            # add production rules of form X -> X X for X<>X=X (merge)
            if allow_merge:
                ccg_production[(cat_idx, cat_idx, cat_idx)] = lexicon_weight

        return ccg_production

    # indexed by (surface forms idx, semantic forms idx), value parameter weight
    def init_lexicon_entry(self, lexicon_weight):
        return {(sem_idx, sf_idx): lexicon_weight
                for sf_idx in range(0, len(self.lexicon.entries))
                for sem_idx in self.lexicon.entries[sf_idx]}

    # takes in a parse node and returns its log probability
    def get_semantic_score(self, n):

        counts = self.count_semantics(n)
        score = 0.0
        for key in counts:
            if key in self.semantic:
                for _ in range(0, counts[key]):
                    score += self.semantic[key]
        return score

    # take in ParseNode y and calculate bigram token counts as dictionary
    def count_token_bigrams(self, y):
        t = [l.surface_form for l in y.get_leaves()]
        for t_idx in range(0, len(t)):
            if type(t[t_idx]) is str:
                t[t_idx] = self.lexicon.surface_forms.index(t[t_idx])
            if t[t_idx] is None:
                raise RuntimeError("Leaf parse node has None surface form "+str(y))
        b = {}
        t.insert(0, -2)  # beginning of sentence token
        t.append(-3)  # end of sentence token
        for t_idx in range(0, len(t)-1):
            key = (t[t_idx], t[t_idx+1])
            if key not in b:
                b[key] = 0
            b[key] += 1
        return b

    # takes in a parse or semantic node and returns the counts of its (pred, arg, pos) entries
    def count_semantics(self, sn):

        # convert passed ParseNodes to SemanticNode member
        try:
            sn = sn.node
        except AttributeError:
            pass

        counts = {}
        if sn.children is not None:
            pred = sn.idx
            if sn.is_lambda:
                pred = "lambda_inst" if sn.is_lambda_instantiation else "lambda"
            for pos in range(0, len(sn.children)):
                arg = sn.children[pos].idx
                if sn.children[pos].is_lambda:
                    arg = "lambda_inst" if sn.children[pos].is_lambda_instantiation else "lambda"
                key = (pred, arg, pos)
                if key not in counts:
                    counts[key] = 0
                counts[key] += 1

            for c in sn.children:
                child_counts = self.count_semantics(c)
                for ckey in child_counts:
                    if ckey not in counts:
                        counts[ckey] = 0
                    counts[ckey] += child_counts[ckey]

        return counts

    # take in examples t=(x,y,z) for x an expression, y an incorrect semantic form, z a correct semantic form
    def update_learned_parameters(self, t):
        debug = False

        lr = 1.0 / math.sqrt(len(t))

        # update counts given new training data
        for x, y, z, y_lex, z_lex, y_skipped, z_skipped in t:

            # update skips separately since they're weird
            for y_key in y_skipped:
                if y_key not in z_skipped:
                    if y_key not in self.lexicon.surface_forms:
                        self.lexicon.surface_forms.append(y_key)
                        self.lexicon.entries.append([])
                    y_sidx = self.lexicon.surface_forms.index(y_key)
                    if y_sidx not in self._skipwords_given_surface_form:
                        self._skipwords_given_surface_form[y_sidx] = 0
                    self._skipwords_given_surface_form[y_sidx] -= lr  # should not have skipped
            for z_key in z_skipped:
                if z_key not in y_skipped:
                    if z_key not in self.lexicon.surface_forms:
                        self.lexicon.surface_forms.append(z_key)
                        self.lexicon.entries.append([])
                    z_sidx = self.lexicon.surface_forms.index(z_key)
                    if z_sidx not in self._skipwords_given_surface_form:
                        self._skipwords_given_surface_form[z_sidx] = 0
                    self._skipwords_given_surface_form[z_sidx] += lr  # should have skipped

            # expand parameter maps for new lexical entries and update parse structures
            # so that roots have correct surface form idxs (which may not have been assigned
            # yet at the time of parsing)
            for form, lex in [[y, y_lex], [z, z_lex]]:
                for surface_form, sem_node in lex:

                    # add this surface form and sem node to the lexicon and tie them together
                    if type(surface_form) is str and surface_form not in self.lexicon.surface_forms:
                        # print "Parameters: adding new surface form '"+surface_form+"'"  # DEBUG
                        self.lexicon.surface_forms.append(surface_form)
                    sf_idx = self.lexicon.surface_forms.index(surface_form)
                    if sem_node not in self.lexicon.semantic_forms:
                        # print "Parameters: adding new semantic form '"+str(sem_node)+"'"  # DEBUG
                        self.lexicon.semantic_forms.append(sem_node)
                    sem_idx = self.lexicon.semantic_forms.index(sem_node)
                    if sf_idx == len(self.lexicon.entries):
                        self.lexicon.entries.append([])
                    if sem_idx not in self.lexicon.entries[sf_idx]:
                        # print "Parameters: adding new entry "+str(sf_idx)+"=>"+str(sem_idx)  # DEBUG
                        self.lexicon.entries[sf_idx].append(sem_idx)

                    # update count data structures given new entry
                    key = (self.lexicon.semantic_forms[sem_idx].category, sf_idx)
                    if key not in self._CCG_given_token_counts:
                        self._CCG_given_token_counts[key] = 1.0
                    # note that no new CCG production rules can be generated from current procedure
                    key = (sf_idx, sem_idx)
                    if key not in self._lexicon_entry_given_token_counts:
                        self._lexicon_entry_given_token_counts[key] = 1.0

                # update form leaves with new surface idx as needed
                form_leaves = form.get_leaves()
                for leaf in form_leaves:
                    if type(leaf.surface_form) is str:
                        leaf.surface_form = self.lexicon.surface_forms.index(leaf.surface_form)

            # do perceptron-style updates of counts
            parameter_extractors = [count_ccg_surface_form_pairs,
                                    count_ccg_productions,
                                    count_lexical_entries,
                                    self.count_semantics,
                                    count_ccg_root]
            parameter_structures = [self._CCG_given_token_counts,
                                    self._CCG_production_counts,
                                    self._lexicon_entry_given_token_counts,
                                    self._semantic_counts,
                                    self._CCG_root_counts]
            if self.use_language_model:
                parameter_extractors.append(self.count_token_bigrams)
                parameter_structures.append(self._token_given_token_counts)
            for i in range(0, len(parameter_structures)):
                y_keys = parameter_extractors[i](y)
                z_keys = parameter_extractors[i](z)
                seen_keys = []
                for z_key in z_keys:
                    z_val = z_keys[z_key]
                    if z_key in y_keys:
                        y_val = y_keys[z_key]
                        seen_keys.append(z_key)
                    else:
                        y_val = 0
                    if z_key not in parameter_structures[i]:
                        parameter_structures[i][z_key] = 0
                    # formerly (z_val - y_val)  / (z_val + y_val)
                    parameter_structures[i][z_key] += lr * (z_val - y_val)
                for y_key in y_keys:
                    if y_key in seen_keys:
                        continue
                    y_val = y_keys[y_key]
                    z_val = 0
                    if y_key not in parameter_structures[i]:
                        parameter_structures[i][y_key] = 0
                    parameter_structures[i][y_key] += lr * (z_val - y_val)

        if debug:
            print "_token_given_token_counts: "+str(self._token_given_token_counts)  # DEBUG
            print "_CCG_given_token_counts: "+str(self._CCG_given_token_counts)  # DEBUG
            print "_CCG_production_counts: "+str(self._CCG_production_counts)  # DEBUG
            print "_lexicon_entry_counts: "+str(self._lexicon_entry_given_token_counts)  # DEBUG
            print "_semantic_counts: "+str([str((self.ontology.preds[pred] if type(pred) is int else pred,
                                                 self.ontology.preds[arg] if type(arg) is int else arg,
                                                 str(pos)))+": " +
                                           str(self._semantic_counts[(pred, arg, pos)])
                                           for pred, arg, pos in self._semantic_counts])  # DEBUG
            print {self.lexicon.compose_str_from_category(idx): self._CCG_root_counts[idx]
                   for idx in self._CCG_root_counts}  # DEBUG

        # update probabilities given new counts
        self.update_probabilities()

        if debug:
            print "token_given_token: "+str(self.token_given_token)  # DEBUG
            print "CCG_given_token: "+str(self.CCG_given_token)  # DEBUG
            print "CCG_production: "+str(self.CCG_production)  # DEBUG
            print "lexicon_entry: "+str(self.lexicon_entry_given_token)  # DEBUG
            print "semantic: "+str([str((self.ontology.preds[pred] if type(pred) is int else pred,
                                        self.ontology.preds[arg] if type(arg) is int else arg,
                                        str(pos)))+": " +
                                    str(self.semantic[(pred, arg, pos)])
                                    for pred, arg, pos in self.semantic])  # DEBUG
            print {self.lexicon.compose_str_from_category(idx): self.CCG_root[idx]
                   for idx in self.CCG_root if idx > -1}  # DEBUG
            print "unseen prob: " + str(self.CCG_root[-1])  # DEBUG


# take in ParseNode y to calculate (surface forms idx, semantic forms idx) pairs
def count_lexical_entries(y):
    pairs = {}
    token_assignments = y.get_leaves()
    for ta in token_assignments:
        k = (ta.surface_form, ta.semantic_form)
        if k not in pairs:
            pairs[k] = 0
        pairs[k] += 1
    return pairs


# take in ParseNode y to calculate (CCG, CCG, CCG) productions
def count_ccg_productions(y):
    productions = {}
    to_explore = [y]
    while len(to_explore) > 0:
        n = to_explore.pop()
        if n.children is not None:
            key = (n.node.category, n.children[0].node.category, n.children[1].node.category)
            if key not in productions:
                productions[key] = 0
            productions[key] += 1
            to_explore.extend(n.children)
    return productions


# take in ParseNode y and return the CCG of its root
def count_ccg_root(y):
    return {y.node.category: 1}


# take in ParseNode y to calculate (CCG, token) pairs
def count_ccg_surface_form_pairs(y):
    pairs = {}
    token_assignments = y.get_leaves()
    for ta in token_assignments:
        k = (ta.node.category, ta.surface_form)
        if k not in pairs:
            pairs[k] = 0
        pairs[k] += 1
    return pairs


class CKYParser:
    def __init__(self, ont, lex, use_language_model=False, lexicon_weight=1.0, perform_type_raising=True,
                 allow_merge=True):

        # resources given on instantiation
        self.ontology = ont
        self.lexicon = lex
        self.use_language_model = use_language_model

        # type-raise bare nouns in lexicon
        self.type_raised = {}  # map from semantic form idx to their type-raised form idx
        if perform_type_raising:
            self.type_raise_bare_nouns()

        # model parameter values
        self.theta = Parameters(ont, lex, allow_merge,
                                use_language_model=use_language_model, lexicon_weight=lexicon_weight)

        # additional linguistic information and parameters
        # TODO: read this from configuration files or have user specify it on instantiation
        self.allow_merge = allow_merge  # allows 'and' to merge to adjacent same-category nodes
        self.commutative_idxs = [self.ontology.preds.index('and')]  # can be expanded by users if there are more
        self.max_multiword_expression = 1  # max span of a multi-word expression to be considered during tokenization
        self.max_new_senses_per_utterance = 3  # max number of new word senses that can be induced on a training example
        self.max_cky_trees_per_token_sequence_beam = 100  # for tokenization of an utterance, max cky trees considered
        self.max_hypothesis_categories_for_unknown_token_beam = 10  # for unknown token, max syntax categories tried
        self.max_expansions_per_non_terminal = 10  # decides how many expansions to store per CKY cell
        self.max_new_skipwords_per_utterance = 2  # how many unknown skipwords to consider for a new utterance
        self.max_missing_words_to_try = 2  # how many words that have meanings already to sample for new meanings
        self.missing_lexicon_entry_given_token_penalty = -100  # additional log probability to lose for missing lex
        self.missing_CCG_given_token_penalty = -100  # additional log probability to lose for missing CCG

        # behavioral parameters
        self.safety = True  # set to False once confident about node combination functions' correctness

        # cache
        self.cached_combinations = {}  # indexed by left, then right node, value at result

    # access language model parameters to get a language score for a given parse node y
    # parse node leaves with string semantic forms are assumed to be unknown tokens
    def get_language_model_score(self, y):
        if y is None:
            return -sys.maxint
        t = [l.surface_form for l in y.get_leaves()]
        for t_idx in range(0, len(t)):
            if type(t[t_idx]) is str:
                t[t_idx] = -1
        score = 0
        t.insert(0, -2)
        t.append(-3)
        for t_idx in range(0, len(t)-1):
            key = (t[t_idx], t[t_idx+1])
            score += self.theta.token_given_token[key] if key in self.theta.token_given_token else 0.0
        return score

    # perform type-raising on leaf-level lexicon entries
    # this alters the given lexicon
    def type_raise_bare_nouns(self):
        bare_noun_cat_idx = self.lexicon.categories.index('N')
        raised_cat_idx = self.lexicon.categories.index([bare_noun_cat_idx, 1, bare_noun_cat_idx])
        e_idx = self.ontology.types.index('e')
        e_to_t_idx = self.ontology.types.index([self.ontology.types.index('e'), self.ontology.types.index('t')])
        to_add = []
        for sf_idx in range(0, len(self.lexicon.surface_forms)):
            for sem_idx in self.lexicon.entries[sf_idx]:
                sem = self.lexicon.semantic_forms[sem_idx]
                if (sem.category == bare_noun_cat_idx and not sem.is_lambda and
                        self.ontology.entries[sem.idx] == e_to_t_idx and sem.children is None):
                    # ensure there isn't already a raised predicate matching this bare noun
                    already_raised = False
                    for alt_idx in self.lexicon.entries[sf_idx]:
                        alt = self.lexicon.semantic_forms[alt_idx]
                        if (alt.category == raised_cat_idx and alt.is_lambda and
                                alt.type == e_idx and
                                not alt.children[0].is_lambda and
                                self.ontology.entries[alt.children[0].idx] == e_to_t_idx and
                                alt.children[0].children[0].is_lambda_instantiation):
                            already_raised = True
                            break
                    if not already_raised:
                        raised_sem = SemanticNode.SemanticNode(None, e_idx, raised_cat_idx,
                                                               True, lambda_name=0, is_lambda_instantiation=True)
                        raised_pred = copy.deepcopy(sem)
                        raised_pred.parent = raised_sem
                        lambda_inst = SemanticNode.SemanticNode(raised_pred, e_idx, bare_noun_cat_idx,
                                                                True, lambda_name=0, is_lambda_instantiation=False)
                        raised_pred.children = [lambda_inst]
                        raised_sem.children = [raised_pred]
                        to_add.append([sf_idx, sem_idx, raised_sem])
        for sf_idx, sem_idx, sem in to_add:
            # print "type_raise_bare_nouns raising: '"+self.lexicon.surface_forms[sf_idx] + \
            #       "':- "+self.print_parse(sem, show_category=True)  # DEBUG
            self.lexicon.semantic_forms.append(sem)
            self.lexicon.entries[sf_idx].append(len(self.lexicon.semantic_forms)-1)
            self.type_raised[sem_idx] = len(self.lexicon.semantic_forms)-1

    # print a SemanticNode as a string using the known ontology
    def print_parse(self, p, show_category=False, show_non_lambda_types=False):
        if p is None:
            return "NONE"
        elif show_category:
            if p.category is not None:
                s = self.lexicon.compose_str_from_category(p.category) + " : "
            else:
                s = "None : "
        else:
            s = ''
        t = self.ontology.compose_str_from_type(p.type)
        if p.is_lambda:
            if p.is_lambda_instantiation:
                s += "lambda " + str(p.lambda_name) + ":" + t + "."
            else:
                s += str(p.lambda_name)
        else:
            s += self.ontology.preds[p.idx]
            if show_non_lambda_types:
                s += ":"+t
        if p.children is not None:
            s += '(' + ','.join([self.print_parse(c, show_non_lambda_types=show_non_lambda_types)
                                 for c in p.children]) + ')'
        return s

    # read in data set of form utterance\nCCG : semantic_form\n\n...
    def read_in_paired_utterance_semantics(self, fname, allow_expanding_ont=False):
        d = []
        f = open(fname, 'r')
        f_lines = f.readlines()
        f.close()
        i = 0
        while i < len(f_lines):
            if len(f_lines[i].strip()) == 0:
                i += 1
                continue
            input_str = f_lines[i].strip()
            ccg_str, form_str = f_lines[i+1].strip().split(" : ")
            ccg = self.lexicon.read_category_from_str(ccg_str)
            form = self.lexicon.read_semantic_form_from_str(form_str, None, None, [],
                                                            allow_expanding_ont=allow_expanding_ont)
            form.category = ccg
            d.append([input_str, form])
            i += 3
        return d

    # take in a data set D=(x,y) for x expressions and y correct semantic form and update CKYParser parameters
    def train_learner_on_semantic_forms(self, d, epochs=10, epoch_offset=0, reranker_beam=1, verbose=2):
        for e in range(0, epochs):
            if verbose >= 1:
                print "epoch " + str(e + epoch_offset)  # DEBUG
            t, failures = self.get_training_pairs(d, verbose, reranker_beam=reranker_beam)
            if len(t) == 0:
                print "training converged at epoch " + str(e)
                if failures == 0:
                    return True
                else:
                    return False
            random.shuffle(t)
            self.theta.update_learned_parameters(t)
        return False

    # take in data set d=(x,y) for x strings and y correct semantic forms and calculate training pairs
    # training pairs in t are of form (x, y_chosen, y_correct, chosen_lex_entries, correct_lex_entries)
    # k determines how many parses to get for re-ranking
    # beam determines how many cky_trees to look through before giving up on a given input
    def get_training_pairs(self, d, verbose, reranker_beam=1):
        t = []
        num_trainable = 0
        num_matches = 0
        num_fails = 0
        num_genlex_only = 0
        for [x, y] in d:
            correct_parse = None
            correct_new_lexicon_entries = []
            cky_parse_generator = self.most_likely_cky_parse(x, reranker_beam=reranker_beam, known_root=y)
            chosen_parse, chosen_score, chosen_new_lexicon_entries, chosen_skipped_surface_forms = \
                next(cky_parse_generator)
            current_parse = chosen_parse
            correct_score = chosen_score
            current_new_lexicon_entries = chosen_new_lexicon_entries
            current_skipped_surface_forms = chosen_skipped_surface_forms
            match = False
            first = True
            if chosen_parse is None:
                if verbose >= 2:
                    print "WARNING: could not find valid parse for '" + x + "' during training"  # DEBUG
                num_fails += 1
                continue
            while correct_parse is None and current_parse is not None:
                if y.equal_allowing_commutativity(
                        current_parse.node, self.commutative_idxs, ontology=self.ontology):
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
                if verbose >= 2:
                    print "WARNING: could not find correct parse for '"+str(x)+"' during training"
                num_fails += 1
                continue
            if verbose >= 2:
                print "\tx: "+str(x)  # DEBUG
                print "\t\tchosen_parse: "+self.print_parse(chosen_parse.node, show_category=True)  # DEBUG
                print "\t\tchosen_score: "+str(chosen_score)  # DEBUG
                print "\t\tchosen_skips: "+str(chosen_skipped_surface_forms)  # DEBUG
                if len(chosen_new_lexicon_entries) > 0:  # DEBUG
                    print "\t\tchosen_new_lexicon_entries: "  # DEBUG
                    for sf, sem in chosen_new_lexicon_entries:  # DEBUG
                        print "\t\t\t'"+sf+"' :- "+self.print_parse(sem, show_category=True)  # DEBUG
            if not match or len(correct_new_lexicon_entries) > 0:
                if len(correct_new_lexicon_entries) > 0:
                    num_genlex_only += 1
                if verbose >= 2:
                    print "\t\ttraining example generated:"  # DEBUG
                    print "\t\t\tcorrect_parse: "+self.print_parse(correct_parse.node, show_category=True)  # DEBUG
                    print "\t\t\tcorrect_score: "+str(correct_score)  # DEBUG
                    print "\t\t\tcorrect_skips: " + str(correct_skipped_surface_forms)  # DEBUG
                    if len(correct_new_lexicon_entries) > 0:  # DEBUG
                        print "\t\t\tcorrect_new_lexicon_entries: "  # DEBUG
                        for sf, sem in correct_new_lexicon_entries:  # DEBUG
                            print "\t\t\t\t'"+sf+"' :- "+self.print_parse(sem, show_category=True)  # DEBUG
                    print "\t\t\ty: "+self.print_parse(y, show_category=True)  # DEBUG
                t.append([x, chosen_parse, correct_parse, chosen_new_lexicon_entries, correct_new_lexicon_entries,
                          chosen_skipped_surface_forms, correct_skipped_surface_forms])
        if verbose >= 1:
            print "\tmatched "+str(num_matches)+"/"+str(len(d))  # DEBUG
            print "\ttrained "+str(num_trainable)+"/"+str(len(d))  # DEBUG
            print "\tgenlex only "+str(num_genlex_only)+"/"+str(len(d))  # DEBUG
            print "\tfailed "+str(num_fails)+"/"+str(len(d))  # DEBUG
        return t, num_fails

    # yields the next most likely CKY parse of input string s
    # if the root of the tree is known (during supervised training, for example),
    # providing it as an argument to this method allows top-down generation
    # to find new lexical entries for surface forms not yet recognized
    def most_likely_cky_parse(self, s, reranker_beam=1, known_root=None):
        debug = False
        if len(s) == 0:
            raise AssertionError("Cannot parse provided string of length zero")

        tk_seq = self.tokenize(s)

        # add lexical entries for unseen tokens based on nearest neighbors
        for tk in tk_seq:
            if tk not in self.lexicon.surface_forms:
                nn = self.lexicon.get_lexicon_word_embedding_neighbors(
                    tk, len(self.lexicon.surface_forms))
                if len(nn) > 0:
                    self.lexicon.surface_forms.append(tk)
                    self.lexicon.entries.append([])
                    sfidx = self.lexicon.surface_forms.index(tk)
                    self.lexicon.neighbor_surface_forms.append(sfidx)
                    # TODO: this should probably be a helper function to Parameters
                    # take on the skipwords score of nearest neighbor, adjusted towards 0 for similarity
                    if sfidx not in self.theta._skipwords_given_surface_form:
                        self.theta._skipwords_given_surface_form[sfidx] = \
                            self.theta._skipwords_given_surface_form[nn[0][0]] * nn[0][1]
                    for nsfidx, sim in nn:
                        for sem_idx in self.lexicon.entries[nsfidx]:
                            # adjust count so that sim 0.5 is the same as a missing entry
                            # sim 1 is the same as no penalty (e.g. identical word)
                            # sim 0 is twice as bad as treating entry as simply missing
                            # TODO: this should probably be a helper function to Parameters
                            self.theta._lexicon_entry_given_token_counts[(sem_idx, sfidx)] = \
                                max(self.theta._lexicon_entry_given_token_counts[(sem_idx, sfidx)]
                                    if (sem_idx, sfidx) in self.theta._lexicon_entry_given_token_counts else neg_inf,
                                self.theta._lexicon_entry_given_token_counts[(sem_idx, nsfidx)] +
                                    ((self.missing_lexicon_entry_given_token_penalty * 2) * (1 - sim)))
                            self.lexicon.entries[sfidx].append(sem_idx)
                            if debug:
                                print ("nearest neighbor expansion to '" + tk + "' includes that for " +
                                       self.lexicon.surface_forms[nsfidx] + " :- " +
                                       self.print_parse(self.lexicon.semantic_forms[sem_idx], True) +
                                       " with initial penalized count " +
                                       str(self.theta._lexicon_entry_given_token_counts[(sem_idx, sfidx)]))
                    self.theta.update_probabilities()  # since we made changes to the counts

        # calculate token sequence variations with number of skips allowed
        num_likely_skips = len([tk for tk in tk_seq if tk not in self.lexicon.surface_forms or
                                self.lexicon.surface_forms.index(tk) not in self.theta.skipwords_given_surface_form or
                                self.theta.skipwords_given_surface_form[self.lexicon.surface_forms.index(tk)]
                                >= math.log(0.5) or np.isclose(math.log(0.5),
                                                               self.theta.skipwords_given_surface_form[
                                                                   self.lexicon.surface_forms.index(tk)])])
        skips_allowed = min(len(tk_seq) - 1, num_likely_skips + self.max_new_skipwords_per_utterance)
        considered_so_far = []
        skip_sequence_generator = self.get_token_skip_sequence(tk_seq, skips_allowed, True,
                                                               yielded_above_threshold=considered_so_far)
        curr_tk_seq, score, skipped_surface_forms = next(skip_sequence_generator)
        considering_below_heuristic = False
        while curr_tk_seq is not None:
            if debug:
                print ("with skips_allowed " + str(skips_allowed) + " generated candidate sequence " +
                       str(curr_tk_seq) + " with score " + str(score) + " skipping " + str(skipped_surface_forms))
                _ = raw_input()

            # create generator for current sequence set and get most likely parses
            ccg_parse_tree_generator = self.most_likely_ccg_parse_tree_given_tokens(curr_tk_seq)
            # get next most likely CCG parse tree out of CKY algorithm
            ccg_tree, tree_score = next(ccg_parse_tree_generator)
            # ccg_tree indexed by spans (i, j) valued at [CCG category, left span, right span]
            while ccg_tree is not None:

                if debug:
                    print "ccg tree: "+str(tree_score)  # DEBUG
                    for span in ccg_tree:  # DEBUG
                        print str(span) + ": [" + self.lexicon.compose_str_from_category(ccg_tree[span][0]) + \
                            "," + str(ccg_tree[span][1]) + "," + str(ccg_tree[span][2]) + "]"  # DEBUG

                # get next most likely assignment of semantics to given CCG categories
                semantic_assignment_generator = self.most_likely_semantic_leaves(curr_tk_seq, ccg_tree,
                                                                                 known_root=known_root)

                # use discriminative re-ranking to pull next most likely cky parse given leaf generator
                parse_tree_generator = self.most_likely_reranked_cky_parse(ccg_tree, semantic_assignment_generator,
                                                                           reranker_beam, known_root=known_root)
                parse_tree, parse_score, new_lexicon_entries = next(parse_tree_generator)
                while parse_tree is not None:
                    yield parse_tree, score + parse_score + tree_score, new_lexicon_entries, skipped_surface_forms
                    parse_tree, parse_score, new_lexicon_entries = next(parse_tree_generator)

                ccg_tree, tree_score = next(ccg_parse_tree_generator)

            curr_tk_seq, score, skipped_surface_forms = next(skip_sequence_generator)
            if curr_tk_seq is None and not considering_below_heuristic:
                if debug:
                    print ("exhausted heuristic skips; moving on to full enumeration with " +
                           "considered_so_far=" + str(considered_so_far))
                skip_sequence_generator = self.get_token_skip_sequence(tk_seq, skips_allowed, False,
                                                                       yielded_above_threshold=considered_so_far)
                curr_tk_seq, score, skipped_surface_forms = next(skip_sequence_generator)
                considering_below_heuristic = True

        # out of parse trees to try
        yield None, neg_inf, [], []

    # yields the next most likely sequence of tokens allowing up to k skips
    def get_token_skip_sequence(self, tks, k, heuristic, yielded_above_threshold=None):
        debug = False
        if debug:
            print "get_token_skip_sequence: " + str(tks) + ", " + str(k) + ", " + str(heuristic)

        # get skip scores from parser parameters
        initial_skip_threshold = math.log(0.5)
        skip_score = {idx: self.theta.skipwords_given_surface_form[self.lexicon.surface_forms.index(tks[idx])]
                      if tks[idx] in self.lexicon.surface_forms and self.lexicon.surface_forms.index(tks[idx])
                      in self.theta.skipwords_given_surface_form else initial_skip_threshold
                      for idx in range(len(tks))}

        # if not allowed to skip anymore, just return given sequence at no penalty
        if k == 0:
            yield tks, 0, []

        # greedily yield every combination with up to k skips for tokens with skip probability greater than 0.5
        # record tokens yielded to the given structure so that they can be skipped by later calls
        elif heuristic:

            if debug:
                print ("get_token_skip_sequence: skip_score sort " +
                       str(sorted(skip_score.items(), key=operator.itemgetter(1), reverse=True)))
                print "get_token_skip_sequence: initial_skip_threshold=" + str(initial_skip_threshold)

            for idx, score in sorted(skip_score.items(), key=operator.itemgetter(1), reverse=True):
                if score >= initial_skip_threshold or np.isclose(score, initial_skip_threshold):
                    _tks = [tks[jdx] for jdx in range(len(tks)) if jdx != idx]
                    _gen = self.get_token_skip_sequence(_tks, k - 1, True, yielded_above_threshold=None)
                    _skip_tks, _score, _ssf = next(_gen)
                    while _skip_tks is not None:
                        if yielded_above_threshold is not None:
                            yielded_above_threshold.append(_skip_tks)
                        yield _skip_tks, score + _score, [tks[idx]] + _ssf
                        _skip_tks, _score, _ssf = next(_gen)
                else:
                    if debug:
                        print "get_token_skip_sequence: no further scores above heuristic at this depth"
                    break
            if yielded_above_threshold is not None:
                yielded_above_threshold.append(tks)
            yield tks, 0, []

        # enumerate remaining skips, ignoring those that have already been yielded according to given list
        # yield the remaining skips in sorted sum order
        else:

            remaining_sequences = []
            remaining_scores = []
            remaining_skipped_surface_forms = []
            # get sequences for lower k's beneath self
            for idx, score in sorted(skip_score.items(), key=operator.itemgetter(1), reverse=True):
                _tks = [tks[jdx] for jdx in range(len(tks)) if jdx != idx]
                _gen = self.get_token_skip_sequence(_tks, k - 1, False,
                                                    yielded_above_threshold=yielded_above_threshold)
                _skip_tks, _score, _ssf = next(_gen)
                while _skip_tks is not None:
                    if _skip_tks not in yielded_above_threshold:
                        remaining_sequences.append(_skip_tks)
                        remaining_scores.append(score + _score)
                        remaining_skipped_surface_forms.append([tks[idx]] + _ssf)
                    _skip_tks, _score, _ssf = next(_gen)
            # add self to remaining sequence list
            if tks not in yielded_above_threshold:
                remaining_sequences.append(tks)
                remaining_scores.append(0)
                remaining_skipped_surface_forms.append([])

            if debug:
                print ("get_token_skip_sequence: enumerated remaining " + str(len(remaining_sequences)) +
                       " sequences; returning them by score total")
            for idx in sorted(range(len(remaining_scores)), key=lambda _i: remaining_scores[_i], reverse=True):
                yield remaining_sequences[idx], remaining_scores[idx], remaining_skipped_surface_forms[idx]

        # explicitly indicate that the generator is empty
        yield None, None, None

    # yields the next most likely parse tree after re-ranking in beam k
    # searches over leaf assignments in beam given a leaf assignment generator
    def most_likely_reranked_cky_parse(self, ccg_tree, semantic_assignment_generator, k, known_root=None):

        # get up to top k candidates, allowing generation
        candidates = []  # list of trees
        scores = []  # list of scores after discriminative re-ranking
        new_lex = []  # new lexical entries associated with each
        for curr_leaves, curr_leaves_score in semantic_assignment_generator:
            curr_generator = self.most_likely_tree_generator(curr_leaves, ccg_tree, sem_root=known_root)
            # print "curr_leaves: "+str([self.print_parse(cl.node, show_category=True) for cl in curr_leaves])  # DEBUG
            for curr_tree, curr_new_lex in curr_generator:
                # print "...added candidate"  # DEBUG
                candidates.append(curr_tree)
                scores.append(self.theta.get_semantic_score(curr_tree) + curr_leaves_score)
                new_lex.append(curr_new_lex)
                if len(candidates) > k:
                    break
            if len(candidates) > k:
                break

        # print "reranker candidate parses:"  # DEBUG
        # for idx in range(0, len(candidates)):  # DEBUG
        #     print "candidate: "+self.print_parse(candidates[idx].node, show_category=True)  # DEBUG
        #     print "\tscore: "+str(scores[idx])  # DEBUG

        # yield remaining best candidates in order
        score_dict = {idx: scores[idx] for idx in range(0, len(scores))}
        for idx, score in sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True):
            yield candidates[idx], score, new_lex[idx]

        # out of candidates
        yield None, neg_inf, []

    # yields the next most likely assignment of semantic values to ccg nodes
    # returns None if leaf assignments cannot propagate to root
    # returns None if no assignment to missing leaf entries will allow propagation to root
    def most_likely_tree_generator(self, parse_leaves, ccg_tree, sem_root=None):
        debug = False

        if debug:
            print "most_likely_tree_generator: called for ccg_tree: "+str(ccg_tree)  # DEBUG
        parse_roots, parse_leaves_keys = self.form_root_from_leaves(parse_leaves, ccg_tree)
        if debug:
            print "parse_roots: "+str([self.print_parse(p.node) for p in parse_roots])  # DEBUG
            _ = raw_input()  # DEBUG

        # yield parse and total score (leaves+syntax) if structure matches
        # set of new lexical entries required is empty in this case
        if len(parse_roots) == 1 and parse_roots[0].node is not None:
            yield parse_roots[0], []  # the ParseNode root of the finished parse tree

        # if structure does not match and there are None nodes, perform top-down generation from
        # supervised root (if available) to fill in unknown semantic gaps given CKY structure of
        # the ccg_tree
        parse_leaves_nodes = [pl.node for pl in parse_roots]
        if sem_root is not None and None in parse_leaves_nodes:

            if debug:
                print "trying top-down parsing..."  # DEBUG
            top_down_chart = {}
            root_key = (0, 1)
            for entry in ccg_tree:
                if entry[0] == 0 and entry[1] > root_key[1]:
                    root_key = entry
            top_down_chart[root_key] = ParseNode.ParseNode(None, sem_root)

            topdown_tree_generator = self.get_most_likely_tree_from_root(top_down_chart[root_key],
                                                                         root_key,
                                                                         ccg_tree,
                                                                         parse_leaves_keys)
            for topdown_root, topdown_score in topdown_tree_generator:

                if debug:
                    print ("... generated topdown leaves:\n\t" +
                           '\n\t'.join([self.print_parse(p.node, True) for p in topdown_root.get_leaves()]) +
                           "\nfor parse roots:\n\t" +
                           '\n\t'.join([self.print_parse(p.node, True) for p in parse_roots]))
                    _ = raw_input()  # DEBUG

                new_lex_entries = []  # values (surface form str, sem node)
                topdown_leaves = topdown_root.get_leaves()
                if len(topdown_leaves) == len(parse_roots):  # possible match was found in reverse parsing
                    match = True
                    if debug:
                        print "...... cardinality match"
                    candidate_parse_leaves = parse_roots[:]
                    for idx in range(0, len(candidate_parse_leaves)):
                        if candidate_parse_leaves[idx].node is None:
                            sf = candidate_parse_leaves[idx].surface_form if \
                                type(candidate_parse_leaves[idx].surface_form) is str else \
                                self.lexicon.surface_forms[candidate_parse_leaves[idx].surface_form]
                            new_lex_entries.append([sf, topdown_leaves[idx].node])
                            candidate_parse_leaves[idx] = topdown_leaves[idx]
                            candidate_parse_leaves[idx].surface_form = sf
                            continue
                        if candidate_parse_leaves[idx].node.category != topdown_leaves[idx].node.category:
                            if debug:
                                print "...... category mismatch"
                            match = False
                            break
                        if not candidate_parse_leaves[idx].node.equal_allowing_commutativity(topdown_leaves[idx].node,
                                                                                             commutative_idxs=
                                                                                             self.commutative_idxs):
                            if debug:
                                print ("...... semantic mismatch; " +
                                       self.print_parse(candidate_parse_leaves[idx].node, True) + " != " +
                                       self.print_parse(topdown_leaves[idx].node, True))
                            match = False
                            break
                    if match:
                        if debug:
                            print "new_lex_entries: "  # DEBUG
                            for nle in new_lex_entries:  # DEBUG
                                print nle[0]+" :- "+self.print_parse(nle[1], show_category=True)  # DEBUG
                            print "parse_leaves_keys: " + str(parse_leaves_keys)  # DEBUG
                            print "ccg_tree: " + str(ccg_tree)  # DEBUG

                        # Create a new ccg tree corresponding to the candidate leaves to be filled.
                        # This basically lops off the bottom, already-parsed part of the CCG tree in favor of
                        # one whose spans cover the candidate leaves and the parse above those leaves is
                        # borrowed from the already-computed ccg tree.
                        ccg_tree_top = {}
                        o_to_top_m = {}
                        for idx in range(len(candidate_parse_leaves)):
                            span = (idx, idx + 1)
                            ccg_tree_top[span] = [candidate_parse_leaves[idx].node.category, None, None]
                        for idx in range(len(parse_leaves_keys)):
                            for jdx in parse_leaves_keys[idx]:
                                if jdx not in o_to_top_m:
                                    o_to_top_m[jdx] = len(o_to_top_m.keys())
                        for key in ccg_tree:
                            if key[0] in o_to_top_m and key[1] in o_to_top_m:
                                top_key = (o_to_top_m[key[0]], o_to_top_m[key[1]])
                                if top_key not in ccg_tree_top:
                                    ccg_tree_top[top_key] = [ccg_tree[key][0],
                                                             (o_to_top_m[ccg_tree[key][1][0]],
                                                              o_to_top_m[ccg_tree[key][1][1]]),
                                                             (o_to_top_m[ccg_tree[key][2][0]],
                                                              o_to_top_m[ccg_tree[key][2][1]])]

                        if debug:
                            print "o_to_top_m: " + str(o_to_top_m)  # DEBUG
                            print "candidate_parse_leaves:"  # DEBUG
                            for cpl in candidate_parse_leaves:  # DEBUG
                                print "\t" + self.print_parse(cpl.node, show_category=True)  # DEBUG
                            print "ccg_tree_top: " + str(ccg_tree_top)  # DEBUG

                        candidate_parse_leaves, candidate_leaf_spans = \
                            self.form_root_from_leaves(candidate_parse_leaves, ccg_tree_top)

                        if debug:
                            print "candidate_parse_leaves after parsing:"  # DEBUG
                            for cpl in candidate_parse_leaves:  # DEBUG
                                print "\t" + self.print_parse(cpl.node, show_category=True)  # DEBUG
                            print "candidate_leaf_spans: " + str(candidate_leaf_spans)  # DEBUG

                        # the ParseNode root of the finished parse tree
                        if len(candidate_parse_leaves) == 1:
                            yield candidate_parse_leaves[0], new_lex_entries

    # given parse leaves, form as close to root as possible
    def form_root_from_leaves(self, parse_leaves, ccg_tree):
        debug = False

        # try to build parse tree from lexical assignments guided by CKY structure
        spans = []
        for key in ccg_tree:
            if ccg_tree[key][1] is None and ccg_tree[key][2] is None:
                idx = 0
                for idx in range(0, len(spans)+1):  # insertion sort to get spans in linear order
                    if idx == len(spans) or key[0] < spans[idx][0]:
                        break
                spans.insert(idx, key)
        if debug:
            print "form_root_from_leaves spans: "+str(spans)  # DEBUG
        found_combination = True
        while len(parse_leaves) > 1 and found_combination:

            if debug:
                print "parse leaves:"
                for i in range(0, len(parse_leaves)):  # DEBUG
                    print str(i)+": " + self.print_parse(parse_leaves[i].node, show_category=True) \
                        if parse_leaves[i] is not None else str(None)  # DEBUG

            found_combination = False
            for i in range(0, len(parse_leaves)-1):
                root_span = (spans[i][0], spans[i+1][1])
                if root_span in ccg_tree:  # these two leaves must combine given tree

                    if debug:
                        print "investigating combination at "+str(root_span)  # DEBUG
                    if parse_leaves[i].node is None or parse_leaves[i+1].node is None:
                        continue

                    l = parse_leaves[i].node
                    r = parse_leaves[i+1].node
                    if (l in self.cached_combinations and
                       r in self.cached_combinations[l]):
                        root = self.cached_combinations[l][r]
                    else:
                        root = None
                        if self.can_perform_fa(i, i+1, l, r):
                            root = self.perform_fa(l, r)
                        elif self.can_perform_fa(i+1, i, r, l):
                            root = self.perform_fa(r, l)
                        elif self.can_perform_merge(l, r):
                            root = self.perform_merge(l, r)
                        elif self.can_perform_merge(r, l):
                            root = self.perform_merge(r, l)
                        if root is not None:
                            if i not in self.cached_combinations:
                                self.cached_combinations[i] = {}
                            self.cached_combinations[i][i+1] = root

                    if root is not None:
                        joined = ParseNode.ParseNode(None, root,
                                                     children=[parse_leaves[i], parse_leaves[i+1]])
                        parse_leaves[i] = joined
                        del parse_leaves[i+1]
                        spans[i] = (spans[i][0], spans[i+1][1])
                        del spans[i+1]
                        found_combination = True  # start iteration over since we modified list
                        if debug:
                            print "found combination at "+str(root_span)  # DEBUG
                        break
        return parse_leaves, spans

    # greedily yields the next most likely tree generated from given parse root
    # subject to the constraints of the ccg_tree and stopping upon reaching all known_leaf_keys
    def get_most_likely_tree_from_root(self, parse_root, root_key, ccg_tree, known_leaf_keys):
        debug = False

        if debug:
            print "get_most_likely_tree_from_root called"  # DEBUG
            print "root_key: "+str(root_key)  # DEBUG
            print "ccg_tree: "+str(ccg_tree)  # DEBUG

        # if root key is the only entry in the chart, can only associate it with the known parse root
        if len(ccg_tree.keys()) == 1:
            yield parse_root, 0
        else:

            # greedily take children with best score until we get a category match
            children_generator = self.get_most_likely_children_from_root(parse_root.node)
            for children, children_score in children_generator:
                if debug:
                    print ("... produced children " + self.print_parse(children[0], True) +
                           " ; " + self.print_parse(children[1], True) + " ; for target categories " +
                           self.lexicon.compose_str_from_category(ccg_tree[ccg_tree[root_key][1]][0]) + " and " +
                           self.lexicon.compose_str_from_category(ccg_tree[ccg_tree[root_key][2]][0]))
                # when we find children with syntax matching ccg tree, save and see whether either should expand
                if children[0].category == ccg_tree[ccg_tree[root_key][1]][0] and \
                   children[1].category == ccg_tree[ccg_tree[root_key][2]][0]:
                    if debug:
                        print "...category match"  # DEBUG
                    parse_root.children = []
                    for c in range(0, 2):  # save calculated semantic children wrapped in ParseNodes
                        parse_root.children.append(ParseNode.ParseNode(parse_root, children[c]))
                    if ccg_tree[root_key][1] in known_leaf_keys and ccg_tree[root_key][2] in known_leaf_keys:
                        if debug:
                            print "...get_most_likely_tree_from_root yielding two known leaf keys"  # DEBUG
                        yield parse_root, children_score
                    subtree_generators = [self.get_most_likely_tree_from_root(parse_root.children[c],
                                                                              ccg_tree[root_key][1+c],
                                                                              ccg_tree,
                                                                              known_leaf_keys)
                                          for c in range(0, 2)]
                    for c in range(0, 2):  # expand children
                        if ccg_tree[root_key][1+c] not in known_leaf_keys:
                            if debug:
                                print ("... expanding child in span not in leaves " +
                                       str(ccg_tree[root_key][1+c]))  # DEBUG
                            for child1, score1 in subtree_generators[c]:
                                parse_root.children[c] = child1
                                if ccg_tree[root_key][1+((c+1) % 2)] not in known_leaf_keys:
                                    for child2, score2 in subtree_generators[(c+1) % 2]:
                                        parse_root.children[(c+1) % 2] = child2
                                        if debug:
                                            print "...get_most_likely_tree_from_root yielding deeper children"  # DEBUG
                                            _ = raw_input()  # DEBUG
                                        yield parse_root, children_score+score1+score2
                                else:
                                    if debug:
                                        print "...get_most_likely_tree_from_root yielding deeper child"  # DEBUG
                                        _ = raw_input()  # DEBUG
                                    yield parse_root, children_score+score1
                        elif debug:
                            print ("... not expanding child in leaf span " +
                                   str(ccg_tree[root_key][1+c]))  # DEBUG

    # yields next most likely pair of children from a given semantic root using production rule parameter scores
    def get_most_likely_children_from_root(self, n):
        debug = False

        if debug:
            print "get_most_likely_children_from_root: called on " + self.print_parse(n, True)

        candidate_pairs = self.perform_reverse_fa(n)
        if self.can_perform_split(n):
            candidate_pairs.extend(self.perform_split(n))

        if debug:
            print ("get_most_likely_children_from_root: candidate pairs " +
                   str([self.print_parse(p1, True) + " dir " + str(d) + " " + self.print_parse(p2, True)
                        for p1, d, p2 in candidate_pairs]))

        match_scores = {}  # indexed by candidate_pair idx, value score
        for prod in self.theta.CCG_production:
            if prod[0] == n.category:
                for pair_idx in range(0, len(candidate_pairs)):
                    pair = candidate_pairs[pair_idx]
                    if ((pair[1] == 1 and prod[1] == pair[0].category and prod[2] == pair[2].category)
                       or (pair[1] == 0 and prod[1] == pair[2].category and prod[2] == pair[0].category)):
                        match_scores[pair_idx] = self.theta.CCG_production[prod]
        for pair_idx, score in sorted(match_scores.items(), key=operator.itemgetter(1), reverse=True):
            children = [candidate_pairs[pair_idx][0], candidate_pairs[pair_idx][2]] \
                if candidate_pairs[pair_idx][1] == 1 else \
                [candidate_pairs[pair_idx][2], candidate_pairs[pair_idx][0]]
            if debug:
                print ("get_most_likely_children_from_root: yielding children " +
                       self.print_parse(children[0], True) + ", " + self.print_parse(children[1], True))
                _ = raw_input()  # DEBUG
            yield children, score

    # yields next most likely assignment of semantic values to ccg tree leaves
    def most_likely_semantic_leaves(self, tks, ccg_tree, known_root=None):
        debug = False

        # get syntax categories for tree leaves
        leaf_categories = []
        curr_idx = 0
        spans = []
        while curr_idx < len(tks):
            for span in range(0, self.max_multiword_expression):
                key = (curr_idx, curr_idx+span+1)
                if key in ccg_tree:
                    leaf_categories.append(ccg_tree[key][0])
                    spans.append(key)
                    curr_idx += span+1
                    break
        if debug:
            print "most_likely_semantic_leaves: called for tks " + str(tks)  # DEBUG
            print "most_likely_semantic_leaves: with CCG tree " + str(ccg_tree)  # DEBUG
            print "most_likely_semantic_leaves: calculated spans " + str(spans)  # DEBUG
            _ = raw_input()  # DEBUG

        # get possible semantic forms for each syntax/surface combination represented by leaf_categories and tks
        semantic_candidates = []
        expressions = []
        for idx in range(0, len(spans)):
            exp = ' '.join([tks[t] for t in range(spans[idx][0], spans[idx][1])])
            expressions.append(exp)
            if exp in self.lexicon.surface_forms:
                semantic_candidates.append([sem_idx for sem_idx
                                            in self.lexicon.entries[self.lexicon.surface_forms.index(exp)]
                                            if self.lexicon.semantic_forms[sem_idx].category == leaf_categories[idx]])
            else:  # unknown surface form with no semantic neighbors
                # if the root is known, semantic candidates will be generated after the fact top-down
                if known_root is not None:
                    semantic_candidates.append([])
                # if the root isn't known, the best we can do is try all semantic forms that match the given
                # syntax; ie look for synonyms in the lexicon; search space is less restricted
                else:
                    semantic_candidates.append([sem_idx for sem_idx
                                                in range(0, len(self.lexicon.semantic_forms))
                                                if self.lexicon.semantic_forms[sem_idx].category
                                                == leaf_categories[idx]])
                    semantic_candidates[-1].append(None)

        # print "semantic_candidates: "+str(semantic_candidates)  # DEBUG

        # score all possible assignments of semantics to given surface forms and syntax categories
        scores = {}  # indexed by tuple of semantic form idxs
        i = 0
        finished = False
        while not finished:
            finished = True
            curr = i
            assignments = []
            score = 0
            for idx in range(0, len(spans)):
                if len(semantic_candidates[idx]) == 0:
                    assignments.append(None)  # no known semantic assignment
                else:
                    assignment_idx = curr % len(semantic_candidates[idx])
                    assignments.append(assignment_idx)
                    if expressions[idx] in self.lexicon.surface_forms:  # surface form is in lexicon
                        key = (semantic_candidates[idx][assignment_idx],
                               self.lexicon.surface_forms.index(expressions[idx]))
                        score += self.theta.lexicon_entry_given_token[key] \
                            if key in self.theta.lexicon_entry_given_token else \
                            self.theta.lexicon_entry_given_token[(semantic_candidates[idx][assignment_idx], -1)] +\
                            self.missing_lexicon_entry_given_token_penalty
                    else:  # surface form not in lexicon; we're just grasping at synonyms
                        # we can use an arbitrary semantic form idx since this is a uniform probability given
                        # unknown surface form
                        score += (self.theta.lexicon_entry_given_token[(0, -1)] +
                                  self.missing_lexicon_entry_given_token_penalty)
                    if assignment_idx < len(semantic_candidates[idx])-1:
                        finished = False
                    curr /= len(semantic_candidates[idx])
            i += 1
            scores[tuple(assignments)] = score

        # yield the assignment tuples in order by score as lists of ParseNodes
        for assignment, score in sorted(scores.items(), key=operator.itemgetter(1), reverse=True):
            nodes = [ParseNode.ParseNode(None,
                                         copy.deepcopy(self.lexicon.semantic_forms[
                                                       semantic_candidates[idx][assignment[idx]]])
                                         if assignment[idx] is not None and
                                         semantic_candidates[idx][assignment[idx]] is not None else None,
                                         surface_form=self.lexicon.surface_forms.index(expressions[idx])
                                         if expressions[idx] in self.lexicon.surface_forms else expressions[idx],
                                         semantic_form=semantic_candidates[idx][assignment[idx]]
                                         if assignment[idx] is not None and
                                         semantic_candidates[idx][assignment[idx]] is not None else None)
                     for idx in range(0, len(spans))]
            yield nodes, score

    # yields the next most likely ccg parse tree given a set of tokens
    def most_likely_ccg_parse_tree_given_tokens(self, tks, new_sense_leaf_limit=0):
        debug = False

        if debug:
            print "most_likely_ccg_parse_tree_given_tokens initialized with tks="+str(tks) + \
                ", new_sense_leaf_limit="+str(new_sense_leaf_limit)  # DEBUG
            _ = raw_input()  # DEBUG

        # for this at -1, assume all words that are defined in the lexicon take a value from the lexicon
        # for this at any idx, ignore that idx's lexicon definitions to allow top-down generation
        # do the subsequent part in randomized order such that if a sentence can't be parsed, we try again
        # allowing each word in turn to take on a new generation-based word sense
        randomized_idxs = range(len(tks))
        random.shuffle(randomized_idxs)
        randomized_idxs = [-1] + randomized_idxs
        missing_tried_so_far = 0
        for init_missing_idx in randomized_idxs:

            # indexed by position in span (i, j) in parse tree
            # value is list of tuples [CCG category, [left key, left index], [right key, right index], score]
            chart = {}

            # values are positions in span (i, j) in parse tree
            # value is present if this span is missing from the lexicon and thus, in the chart, eligible to
            # receive arbitrary matching syntax categories given neighbors during CKY
            missing = []

            # values are positions in span (i, j) in parse tree
            # value is present if span is present in lexicon but eligible to receive arbitrary matching syntax
            # categories given neighbors during CKY for purposes of multi-sense detection
            sense_leaf_keys = []

            # assign leaf values
            max_entries = [None, None]  # track which span has the most lexical entries
            skip_this_missing_set = False
            for span in range(0, self.max_multiword_expression):
                for idx in range(0, len(tks)-span):
                    pos = (idx, idx+span+1)
                    chart[pos] = []
                    exp = ' '.join([tks[t] for t in range(pos[0], pos[1])])
                    if exp in self.lexicon.surface_forms and idx != init_missing_idx:
                        sf_idx = self.lexicon.surface_forms.index(exp)
                        cats = [self.lexicon.semantic_forms[sem_idx].category
                                for sem_idx in self.lexicon.entries[sf_idx]]
                        for cat_idx in cats:
                            score = self.theta.CCG_given_token[(cat_idx, sf_idx)] \
                                if (cat_idx, sf_idx) in self.theta.CCG_given_token \
                                else self.theta.CCG_given_token[(cat_idx, -1)] + self.missing_CCG_given_token_penalty
                            score *= span  # NEW
                            chart[pos].append([cat_idx, None, None, score])
                        if max_entries[1] is None or max_entries[1] < len(self.lexicon.entries[sf_idx]):
                            max_entries[0] = pos
                            max_entries[1] = len(self.lexicon.entries[sf_idx])
                    else:
                        missing.append(pos)
                        if idx == init_missing_idx:
                            if exp in self.lexicon.surface_forms:
                                missing_tried_so_far += 1
                            else:
                                # skipping this idx is useless because we don't have entries for it
                                # so we already tried skipping it previously
                                skip_this_missing_set = True
                                break
            if skip_this_missing_set:
                continue

            # assign sense leaves based on max entries
            for i in range(0, new_sense_leaf_limit):
                if max_entries[0] is not None:
                    sense_leaf_keys.append(max_entries[0])
                max_entries = [None, None]
                for span in range(1, self.max_multiword_expression):
                    for idx in range(0, len(tks)-span):
                        pos = (idx, idx+span+1)
                        exp = ' '.join([tks[t] for t in range(pos[0], pos[1])])
                        if pos not in missing:
                            sf_idx = self.lexicon.surface_forms.index(exp)
                            if (pos not in sense_leaf_keys and
                                    (max_entries[0] is not None or max_entries[1] < len(self.lexicon.entries[sf_idx]))):
                                max_entries[0] = pos
                                max_entries[1] = len(self.lexicon.entries[sf_idx])

            if debug:
                print "init_missing_idx: " + str(init_missing_idx)
                print "missing_tried_so_far: " + str(missing_tried_so_far)
                print "leaf chart: " + str(chart)
                print "leaf missing: " + str(missing)
                _ = raw_input()  # DEBUG

            # populate chart for length 1 utterance
            if len(tks) == 1:
                # chart entries should be all top-level CCG categories (those that don't take arguments)
                pos = (0, 1)
                for cat_idx in range(0, len(self.lexicon.categories)):
                    if type(self.lexicon.categories[cat_idx]) is str:
                        score = self.theta.CCG_given_token[(cat_idx, -1)] + self.missing_CCG_given_token_penalty
                        chart[pos].append([cat_idx, None, None, score])

            # populate chart using CKY
            for width in range(2, len(tks)+1):
                # print "width: "+str(width)  # DEBUG
                for start in range(0, len(tks)-width+1):
                    end = start + width
                    key = (start, end)
                    if key not in chart:
                        chart[key] = []
                    # print "key: "+str(key)  # DEBUG
                    for mid in range(start+1, end):
                        l_key = (start, mid)
                        r_key = (mid, end)
                        left = chart[l_key] if l_key in chart else []
                        right = chart[r_key] if r_key in chart else []
                        # print "l_key: "+str(l_key)  # DEBUG
                        # print "r_key: "+str(r_key)  # DEBUG
                        for l_idx in range(0, len(left)):
                            l = left[l_idx]
                            # if debug_chr == 'y' : # DEBUG
                            #     print "l: "+str(l)+", cat="+self.lexicon.compose_str_from_category(l[0])  # DEBUG
                            for r_idx in range(0, len(right)):
                                r = right[r_idx]
                                # if debug_chr == 'y' : # DEBUG
                                #     print "r: "+str(r)+", cat="+self.lexicon.compose_str_from_category(r[0])  # DEBUG
                                for prod in self.theta.CCG_production:
                                    # if debug_chr == 'y' : # DEBUG
                                    #     print "prod: "+str([self.lexicon.compose_str_from_category(c)
                                    #  for c in prod])  # DEBUG
                                    if prod[1] == l[0] and prod[2] == r[0]:
                                        new_score = self.theta.CCG_production[prod] + l[3] + r[3]
                                        # Find all expansions of this non-terminal in the cell
                                        expansions = [expansion for expansion in chart[key] if expansion[0] == prod[0]]
                                        if len(expansions) < self.max_expansions_per_non_terminal - 1:
                                            chart[key].append([prod[0], [l_key, l_idx], [r_key, r_idx], new_score])
                                        else:
                                            # Sort expansions in order of score
                                            scores_and_expansions = [(expansion[-1], expansion)
                                                                     for expansion in expansions]
                                            scores_and_expansions.sort()
                                            if new_score > scores_and_expansions[0][0]:
                                                # Remove the least scoring existing expansion and
                                                # add the one just found
                                                chart[key].remove(scores_and_expansions[0][1])
                                                chart[key].append([prod[0], [l_key, l_idx], [r_key, r_idx], new_score])

                                        # print "new chart entry "+str(key)+" : "+str(chart[key][-1])  # DEBUG
                        lr_keys = [l_key, r_key]
                        lr_dirs = [left, right]
                        lr_idxs = [1, 2]
                        for branch_idx in range(0, 2):
                            m_idx = branch_idx
                            p_idx = 0 if branch_idx == 1 else 1
                            if ((lr_keys[m_idx] in sense_leaf_keys or lr_keys[m_idx] in missing)
                                    and lr_keys[p_idx] not in missing):
                                # print "searching for matches for missing "+str(lr_keys[m_idx]) + \
                                #       " against present "+str(lr_keys[p_idx])  # DEBUG
                                # investigate syntax for leaf m that can consume or be consumed by p
                                # because of rewriting the CCG rules in CNF form, if a production rule is present
                                # it means that a consumation or merge can happen and we can be agnostic to how
                                for idx in range(0, len(lr_dirs[p_idx])):
                                    p = lr_dirs[p_idx][idx]
                                    hypothesis_categories_added = 0
                                    ccg_productions = self.theta.CCG_production.keys()[:]
                                    random.shuffle(ccg_productions)  # could weight random by liklihood
                                    for prod in ccg_productions:
                                        # leaf m can be be combine with p
                                        if prod[lr_idxs[p_idx]] == p[0]:
                                            # give probability of unknown token assigned to this CCG category
                                            m_score = (self.theta.CCG_given_token[(prod[lr_idxs[m_idx]], -1)] +
                                                       self.missing_CCG_given_token_penalty)
                                            m = [prod[lr_idxs[m_idx]], None, None, m_score]
                                            chart[lr_keys[m_idx]].append(m)
                                            new_score = self.theta.CCG_production[prod] + p[3] + m[3]
                                            new_entry = [prod[0], None, None, new_score]
                                            new_entry[lr_idxs[m_idx]] = [lr_keys[m_idx], len(chart[lr_keys[m_idx]])-1]
                                            new_entry[lr_idxs[p_idx]] = [lr_keys[p_idx], idx]
                                            chart[key].append(new_entry)
                                            # print "new chart entry "+str(key)+" : "+str(chart[key][-1])  # DEBUG
                                            hypothesis_categories_added += 1
                                            if hypothesis_categories_added == \
                                                    self.max_hypothesis_categories_for_unknown_token_beam:
                                                break

                        if debug:
                            print "chart: "+str(chart)  # DEBUG
                            for key in chart:
                                print (str(key) + " " + str([_t for _t in tks[key[0]:key[1]]]) +
                                       ":\n\t" + '\n\t'.join(
                                    [self.lexicon.compose_str_from_category(chart[key][idx][0])
                                     + " " + str(chart[key][idx][1]) + " " + str(chart[key][idx][2]) for idx in
                                     range(len(chart[key]))]))
                            _ = raw_input()

            if debug:
                print "finished chart: "+str(chart)  # DEBUG

            # weight trees using prior on root CCG node
            key = (0, len(tks))
            for i in range(0, len(chart[key])):
                chart[key][i][3] += (self.theta.CCG_root[chart[key][i][0]]
                                     if chart[key][i][0] in self.theta.CCG_root
                                     else self.theta.CCG_root[-1])

            # return most likely trees in order at root
            if debug:
                print "\tnumber of roots to yield: "+str(len(chart[key]))  # DEBUG
            roots_yielded = 0
            while len(chart[key]) > 0 and roots_yielded < self.max_cky_trees_per_token_sequence_beam:

                # find and yield best root
                best_idx = 0
                for i in range(1, len(chart[key])):
                    if chart[key][i][3] > chart[key][best_idx][3]:
                        best_idx = i
                best = chart[key][best_idx][:]

                # build and return tree from this root
                tree = {}  # indexed as spans (i, j) like chart, valued at [prod, left span, right span, score]
                to_add = [[key, best]]
                while len(to_add) > 0:
                    new_key, to_expand = to_add.pop()
                    tree_add = [to_expand[0]]
                    for i in range(1, 3):
                        if to_expand[i] is not None:
                            tree_add.append(to_expand[i][0])
                            to_add.append([to_expand[i][0], chart[to_expand[i][0]][to_expand[i][1]]])
                        else:
                            tree_add.append(None)
                    tree[new_key] = tree_add

                # yield the current tree and remove the previous root from the structure
                if debug:
                    print ("most_likely_ccg_parse_tree_given_tokens with tks="+str(tks) +
                           ", new_sense_leaf_limit=" + str(new_sense_leaf_limit)+", score=" +
                           str(chart[key][best_idx][3]) + " yielding tree " + str(tree))  # DEBUG
                yield tree, chart[key][best_idx][3]  # return tree and score
                del chart[key][best_idx]
                roots_yielded += 1

            if roots_yielded == self.max_cky_trees_per_token_sequence_beam:  # DEBUG
                if debug:
                    print "WARNING: beam search limit hit"  # DEBUG
                pass

            # stop repeating missing mask if we've hit the limit
            if missing_tried_so_far == self.max_missing_words_to_try:
                break

        # no parses left
        yield None, neg_inf

    # return A<>B; A and B must have matching lambda headers and syntactic categories to be AND merged
    def perform_merge(self, a, b):
        debug = False

        if debug:
            print "performing Merge with '"+self.print_parse(a, True)+"' taking '"+self.print_parse(b, True)+"'"

        and_idx = self.ontology.preds.index('and')

        if a.is_lambda_instantiation:
            ab = copy.deepcopy(a)
            ab.set_category(a.category)
            innermost_outer_lambda = ab
            a_child = a.children[0]
            b_child = b.children[0]
            while (innermost_outer_lambda.children is not None and innermost_outer_lambda.children[0].is_lambda
                   and innermost_outer_lambda.children[0].is_lambda_instantiation):
                innermost_outer_lambda = innermost_outer_lambda.children[0]
                a_child = a.children[0]
                b_child = b.children[0]
            innermost_outer_lambda.children = [
                SemanticNode.SemanticNode(innermost_outer_lambda, self.ontology.entries[and_idx],
                                          innermost_outer_lambda.children[0].category, False, idx=and_idx)]
            innermost_outer_lambda.children[0].children = [copy.deepcopy(a_child), copy.deepcopy(b_child)]

            # 'and' adopts type taking each child's and returning the same
            a_child.set_return_type(self.ontology)
            a_child.set_return_type(self.ontology)
            input_type = [a_child.return_type, a_child.return_type]
            if input_type not in self.ontology.types:
                self.ontology.types.append(input_type)
            full_type = [a_child.return_type, self.ontology.types.index(input_type)]
            if full_type not in self.ontology.types:
                self.ontology.types.append(full_type)
            innermost_outer_lambda.children[0].type = self.ontology.types.index(full_type)
            innermost_outer_lambda.children[0].set_return_type(self.ontology)
            innermost_outer_lambda.children[0].children[0].parent = innermost_outer_lambda.children[0]
            innermost_outer_lambda.children[0].children[1].parent = innermost_outer_lambda.children[0]

            ab.set_return_type(self.ontology)
            ab.commutative_raise_node(self.commutative_idxs, self.ontology)

            # Check whether we've duplicated lambda instantiations among children and remove one if so.
            # This seems kind of kludgy but should handle adjectives being merged.
            if len(ab.children) == 1 and ab.children[0].idx == and_idx:
                if [c.lambda_name for c in ab.children[0].children].count(ab.lambda_name) > 1:
                    for cidx in range(len(ab.children[0].children)):
                        if ab.children[0].children[cidx].lambda_name == ab.lambda_name:
                            del ab.children[0].children[cidx]
                            break
            ab.set_return_type(self.ontology)
        else:
            try:
                a.set_return_type(self.ontology)
                b.set_return_type(self.ontology)
            except TypeError:
                raise TypeError("Non-matching child/parent relationship for one of two nodes " +
                                self.print_parse(a, True) + " , " + self.print_parse(b, True))
            if a.return_type != b.return_type:
                raise RuntimeError("performing Merge with '"+self.print_parse(a, True) +
                                   "' taking '"+self.print_parse(a, True) +
                                   "' generated mismatched return types" +
                                   self.ontology.compose_str_from_type(a.return_type)+"," +
                                   self.ontology.compose_str_from_type(b.return_type))
            input_type = [a.return_type, a.return_type]
            if input_type not in self.ontology.types:
                self.ontology.types.append(input_type)
            full_type = [a.return_type, self.ontology.types.index(input_type)]
            if full_type not in self.ontology.types:
                self.ontology.types.append(full_type)
            ab = SemanticNode.SemanticNode(None, self.ontology.types.index(full_type),
                                           a.category, False, idx=and_idx)
            ab.children = [copy.deepcopy(a), copy.deepcopy(b)]
            ab.children[0].parent = ab
            ab.children[1].parent = ab

            ab.set_return_type(self.ontology)
            ab.commutative_raise_node(self.commutative_idxs, self.ontology)

        if debug:
            print "performed Merge with '"+self.print_parse(a, True)+"' taking '"+self.print_parse(b, True) + \
                "' to form '"+self.print_parse(ab, True)+"'"  # DEBUG
        if self.safety and not ab.validate_tree_structure():
            raise RuntimeError("ERROR: invalidly linked structure generated by FA: " +
                               self.print_parse(ab, True))
        return ab

    # return true if A,B can be merged
    def can_perform_merge(self, a, b):
        if not self.allow_merge:
            return False
        if a is None or b is None:
            return False
        if self.lexicon.categories[a.category] != self.lexicon.categories[b.category]:
            return False
        if a.return_type is None:
            a.set_return_type(self.ontology)
        if b.return_type is None:
            b.set_return_type(self.ontology)
        if a.return_type != b.return_type:
            return False
        curr_a = a
        curr_b = b
        while curr_a.is_lambda and curr_a.is_lambda_instantiation:
            if curr_a.return_type is None:
                curr_a.set_return_type(self.ontology)
            if curr_b.return_type is None:
                curr_b.set_return_type(self.ontology)
            if (not curr_b.is_lambda or not curr_b.is_lambda_instantiation or curr_b.type != curr_a.type
                    or curr_a.return_type != curr_b.return_type):
                return False
            curr_a = curr_a.children[0]
            curr_b = curr_b.children[0]
        return True

    # return A(B); A must be lambda headed with type equal to B's root type
    def perform_fa(self, a, b, renumerate=True):
        debug = False
        # if debug:  # DEBUG
        #     _ = raw_input()  # DEBUG

        if self.safety:
            if not a.validate_tree_structure():  # DEBUG
                raise RuntimeError("WARNING: got invalidly linked node '"+self.print_parse(a)+"'")
            if not b.validate_tree_structure():  # DEBUG
                raise RuntimeError("WARNING: got invalidly linked node '"+self.print_parse(b)+"'")
        if debug:
            print "performing FA with '"+self.print_parse(a, True)+"' taking '"+self.print_parse(b, True)+"'"  # DEBUG

        # if A is 'and', apply B to children
        if not a.is_lambda and self.ontology.preds[a.idx] == 'and':
            a_new = copy.deepcopy(a)
            for i in range(0, len(a.children)):
                c = a_new.children[i]
                c_obj = copy.deepcopy(self.lexicon.semantic_forms[c]) if type(c) is int else c
                copy_obj = SemanticNode.SemanticNode(None, None, None, False, 0)
                copy_obj.copy_attributes(c_obj, preserve_parent=True)
                if (copy_obj in self.lexicon.semantic_forms and
                        self.lexicon.semantic_forms.index(copy_obj) in self.type_raised):
                    copy_obj.copy_attributes(self.lexicon.semantic_forms[
                                             self.type_raised[self.lexicon.semantic_forms.index(copy_obj)]],
                                             preserve_parent=True)
                a_new.children[i] = self.perform_fa(copy_obj, b, renumerate=False)
                a_new.children[i].set_return_type(self.ontology)
                a_new.children[i].parent = a_new
            a_new.set_type_from_children_return_types(a_new.children[0].return_type, self.ontology)
            a_new.set_return_type(self.ontology)
            a_new.set_category(a_new.children[0].category)
            a_new.commutative_raise_node(self.commutative_idxs, self.ontology)
            if debug:
                print "performed FA(1) with '"+self.print_parse(a, True)+"' taking '"+self.print_parse(b, True) + \
                    "' to form '"+self.print_parse(a_new, True)+"'"  # DEBUG
            if self.safety and not a_new.validate_tree_structure():
                raise RuntimeError("ERROR: invalidly linked structure generated by FA: " +
                                   self.print_parse(a_new, True))
            return a_new

        # if A is not 'and' but also not lambda-headed, we need to implicitly attach the child to all of a
        # thus, we need to give ab a lambda child for b to become
        if not a.is_lambda and a.children is None and b.is_lambda:
            ab = copy.deepcopy(a)
            ab.children = [SemanticNode.SemanticNode(ab, b.return_type, b.category,
                                                     True, lambda_name=b.lambda_name,
                                                     is_lambda_instantiation=False)]
            if debug:
                print "performed FA(3) with '"+self.print_parse(a, True)+"' taking '"+self.print_parse(b, True) + \
                    "' to form '"+self.print_parse(ab, True)+"'"  # DEBUG
            return ab

        # If A is lambda headed, it has a single child which will be the root of the composed tree
        ab = copy.deepcopy(a.children[0])
        ab.parent = None
        # traverse A_FA_B and replace references to lambda_A with B
        ab_deepest_lambda = a.lambda_name
        to_traverse = [[ab, ab_deepest_lambda]]
        while len(to_traverse) > 0:
            [curr, deepest_lambda] = to_traverse.pop()
            entire_replacement = False
            if curr.is_lambda and curr.is_lambda_instantiation:
                if debug:
                    print "detected deeper lambda "+str(curr.lambda_name)  # DEBUG
                deepest_lambda = curr.lambda_name
            # an instance of lambda_A to be replaced by B
            elif curr.is_lambda and not curr.is_lambda_instantiation and curr.lambda_name == a.lambda_name:
                if debug:
                    print "substituting '"+self.print_parse(b, True)+"' for '"+self.print_parse(curr, True) + \
                        "' with lambda offset "+str(deepest_lambda)  # DEBUG
                if (not b.is_lambda and self.ontology.preds[b.idx] == 'and'
                        and curr.children is not None and b.children is not None):
                    if debug:
                        print "entering B substitution of curr taking curr's args"  # DEBUG
                    # if B is 'and', can preserve it and interleave A's children as arguments
                    raised = False
                    b_new = copy.deepcopy(b)
                    for i in range(0, len(b_new.children)):
                        c = b_new.children[i]
                        c.parent = None
                        c_obj = copy.deepcopy(self.lexicon.semantic_forms[c]) if type(c) is int else c
                        if (c_obj in self.lexicon.semantic_forms and
                                self.lexicon.semantic_forms.index(c_obj) in self.type_raised):
                            c_obj.copy_attributes(self.lexicon.semantic_forms[
                                                  self.type_raised[self.lexicon.semantic_forms.index(c_obj)]])
                            raised = True
                        b_new.children[i] = self.perform_fa(c_obj, curr.children[0], renumerate=False)
                        if b.idx != self.ontology.preds.index('and'):
                            # don't increment if special FA(3) invoked
                            # ^ this rule is new as of adding 'and' special rules and may in general break something
                            # so, watch out!
                            b_new.children[i].increment_lambdas(inc=deepest_lambda)
                        b_new.children[i].parent = curr
                        b_new.children[i].set_return_type(self.ontology)
                    if curr.parent.is_lambda_instantiation:  # eg. 1(2) taking and(pred,pred) -> and(pred(2),pred(2)
                        b_new_arg = b_new.children[0]
                        while self.ontology.preds[b_new_arg.idx] == 'and':
                            if debug:
                                print "B_new_arg: "+self.print_parse(b_new_arg)  # DEBUG
                            b_new_arg = b_new_arg.children[0]
                        if debug:
                            print "setting curr parent lambda name to child of "+self.print_parse(b_new_arg)  # DEBUG
                        curr.parent.lambda_name = b_new_arg.children[0].lambda_name
                    if raised:
                        b_new.set_category(self.lexicon.categories.index(
                            [self.lexicon.categories.index('N'), 1, self.lexicon.categories.index('N')]))
                    curr.copy_attributes(b_new, preserve_parent=True)
                    curr.set_type_from_children_return_types(curr.children[0].return_type, self.ontology)
                    curr.set_return_type(self.ontology)
                    if debug:
                        print "created 'and' consumption result "+self.print_parse(curr, True)  # DEBUG
                elif curr.parent is None:
                    if debug:
                        print "entering None parent for curr"  # DEBUG
                    if curr.children is None:
                        if debug:
                            print "...whole tree is instance"  # DEBUG
                        curr.copy_attributes(b)  # instance is whole tree; add nothing more and loop will now exit
                    elif b.children is None:
                        if debug:
                            print "...instance heads tree; preserve children taking B"
                        curr.copy_attributes(b, deepest_lambda, preserve_children=True)
                    else:
                        raise RuntimeError("Error: incompatible parentless, childed node A with childed node B")
                    entire_replacement = True
                    curr.set_category(self.lexicon.categories[a.category][0])  # take on return type of A
                    curr.set_return_type(self.ontology)
                else:
                    if debug:
                        print "entering standard implementation for curr"  # DEBUG
                    for curr_parent_matching_idx in range(0, len(curr.parent.children)):
                        if not curr.parent.children[curr_parent_matching_idx] != curr:  # find matching address
                            break
                    if curr.children is None:
                        if debug:
                            print "...instance of B ("+self.print_parse(b)+") will preserve its children"  # DEBUG
                        # lambda instance is a leaf
                        curr.parent.children[curr_parent_matching_idx].copy_attributes(b, deepest_lambda,
                                                                                       preserve_parent=True)
                        if not curr.parent.children[curr_parent_matching_idx].validate_tree_structure():  # DEBUG
                            raise RuntimeError("ERROR: copy operation produced invalidly linked tree " +
                                               self.print_parse(curr.parent.children[curr_parent_matching_idx], True))
                    else:
                        if b.children is None:
                            if debug:
                                print "...instance of B will keep children from A"  # DEBUG
                            curr.parent.children[curr_parent_matching_idx].copy_attributes(b, deepest_lambda,
                                                                                           preserve_parent=True,
                                                                                           preserve_children=True)
                        else:
                            if debug:
                                print "...instance of A and B have matching lambda headers to be merged"  # DEBUG
                            b_without_lambda_headers = b
                            lambda_types = {}
                            while (b_without_lambda_headers.is_lambda and
                                    b_without_lambda_headers.is_lambda_instantiation):
                                lambda_types[b_without_lambda_headers.lambda_name] = b_without_lambda_headers.type
                                b_without_lambda_headers = b_without_lambda_headers.children[0]
                            a_trace = a
                            lambda_map = {}
                            while len(lambda_types.keys()) > 0:
                                name_found = None
                                for name in lambda_types:
                                    if lambda_types[name] == a_trace.type:
                                        lambda_map[name] = a_trace.lambda_name
                                        name_found = name
                                        break
                                if name_found is not None:
                                    del lambda_types[name_found]
                                a_trace = a_trace.children[0]
                            if debug:
                                print "lambda_map: "+str(lambda_map)  # DEBUG
                            curr.parent.children[curr_parent_matching_idx].copy_attributes(b_without_lambda_headers,
                                                                                           lambda_map=lambda_map,
                                                                                           preserve_parent=True,
                                                                                           preserve_children=False)
                    curr.parent.children[curr_parent_matching_idx].set_return_type(self.ontology)
                    if debug:
                        print "substitution created "+self.print_parse(curr, True)  # DEBUG
            if not entire_replacement and curr.children is not None:
                to_traverse.extend([[c, deepest_lambda] for c in curr.children])
        if renumerate:
            if debug:
                print "renumerating result '"+self.print_parse(ab)+"'"  # DEBUG
            ab.renumerate_lambdas([])
        try:
            ab.set_return_type(self.ontology)
        except TypeError as e:
            raise e
        ab.set_category(self.lexicon.categories[a.category][0])
        ab.commutative_raise_node(self.commutative_idxs, self.ontology)
        if debug:
            print "performed FA(2) with '"+self.print_parse(a, True)+"' taking '"+self.print_parse(b, True) + \
                "' to form '"+self.print_parse(ab, True)+"'"  # DEBUG
        if self.safety and not ab.validate_tree_structure():
            raise RuntimeError("ERROR: invalidly linked structure generated by FA: " +
                               self.print_parse(ab, True))
        return ab

    # return true if A(B) is a valid for functional application
    def can_perform_fa(self, i, j, a, b):
        debug = False
        if debug:
            print "can_perform_fa: considering " + str(self.print_parse(a)) + ", " + str(self.print_parse(b))

        if a is None or b is None:
            if debug:
                print "A or B is None"  # DEBUG
            return False
        if a.category is None or type(self.lexicon.categories[a.category]) is not list or (
                i - j > 0 and self.lexicon.categories[a.category][1] == 1) or (
                i - j < 0 and self.lexicon.categories[a.category][1] == 0):
            if debug:
                print "A consumes nothing or B is left/right when A expects right/left"  # DEBUG
            return False  # B is left/right when A expects right/left
        if not a.is_lambda or not a.is_lambda_instantiation or a.type != b.return_type:
            if a.is_lambda_instantiation and a.children is None:  # DEBUG
                raise RuntimeError("ERROR: found lambda with no children: "+str(self.print_parse(a)))
            if debug:
                print "A is not lambda instantiation or types are mismatched"
            return False
        if self.lexicon.categories[a.category][2] != b.category:
            if debug:
                print "B category does not match A expected consumption"  # DEBUG
            return False  # B is not the input category A expects
        if a.parent is None and a.is_lambda and not a.is_lambda_instantiation:
            return True  # the whole tree of A will be replaced with the whole tree of B
        to_traverse = [b]
        b_lambda_context = []
        while len(to_traverse) > 0:
            curr = to_traverse.pop()
            if curr.is_lambda and curr.is_lambda_instantiation:
                b_lambda_context.append(curr.type)
            else:
                break
            to_traverse.extend(b.children)
        # return True if all instances of A lambda appear in lambda contexts identical to what B expects
        return self.lambda_value_replacements_valid(a.children[0], a.lambda_name, [], b,
                                                    b_lambda_context)

    def lambda_value_replacements_valid(self, a, lambda_name, a_lambda_context, b, b_lambda_context):
        # print "checking whether '"+self.print_parse(A)+"' lambda "+str(lambda_name) + \
        #       " instances can be replaced by '"+self.print_parse(B)+"' under contexts " + \
        #       str(A_lambda_context)+","+str(B_lambda_context)  # DEBUG
        if a.is_lambda and a.is_lambda_instantiation:
            extended_context = a_lambda_context[:]
            extended_context.append(a.type)
            return self.lambda_value_replacements_valid(a.children[0], lambda_name, extended_context,
                                                        b, b_lambda_context)
        # this is an instance of A's lambda to be replaced by B
        if a.is_lambda and not a.is_lambda_instantiation and a.lambda_name == lambda_name:
            if a.children is not None and b.children is not None:  # the instance takes arguments
                if (len(a_lambda_context) - len(b_lambda_context) == 1 and
                   b.idx == self.ontology.preds.index('and')):
                    return True  # special 'and' consumption rule (A consumes B 'and' and B takes A's arguments)
                if len(a_lambda_context) != len(b_lambda_context):
                    # print "contexts differ in length"  # DEBUG
                    return False  # the lambda contexts differ in length
                matches = [1 if a_lambda_context[i] == b_lambda_context[i] else 0 for i in
                           range(0, len(a_lambda_context))]
                if sum(matches) != len(a_lambda_context):
                    # print "contexts differ in content"  # DEBUG
                    return False  # the lambda contexts differ in content
                return True
            return True  # ie A, B have no children
        if a.children is None:
            return True
        valid_through_children = True
        for c in a.children:
            valid_through_children = self.lambda_value_replacements_valid(
                c, lambda_name, a_lambda_context, b, b_lambda_context)
            if not valid_through_children:
                break
        return valid_through_children

    # return A,B from A<>B; A<>B must be AND headed and its lambda headers will be distributed to A, B
    def perform_split(self, ab):
        # print "performing Split with '"+self.print_parse(ab, True)+"'" #DEBUG

        curr = ab
        while curr.is_lambda and curr.is_lambda_instantiation:
            curr = curr.children[0]  # first non-lambda must be 'and' predicate
        to_return = []
        for idx in range(0, 2):
            to_return.append(copy.deepcopy(ab))
            curr_t = to_return[-1]
            parent = None
            while curr_t.is_lambda and curr_t.is_lambda_instantiation:
                parent = curr_t
                curr_t = curr_t.children[0]
            if parent is not None:
                parent.children = [copy.deepcopy(curr.children[idx])]
                for c in parent.children:
                    c.parent = parent
            else:
                to_return[-1] = copy.deepcopy(curr.children[idx])
            to_return[-1].set_category(ab.category)
            to_return[-1].set_return_type(self.ontology)

        # print "performed Split with '"+self.print_parse(ab, True)+"' to form '" + \
        #     self.print_parse(to_return[0], True)+"', '"+self.print_parse(to_return[1], True)+"'"  # DEBUG
        candidate_pairs = [[to_return[0], to_return[1]], [copy.deepcopy(to_return[1]), copy.deepcopy(to_return[0])]]
        if self.safety:
            for idx in range(0, len(candidate_pairs)):
                for jdx in range(0, len(candidate_pairs[idx])):
                    if not candidate_pairs[idx][jdx].validate_tree_structure():
                        raise RuntimeError("ERROR: invalidly linked structure generated by split: " +
                                           self.print_parse(to_return[-1], True))
        return candidate_pairs

    # return true if AB can be split
    def can_perform_split(self, ab):
        if not self.allow_merge:
            return False
        if ab is None:
            return False
        curr = ab
        while curr.is_lambda and curr.is_lambda_instantiation:
            curr = curr.children[0]
        if curr.is_lambda or curr.children is None or curr.idx != self.ontology.preds.index('and'):
            return False
        return True

    # given A1(A2), attempt to determine an A1, A2 that satisfy and return them
    def perform_reverse_fa(self, a):
        debug = False
        if debug:  # DEBUG
            _ = raw_input()  # DEBUG

        consumables = self.lexicon.category_consumes[a.category]
        if len(consumables) == 0:
            return []
        if debug:
            print "performing reverse FA with '"+self.print_parse(a, True)+"'"  # DEBUG

        # for every predicate p in A of type q, generate candidates:
        # A1 = A with a new outermost lambda of type q, p replaced by an instance of q
        # A2 = p with children stripped
        # if p alone has no unbound variables, generate additional candidates:
        # A1 = A with new outermost lambda of type q return type p, with p and children replaced by q
        # A2 = p with children preserved

        # calculate largest lambda in form
        to_examine = [a]
        deepest_lambda = 0
        while len(to_examine) > 0:
            curr = to_examine.pop()
            if curr.is_lambda_instantiation and curr.lambda_name > deepest_lambda:
                deepest_lambda = curr.lambda_name
            if curr.children is not None:
                to_examine.extend(curr.children)

        candidate_pairs = []
        to_examine = [a]
        while len(to_examine) > 0:
            curr = to_examine.pop()
            if curr.children is not None:
                to_examine.extend(curr.children)
            if curr.is_lambda:
                continue
            pred = curr

            # create A1, A2 with A2 the predicate without children, A1 abstracting A2 instances
            pairs_to_create = [[True, False, False]]

            # check whether pred is 'and' and children arguments match, in which case additional abstraction
            # is possible
            add_and_abstraction = False
            if pred.idx == self.ontology.preds.index('and') and pred.children is not None:
                arg_children_match = True
                if debug:
                    print "considering for 'and' abstraction '"+self.print_parse(pred, True)+"'"  # DEBUG
                if (pred.children[0].children is not None and pred.children[1].children is not None and
                        len(pred.children[0].children) == len(pred.children[1].children)):
                    ac_to_examine = [[pred.children[0].children[ac_idx],
                                      pred.children[1].children[ac_idx]] for
                                     ac_idx in range(0, len(pred.children[0].children))]
                    while len(ac_to_examine) > 0:
                        c1, c2 = ac_to_examine.pop()
                        if not c1.equal_ignoring_syntax(c2):
                            arg_children_match = False
                            break
                        if c1.children is not None:
                            ac_to_examine.extend([[c1.children[ac_idx], c2.children[ac_idx]]
                                                 for ac_idx in range(0, len(c1.children))])
                else:
                    arg_children_match = False
                if arg_children_match:
                    add_and_abstraction = True
                if debug:
                    print "add_and_abstraction decision: " + str(add_and_abstraction)

            # create A1, A2 with A2 the predicate and children preserved, A1 abstracting A2 return type
            if pred.children is not None:
                unbound_vars_in_pred = False
                bound_lambda_in_pred = []
                check_bound = [pred]
                while len(check_bound) > 0:
                    curr = check_bound.pop()
                    if curr.is_lambda:
                        if curr.is_lambda_instantiation:
                            bound_lambda_in_pred.append(curr.lambda_name)
                        elif curr.lambda_name not in bound_lambda_in_pred:
                            unbound_vars_in_pred = True
                            break
                    if curr.children is not None:
                        check_bound.extend(curr.children)
                if not unbound_vars_in_pred:
                    pred.set_return_type(self.ontology)
                    pairs_to_create.append([False, add_and_abstraction, False])

                # detect special case of 'and' children taking lambda instantiations as unary args themselves
                # e.g. for a(x.(P)) taking and(p,q) -> a(x.(and(p(x),q(x))), a special case of FA distributing x
                elif pred.idx == self.ontology.preds.index('and') and pred.children is not None:
                    if debug:
                        print "'and' abstraction has unbound_vars_in_pred=True for " + self.print_parse(pred, True)
                    if pred.parent is not None and pred.parent.idx == self.ontology.preds.index('and'):
                        if debug:
                            print "... however, parent is 'and', so a higher nesting will handle this"
                        parent_is_and = True
                    else:
                        parent_is_and = False
                    and_children_are_l_inst = True
                    ands_to_examine = [pred]
                    while len(ands_to_examine) > 0:
                        and_to_examine = ands_to_examine.pop()
                        for c in and_to_examine.children:
                            if c.idx == self.ontology.preds.index('and'):
                                ands_to_examine.append(c)
                            elif (c.is_lambda or c.children is None or len(c.children) != 1 or
                                    not (c.children[0].is_lambda and not c.children[0].is_lambda_instantiation)):
                                and_children_are_l_inst = False
                                break
                    if not parent_is_and and and_children_are_l_inst:
                        if debug:
                            print "'and' abstraction is special case of FA and will have children lambda args stripped"
                        pairs_to_create.append([False, add_and_abstraction, True])

            # create pairs given directives
            for preserve_host_children, and_abstract, special_and_rule in pairs_to_create:

                and_abstracts = [False]
                if and_abstract:
                    and_abstracts.append(True)
                for aa in and_abstracts:

                    if debug:
                        print ("pairs_to_create next params: preserve_host_children " + str(preserve_host_children) +
                               " with and_abstract " + str(and_abstract) + " and special_and_rule " +
                               str(special_and_rule) + " and base pred " + self.print_parse(pred, True))

                    # TODO: this procedure replaces all instances of the identified predicate with lambdas
                    # TODO: in principle, however, should have candidates for all possible subsets of replacements
                    # TODO: eg. askperson(ray,ray) -> lambda x.askperson(x,x), lambda x.askperson(ray,x), etc

                    a2 = copy.deepcopy(pred)
                    a1_lambda_type = None

                    # strip children from a2 and set return type to what would happen if they had not existed
                    if special_and_rule:
                        preserve_r = None
                        ands_to_strip = [a2]
                        ands_to_retype = [a2]
                        while len(ands_to_strip) > 0:
                            and_to_strip = ands_to_strip.pop()
                            for c in and_to_strip.children:
                                if c.idx == self.ontology.preds.index('and'):
                                    ands_to_strip.append(c)
                                    ands_to_retype.append(c)
                                else:
                                    if preserve_r is None:
                                        preserve_r = c.type
                                        a1_lambda_type = preserve_r
                                    c.return_type = c.type
                                    c.children = None
                        # e.g. and<t,<t,t>>(p(x),q(x)) -> and<<e,t>,<<e,t>,<e,t>>>(p,q)
                        ands_to_retype.reverse()
                        for and_to_retype in ands_to_retype:
                            and_to_retype.set_type_from_children_return_types(preserve_r, self.ontology)
                            and_to_retype.return_type = self.ontology.types[and_to_retype.type][0]

                    if preserve_host_children:
                        a1_lambda_type = a2.type
                    elif aa:
                        a1_lambda_type = a2.children[0].type
                    elif not special_and_rule:
                        a1_lambda_type = a2.return_type

                    a1 = SemanticNode.SemanticNode(
                        None, a1_lambda_type, None, True, lambda_name=deepest_lambda+1, is_lambda_instantiation=True)
                    a1.children = [copy.deepcopy(a)]
                    a1.children[0].parent = a1

                    to_replace = [[a1, 0, a1.children[0]]]
                    while len(to_replace) > 0:
                        p, c_idx, r = to_replace.pop()
                        if not r.is_lambda and r.idx == a2.idx:
                            lambda_instance = SemanticNode.SemanticNode(
                                p, a1_lambda_type, None, True, lambda_name=deepest_lambda+1,
                                is_lambda_instantiation=False)
                            p.children[c_idx].copy_attributes(
                                lambda_instance, preserve_parent=True, preserve_children=preserve_host_children)
                            if aa and not special_and_rule:
                                p.children[c_idx].children = copy.deepcopy(a2.children[0].children)
                                for c in p.children[c_idx].children:
                                    c.parent = p.children[c_idx]
                        if r.children is not None:
                            to_replace.extend([[r, idx, r.children[idx]] for idx in range(0, len(r.children))])
                    # print "A1 before renumeration: "+self.print_parse(a1, True)  # DEBUG
                    if preserve_host_children:
                        a2.children = None
                    if aa:
                        if debug:
                            print "... producing an and-abstracted pair"
                        for c in a2.children:
                            c.children = None
                            c.set_return_type(self.ontology)
                            c.parent = a2
                        input_type = [a2.children[0].return_type, a2.children[0].return_type]
                        if input_type not in self.ontology.types:
                            self.ontology.types.append(input_type)
                        full_type = [a2.children[0].return_type, self.ontology.types.index(input_type)]
                        if full_type not in self.ontology.types:
                            self.ontology.types.append(full_type)
                        a2.type = self.ontology.types.index(full_type)
                        a2.set_return_type(self.ontology)

                    if special_and_rule:  # need to add a lambda instance to be consumed by a1
                        if debug:
                            print "adding lambda instance to bottom of a1=" + self.print_parse(a1, True)
                        parent_finder = [a1.children[0]]
                        scoped_type_match_lambda = None
                        while len(parent_finder) > 0:
                            curr_candidate = parent_finder.pop()
                            if (curr_candidate.is_lambda_instantiation and
                                    curr_candidate.type == self.ontology.types[a1.type][0]):
                                scoped_type_match_lambda = curr_candidate.lambda_name
                            if (scoped_type_match_lambda is not None and curr_candidate.is_lambda and
                                    not curr_candidate.is_lambda_instantiation and
                                    curr_candidate.lambda_name == a1.lambda_name):
                                curr_candidate.children = [SemanticNode.SemanticNode(
                                    curr_candidate, self.ontology.types[a1.type][0], None, True,
                                    lambda_name=scoped_type_match_lambda,
                                    is_lambda_instantiation=False)]
                            elif curr_candidate.children is not None:
                                parent_finder.extend(curr_candidate.children)
                        if debug:
                            print "... result a1=" + self.print_parse(a1, True)

                    if debug:
                        print "prenumerated a1 " + self.print_parse(a1, True)
                        print "prenumerated a2 " + self.print_parse(a2, True)
                    a1.set_return_type(self.ontology)
                    a1.renumerate_lambdas([])
                    a2.renumerate_lambdas([])
                    for d, cat in consumables:
                        a1_with_cat = copy.deepcopy(a1)
                        a1_with_cat.set_category(self.lexicon.categories.index([a.category, d, cat]))
                        a2_with_cat = copy.deepcopy(a2)
                        a2_with_cat.set_category(cat)
                        candidate_pairs.append([a1_with_cat, d, a2_with_cat])
                        if debug:
                            print "produced: "+self.print_parse(a1_with_cat, True)+" consuming " + \
                                self.print_parse(a2_with_cat, True)+" in dir "+str(d)+" with params " + \
                                ",".join([str(a1_lambda_type), str(preserve_host_children), str(aa)])  # DEBUG
                        if self.safety and not a1_with_cat.validate_tree_structure():
                            raise RuntimeError("ERROR: invalidly linked structure generated by reverse FA: " +
                                               self.print_parse(a1_with_cat, True) +
                                               "with params "+",".join([str(a1_lambda_type), str(preserve_host_children),
                                                                        str(aa)]))
                        if self.safety and not a2_with_cat.validate_tree_structure():
                            raise RuntimeError("ERROR: invalidly linked structure generated by reverse FA: " +
                                               self.print_parse(a2_with_cat, True) +
                                               "with params "+",".join([str(a1_lambda_type), str(preserve_host_children),
                                                                        str(aa)]))

        return candidate_pairs

    # given a string, return the set of possible tokenizations of that string given lexical entries
    def tokenize(self, s):
        str_parts = s.split()
        return str_parts
