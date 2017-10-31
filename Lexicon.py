__author__ = 'jesse'

import gensim
import numpy as np
import sys
import SemanticNode


class Lexicon:
    def __init__(self, ontology, lexicon_fname, word_embeddings_fn=None):
        self.ontology = ontology
        self.categories = []  # will grow on its own
        self.surface_forms, self.semantic_forms, self.entries, self.pred_to_surface = \
            self.read_lex_from_file(lexicon_fname)
        self.reverse_entries = []
        self.neighbor_surface_forms = []  # induced from embedding neighbors
        self.sem_form_expected_args = None
        self.sem_form_return_cat = None
        self.category_consumes = None
        self.generator_should_flush = False
        self.update_support_structures()
        self.wv = self.load_word_embeddings(word_embeddings_fn)

    def load_word_embeddings(self, fn):
        if fn is not None:
            wvb = True if fn.split('.')[-1] == 'bin' else False
            wv = gensim.models.Word2Vec.load_word2vec_format(fn, binary=wvb, limit=50000)
        else:
            wv = None
        return wv

    # Return n nearest neighbors of w that are in the lexicon by cosine distance in the embedding vectors.
    # Returns a tuple of size n (more if there are ties, less if fewer than n lexical entries)
    # Tuples are valued (v, d) for v a lexicon surface form and d the cosine similarity scaled to [0, 1]
    def get_lexicon_word_embedding_neighbors(self, w, n):
        debug = False
        if self.wv is None or w not in self.wv.vocab:
            return []
        # only use initial and generator-based lexical entries, not neighbor results since they're duplicates
        candidate_neighbors = [sfidx for sfidx in range(len(self.surface_forms))
                               if sfidx not in self.neighbor_surface_forms]
        pred_cosine = [(1 + self.wv.similarity(w, self.surface_forms[vidx])) / 2.0
                       if self.surface_forms[vidx] in self.wv.vocab else 0 for vidx in candidate_neighbors]
        if max(pred_cosine) == 0:
            return []
        max_sims = [(i, x) for i, x in enumerate(pred_cosine)
                    if np.isclose(x, np.max([pred_cosine[sidx] for sidx in range(len(candidate_neighbors))]))]
        top_k_sims = max_sims[:]
        while len(top_k_sims) < n and len(top_k_sims) < len(candidate_neighbors):  # get top n
            curr_max_val = np.max([pred_cosine[sidx] for sidx in range(len(candidate_neighbors))
                                   if sidx not in [p[0] for p in top_k_sims]])
            top_k_sims.extend([(i, x) for i, x in enumerate(pred_cosine)
                               if np.isclose(x, curr_max_val)])
        if debug:
            print ("get_lexicon_word_embedding_neighbors: top k sims '" + str(w) + "': " +
                   ','.join([str((self.surface_forms[candidate_neighbors[sidx]], sim))
                             for sidx, sim in top_k_sims]))
        return top_k_sims

    def update_support_structures(self):
        self.compute_pred_to_surface(self.pred_to_surface)
        self.reverse_entries = self.compute_reverse_entries()
        self.sem_form_expected_args = [self.calc_exp_args(i) for i in range(0, len(self.semantic_forms))]
        self.sem_form_return_cat = [self.calc_return_cat(i) for i in range(0, len(self.semantic_forms))]
        self.category_consumes = [self.find_consumables_for_cat(i) for i in range(0, len(self.categories))]
        self.generator_should_flush = True

    def compute_pred_to_surface(self, pts):
        for sur_idx in range(0, len(self.entries)):
            for sem_idx in self.entries[sur_idx]:
                to_examine = [self.semantic_forms[sem_idx]]
                while len(to_examine) > 0:
                    curr = to_examine.pop()
                    if not curr.is_lambda:
                        if curr.idx in pts and sur_idx not in pts[curr.idx]:
                            pts[curr.idx].append(sur_idx)
                        elif curr.idx not in pts:
                            pts[curr.idx] = [sur_idx]
                    if curr.children is not None:
                        to_examine.extend(curr.children)

    def compute_reverse_entries(self):
        r = {}
        for sur_idx in range(0, len(self.surface_forms)):
            for sem_idx in self.entries[sur_idx]:
                if sem_idx in r and sur_idx not in r[sem_idx]:
                    r[sem_idx].append(sur_idx)
                else:
                    r[sem_idx] = [sur_idx]
        for sem_idx in range(0, len(self.semantic_forms)):
            if sem_idx not in r:
                r[sem_idx] = []
        r_list = []
        for i in range(0, len(r)):
            if i in r:
                r_list.append(r[i])
            else:
                r_list.append([])
        return r_list

    def calc_exp_args(self, idx):
        exp_args = 0
        curr_cat = self.semantic_forms[idx].category
        while type(self.categories[curr_cat]) is list:
            exp_args += 1
            curr_cat = self.categories[curr_cat][0]
        return exp_args

    def calc_return_cat(self, idx):
        curr_cat = self.semantic_forms[idx].category
        while type(self.categories[curr_cat]) is list:
            curr_cat = self.categories[curr_cat][0]
        return curr_cat

    def find_consumables_for_cat(self, idx):
        consumables = []
        for sem_form in self.semantic_forms:
            curr = self.categories[sem_form.category]
            while type(curr) is list and type(self.categories[curr[0]]):
                if curr[0] == idx:
                    break
                curr = self.categories[curr[0]]
            if curr[0] == idx:
                cons = [curr[1], curr[2]]
                if cons not in consumables:
                    consumables.append(cons)  # store both direction and consumable category
        return consumables

    def get_or_add_category(self, c):
        if c in self.categories:
            return self.categories.index(c)
        self.categories.append(c)
        return len(self.categories) - 1

    def compose_str_from_category(self, idx):
        if idx is None:
            return "NONE IDX"
        if type(self.categories[idx]) is str:
            return self.categories[idx]
        s = self.compose_str_from_category(self.categories[idx][0])
        if type(self.categories[self.categories[idx][0]]) is not str:
            s = '(' + s + ')'
        if self.categories[idx][1] == 0:
            s += '\\'
        else:
            s += '/'
        s2 = self.compose_str_from_category(self.categories[idx][2])
        if type(self.categories[self.categories[idx][2]]) is not str:
            s2 = '(' + s2 + ')'
        return s + s2

    def get_semantic_forms_for_surface_form(self, surface_form):
        if surface_form not in self.surface_forms:
            return []
        else:
            return self.entries[self.surface_forms.index(surface_form)]

    def get_surface_forms_for_predicate(self, pred):
        if type(pred) is str:
            if pred in self.ontology.preds:
                return self.pred_to_surface[self.ontology.preds.index(pred)]
            else:
                return []
        else:
            if pred in self.pred_to_surface:
                return self.pred_to_surface[pred]
            else:
                return []

    def read_lex_from_file(self, fname):
        surface_forms = []
        semantic_forms = []
        entries = []
        pred_to_surface = {}
        f = open(fname, 'r')
        lines = f.readlines()
        self.expand_lex_from_strs(lines, surface_forms, semantic_forms, entries, pred_to_surface)
        f.close()
        return surface_forms, semantic_forms, entries, pred_to_surface

    def expand_lex_from_strs(
            self, lines, surface_forms, semantic_forms, entries, pred_to_surface):
        for line_idx in range(0, len(lines)):
            line = lines[line_idx]

            # skip blank and commented lines
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue

                # add lexical entry
            lhs, rhs = line.split(" :- ")
            surface_form = lhs.strip()
            try:
                sur_idx = surface_forms.index(surface_form)
            except ValueError:
                sur_idx = len(surface_forms)
                surface_forms.append(surface_form)
                entries.append([])
                
            cat_idx, semantic_form = self.read_syn_sem(rhs)
            try:
                sem_idx = semantic_forms.index(semantic_form)
            except ValueError:
                sem_idx = len(semantic_forms)
                semantic_forms.append(semantic_form)
            entries[sur_idx].append(sem_idx)
            preds_in_semantic_form = self.get_all_preds_from_semantic_form(semantic_forms[sem_idx])
            for pred in preds_in_semantic_form:
                if pred in pred_to_surface:
                    pred_to_surface[pred].append(sur_idx)
                else:
                    pred_to_surface[pred] = [sur_idx]

    def read_syn_sem(self, s):
        lhs, rhs = s.split(" : ")
        cat_idx = self.read_category_from_str(lhs.strip())
        semantic_form = self.read_semantic_form_from_str(rhs.strip(), cat_idx, None, [])
        return cat_idx, semantic_form

    def get_all_preds_from_semantic_form(self, node):
        node_preds = []
        if not node.is_lambda:
            node_preds.append(node.idx)
        if node.children is None:
            return node_preds
        for c in node.children:
            node_preds.extend(self.get_all_preds_from_semantic_form(c))
        return node_preds

    def read_category_from_str(self, s):
        # detect whether (,) surrounding s and pop it out of them
        if s[0] == '(':
            p = 1
            for i in range(1, len(s) - 1):
                if s[i] == '(':
                    p += 1
                elif s[i] == ')':
                    p -= 1
                if p == 0: break
            if i == len(s) - 2 and p == 1 and s[-1] == ')': s = s[1:-1]

            # everything up to final unbound /,\ is output, category following is input
        p = 0
        fin_slash_idx = len(s) - 1
        direction = None
        while fin_slash_idx >= 0:
            if s[fin_slash_idx] == ')':
                p += 1
            elif s[fin_slash_idx] == '(':
                p -= 1
            elif p == 0:
                if s[fin_slash_idx] == '/':
                    direction = 1
                    break
                elif s[fin_slash_idx] == '\\':
                    direction = 0
                    break
            fin_slash_idx -= 1
        if fin_slash_idx > 0:  # input/output pair is being described
            output_category_idx = self.read_category_from_str(s[:fin_slash_idx])
            input_category_idx = self.read_category_from_str(s[fin_slash_idx + 1:])
            category = [output_category_idx, direction, input_category_idx]
        else:  # this is an atomic category
            if '(' in s or ')' in s or '/' in s or '\\' in s: sys.exit("Invalid atomic category '" + s + "'")
            category = s
        try:
            idx = self.categories.index(category)
        except ValueError:
            idx = len(self.categories)
            self.categories.append(category)
        return idx

    def read_semantic_form_from_str(self, s, category, parent, scoped_lambdas):
        # the node to be instantiated and returned
        s = s.strip()

        # if head is lambda, create unary lambda node branched from root
        if s[:6] == "lambda":
            str_parts = s[6:].strip().split('.')
            info = str_parts[0]
            name, type_str = info.split(':')
            scoped_lambdas.append(name)
            name_idx = len(scoped_lambdas)
            t = self.ontology.read_type_from_str(type_str)
            node = SemanticNode.SemanticNode(parent, t, category, True, lambda_name=name_idx,
                                             is_lambda_instantiation=True)
            str_remaining = '.'.join(str_parts[1:])  # remove lambda prefix
            str_remaining = str_remaining[1:-1]  # remove scoping parens that follow lambda declaration

            # gather head (either atomic or a predicate with following arguments)
        else:
            end_of_pred = 1
            while end_of_pred < len(s):
                if s[end_of_pred] == '(':
                    break
                end_of_pred += 1
            pred = s[:end_of_pred]

            # check whether head is a lambda name
            curr = parent
            is_scoped_lambda = False
            while curr is not None and not is_scoped_lambda:
                try:
                    pred_idx = scoped_lambdas.index(pred) + 1
                except ValueError:
                    pred_idx = None
                is_scoped_lambda = True if (curr.is_lambda and curr.lambda_name == pred_idx) else False
                if is_scoped_lambda: break
                curr = curr.parent
            if is_scoped_lambda:
                node = SemanticNode.SemanticNode(parent, curr.type, None, True, lambda_name=curr.lambda_name,
                                                 is_lambda_instantiation=False)

                # else, head is an ontological predicate
            else:
                try:
                    pred_idx = self.ontology.preds.index(pred)
                except ValueError:
                    sys.exit("Symbol not found within ontology or lambdas in scope: '" + pred + "'")
                node = SemanticNode.SemanticNode(parent, self.ontology.entries[pred_idx], category, False, idx=pred_idx)

            # remove scoping parens that enclose argument(s) to predicate (if atomic named lambda, may be empty)
            str_remaining = s[end_of_pred + 1:-1]
            # find strings defining children and add recursively
        if len(str_remaining) > 0:
            delineating_comma_idxs = []
            p = d = 0
            for i in range(0, len(str_remaining)):
                if str_remaining[i] == '(':
                    p += 1
                elif str_remaining[i] == ')':
                    p -= 1
                elif str_remaining[i] == '<':
                    d += 1
                elif str_remaining[i] == '>':
                    d -= 1
                elif str_remaining[i] == ',' and p == 0 and d == 0:
                    delineating_comma_idxs.append(i)
            children = []
            splits = [-1]
            splits.extend(delineating_comma_idxs)
            splits.append(len(str_remaining))
            expected_child_cats = []
            curr_cat = category
            while curr_cat is not None and type(self.categories[curr_cat]) is list:
                curr_cat = self.categories[curr_cat][0]
                expected_child_cats.append(curr_cat)
            for i in range(1, len(splits)):
                e_cat = expected_child_cats[i - 1] if len(expected_child_cats) >= i else None
                children.append(
                    self.read_semantic_form_from_str(
                        str_remaining[splits[i-1] + 1:splits[i]], e_cat, node, scoped_lambdas[:]))
            node.children = children
        try:
            node.set_return_type(self.ontology)
        except TypeError as e:
            print e
            sys.exit("Offending string: '" + s + "'")

        if not node.validate_tree_structure():
            sys.exit("ERROR: read in invalidly linked semantic node from string '"+s+"'")  # DEBUG

        # Make a pass through the finished tree to find any * types and replace them with their functional use.
        node = self.instantiate_wild_type(node)

        return node

    # Makes a pass through the tree to replace allowed '*' with instantiated types
    # Currently we only allow 'and' which goes from <*,<*,*>> to <ct,<ct,ct>> for ct the children return types
    def instantiate_wild_type(self, root):
        debug = False

        if root.idx == self.ontology.preds.index('and'):
            crta = self.ontology.compose_str_from_type(root.children[0].return_type)
            crtb = self.ontology.compose_str_from_type(root.children[1].return_type)
            if crta != crtb:
                sys.exit("ERROR: 'and' taking children of different return types " +
                         self.ontology.compose_str_from_type(crta) + ", " +
                         self.ontology.compose_str_from_type(crtb))
            root.type = self.ontology.read_type_from_str("<" + crta + ",<" + crta + "," + crta + ">>")
            if debug:
                print ("instantiate_wild_type: replaced 'and' type with " +
                       self.ontology.compose_str_from_type(root.type))
        if root.children is not None:
            for cidx in range(len(root.children)):
                root.children[cidx] = self.instantiate_wild_type(root.children[cidx])
        return root

    def delete_semantic_form_for_surface_form(self, surface_form, ont_idx):
        if surface_form not in self.surface_forms:
            return
        matching_semantic_form = None
        for semantic_form in self.semantic_forms:
            if semantic_form.idx == ont_idx:
                matching_semantic_form = semantic_form
                break
        if matching_semantic_form is None:
            return
            
        sur_idx = self.surface_forms.index(surface_form)
        sem_idx = self.semantic_forms.index(matching_semantic_form)
        
        if sur_idx in self.entries:
            if sem_idx in self.entries[sur_idx]:
                self.entries[sur_idx].remove(sem_idx)
                
        if ont_idx in self.pred_to_surface:
            if sur_idx in self.pred_to_surface[ont_idx]:
                del self.pred_to_surface[sur_idx]
                
        if sem_idx in self.reverse_entries:
            if sur_idx in self.reverse_entries[sem_idx]:
                self.reverse_entries.remove(sur_idx)
