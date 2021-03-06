__author__ = 'jesse'

import sys


class Ontology:
    # initializes an ontology data structure which reads in atoms and predicates from given files
    def __init__(self, ont_fname):

        # subsequent entries are tuples of indices into this list defining a binary hierarchy
        # when reading in from the ontology, if a base type hasn't been seen before it will be added
        self.types = ['*']

        # get predicates and map from predicates to types
        self.preds, self.entries = self.read_sem_from_file(ont_fname)

        # store commutative predicates
        self.commutative = []  # indexed in preds
        for pidx in range(len(self.preds)):
            if self.preds[pidx][0] == '*':
                self.preds[pidx] = self.preds[pidx][1:]
                self.commutative.append(pidx)

        # calculate and store number of arguments each predicate takes (atoms take 0)
        self.num_args = [self.calc_num_pred_args(i) for i in range(0, len(self.preds))]

    # calculate the number of arguments a predicate takes
    def calc_num_pred_args(self, idx):
        num_args = 0
        curr_type = self.types[self.entries[idx]]
        while type(curr_type) is list:
            num_args += 1
            curr_type = self.types[curr_type[1]]
        return num_args

    # reads semantic atom/predicate declarations from a given file of format:
    # atom_name:type
    # pred_name:<complex_type>
    # *commutative_pred_name:<complex_type>
    def read_sem_from_file(self, fname):

        preds = ['*and']
        entries = [self.read_type_from_str('<*,<*,*>>', allow_wild=True)]  # map of pred_idx:type read in
        f = open(fname, 'r')
        for line in f.readlines():

            # ignore blank lines and comments
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue

            # create semantic meaning representation from string
            [name, type_str] = line.split(':')
            if name in preds:
                sys.exit("Multiply defined type for predicate '" + name + "'")
            entries.append(self.read_type_from_str(type_str))
            preds.append(name)
        f.close()
        return preds, entries

    # returns the index of self.types at which this type is stored; adds types to this list as necessary to
    # compose such a type
    def read_type_from_str(self, s, allow_wild=False):

        # a complex type
        if s[0] == "<" and s[-1] == ">":
            d = 0
            split_idx = None
            for split_idx in range(1, len(s) - 1):
                if s[split_idx] == '<':
                    d += 1
                elif s[split_idx] == '>':
                    d -= 1
                elif s[split_idx] == ',' and d == 0:
                    break
            comp_type = [self.read_type_from_str(s[1:split_idx], allow_wild=allow_wild),
                         self.read_type_from_str(s[split_idx + 1:-1], allow_wild=allow_wild)]
            try:
                return self.types.index(comp_type)
            except ValueError:
                self.types.append(comp_type)
                return len(self.types) - 1

        # a primitive type
        else:
            if s not in self.types:
                self.types.append(s)
            t = self.types.index(s)
            if not allow_wild and t == self.types.index('*'):
                sys.exit("The '*' type only has internal support.")
            return t

    # returns a string representing the given ontological type
    def compose_str_from_type(self, t):
        s = ''
        # a complex type
        if type(self.types[t]) is list:
            s += '<' + self.compose_str_from_type(self.types[t][0]) + ',' + self.compose_str_from_type(self.types[t][1]) + '>'
        # a primitive type
        else:
            s += self.types[t]
        return s

    # returns true if the given types are equivalent replacing '*' with matches
    def types_equal(self, tidx, tjdx):
        ti = self.types[tidx]
        tj = self.types[tjdx]
        if type(ti) is list and type(tj) is list:
            if self.types_equal(ti[0], tj[0]) and self.types_equal(ti[1], tj[1]):
                return True
        elif tidx == tjdx or tidx == self.types.index('*') or tjdx == self.types.index('*'):
            return True
        return False
