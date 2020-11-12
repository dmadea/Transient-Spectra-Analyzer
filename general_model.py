import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.integrate import odeint


class GeneralModel:

    def __init__(self):
        self.elem_reactions = []  # list of dictionaries of elementary reactions

    @classmethod
    def from_text(cls, text_model):
        """
        Expected format is single or multiline or separated by comma, reactions are denoted with equal sign,
        which allows for forward and backward rates. Names of species are case sensitive and cannot contain numbers.
        Eg. decay of triplet benzophenone with mixed 1st and 2nd order (self TT annihilation),
        here zero represents the ground state BP:
            BP = zero, 2BP = BP + zero

        Eg. pandemic SIR(D) model:
            Susceptible + Infected = 2Infected, Infected = Recovered, Infected = Dead

        :param text_model:
            input text
        :return:
            Model representing input reaction scheme.
        """

        _model = cls()

        for line in filter(None, text_model.replace('\n', ',').split(',')):  # filter removes empty entries from split
            tokens = []

            for token in filter(None, line.split('=')):  # remove empty entries
                entries = []

                # process possible number in front of species, species cannot have numbers in their text
                for entry in list(filter(None, token.split('+'))):
                    chars = list(entry)
                    number = ''.join(filter(lambda d: d.isdigit(), chars))
                    text = ''.join(filter(lambda d: d.isalpha(), chars))

                    entries += [text] if number == '' else int(number) * [text]  # list arithmetics

                tokens.append(entries)

            for reactants, products in zip(tokens[:-1], tokens[1:]):
                _model.add_elementary_reaction(reactants, products)

        return _model

    def add_elementary_reaction(self, from_comp=('A', 'A'), to_comp=('B', 'C'), forward_rate=1, backward_rate=0):
        from_comp = from_comp if isinstance(from_comp, (list, tuple)) else [from_comp]
        to_comp = to_comp if isinstance(to_comp, (list, tuple)) else [to_comp]

        reaction = dict(from_comp=from_comp, to_comp=to_comp, forward_rate=forward_rate, backward_rate=backward_rate)

        for el in self.elem_reactions:  # replace rate if the same elementary reaction is already in list
            if el['from_comp'] == reaction['from_comp'] and el['to_comp'] == reaction['to_comp']:
                el['forward_rate'] = reaction['forward_rate']
                el['backward_rate'] = reaction['backward_rate']
                return

        self.elem_reactions.append(reaction)

    def get_compartments(self):
        """
        Return the compartment names, the names are case sensitive.
        """
        names = []
        for el in self.elem_reactions:
            for c in el['from_comp']:
                if c not in names:
                    names.append(c)

            for c in el['to_comp']:
                if c not in names:
                    names.append(c)
        return names

    def build_model(self):
        """
        Builds model and returns the function that takes c, t and rates as an argument
        and can be directly used for odeint method.
        """

        comps = self.get_compartments()

        idx_dict = dict(enumerate(comps))
        inv_idx = dict(zip(idx_dict.values(), idx_dict.keys()))

        r = len(self.elem_reactions)
        idx_from = []  # arrays of arrays of indexes for each elementary reaction
        idx_to = []  # arrays of arrays of indexes for each elementary reaction
        _rates = np.empty((r, 2), dtype=np.float64)

        # build the lists of indexes and so on...
        for i, el in enumerate(self.elem_reactions):
            i_from = list(map(lambda com: inv_idx[com], el['from_comp']))  # list of indexes of starting materials
            i_to = list(map(lambda com: inv_idx[com], el['to_comp']))  # list of indexes of reaction products

            idx_from.append(i_from)
            idx_to.append(i_to)

            _rates[i, 0] = el['forward_rate']
            _rates[i, 1] = el['backward_rate']

        # TODO: possible space for optimization if found too slow for odeint, probably some Cython or C code would be
        # TODO: needed
        def func(c, t, rates=None):
            """Rates if provided must be (r x 2) matrix, where in first column are forward rates and second column
            backward rates. r is number of elementary reactions."""

            rates = _rates if rates is None else rates

            dc_dt = np.zeros_like(c)

            for i in range(r):
                forward_prod, backward_prod = rates[i]

                # eg. for elementary step A + B = C + D
                for k in idx_from[i]:
                    forward_prod *= c[k]  # forward rate products, eg. k_AB * [A] * [B]

                for k in idx_to[i]:
                    backward_prod *= c[k]  # backward rate products, eg. k_CD * [C] * [D]

                for k in idx_from[i]:
                    dc_dt[k] += backward_prod - forward_prod  # reactants

                for k in idx_to[i]:
                    dc_dt[k] += forward_prod - backward_prod  # products

            return dc_dt

        return func

    def simulate_model(self, times, j=None):
        """j = initial population vector"""
        comp = self.get_compartments()

        if j is None:
            j = np.zeros(len(comp))
            j[0] = 1
        else:
            j = np.asarray(j)
            assert j.shape[0] == len(comp)

        solution = odeint(self.build_model(), j, times)

        plt.xlabel('Time')
        plt.ylabel('Concentration')

        for i, c in enumerate(comp):
            plt.plot(times, solution[:, i], label=f'{c}')

        plt.legend(frameon=False)
        plt.show()

    def get_rate_names(self):
        return [f"k_{''.join(el['from_comp'])}_{''.join(el['to_comp'])}" for el in self.elem_reactions]

    @classmethod
    def load(cls, fpath='general models/mod1.json'):
        _model = cls()

        try:
            with open(fpath, "r") as file:
                _model.elem_reactions = json.load(file)

        except Exception as ex:
            print('Error loading target model:\n' + ex.__str__())

        return _model

    def save(self, fpath='general models/mod1.json'):

        try:
            with open(fpath, "w") as file:
                json.dump(self.elem_reactions, file, sort_keys=False, indent=4, separators=(',', ': '))

        except Exception as ex:
            print('Error saving target model:\n' + ex.__str__())

    def print_model(self):
        for el in self.elem_reactions:
            print(f"Elementary reaction: {' + '.join(el['from_comp'])} \u2192 {' + '.join(el['to_comp'])}, "
                  f"forward_rate: {el['forward_rate']}, backward_rate: {el['backward_rate']}")


def main():
    model = GeneralModel()

    model.add_elementary_reaction(['Susceptible', 'Infected'], ['Infected', 'Infected'], 0.5)
    model.add_elementary_reaction('Infected', 'Recovered', 0.1)
    model.add_elementary_reaction('Infected', 'Dead', 0.01)

    # model.add_elementary_reaction(['A', 'B'], ['B'], 0.5)
    # model.add_elementary_reaction('Infected', 'Recovered', 0.1)
    # model.add_elementary_reaction('Infected', 'Dead', 0.01)
    # model.add_elementary_reaction('Recovered', 'Susceptible', 0.01)

    times = np.linspace(0, 100, 1000, dtype=np.float64)

    model.simulate_model(times, [1, 0.001, 0, 0])

    model.save(fpath='general models/SIR_model.json')


if __name__ == '__main__':

    # main()

    reaction = 'A + B = C = D = E\n A + E=B'
    reaction = 'BP = zero, 2 BP = zero + BP'
    SIR = 'Susceptible + Infected = 2 Infected, Infected = Recovered, Infected = Dead'

    model = GeneralModel.from_text(SIR)
    print(model.get_compartments())
    model.print_model()

    model.elem_reactions[1]['forward_rate'] = 0.1
    model.elem_reactions[2]['forward_rate'] = 0.01

    times = np.linspace(0, 50, 1000, dtype=np.float64)

    model.simulate_model(times, [1, 0.01, 0, 0])

    # cProfile.run('main()')
    # model = TargetModel.load()
    # model.print_model()
    # print(model.get_rate_names())
    # model.plot_model()
    #
    # model = GeneralModel()
    #
    # # compartments = list('abcdefghijk')
    # # n = len(compartments)
    # # for i in range(n):
    # #     j = (i + 1) % n
    # #     k = (i + 2) % n
    # #     model.add_transition(compartments[i], compartments[j], i*j + 1)
    # #     # model.add_transition(compartments[i], compartments[k], i*k + 1)
    # #
    # model.add_elementary_reaction(['Susceptible', 'Infected'], ['Infected', 'Infected'], 0.5)
    # model.add_elementary_reaction('Infected', 'Recovered', 0.1)
    # model.add_elementary_reaction('Infected', 'Dead', 0.01)
    #
    # # model.add_elementary_reaction('A', ['B', 'C'], 1, 1)
    # # # model.add_elementary_reaction(['A', 'B'], 'C', 1, 0.01)
    # # model.add_elementary_reaction(['C', 'A'], 'B', 1, 0)
    #
    # times = np.linspace(0, 80, 1000, dtype=np.float64)
    #
    # # model.print_model()
    # model.simulate_model(times, [1, 0.001, 0, 0])

    # print(model.get_compartments())
    # print(model.get_rate_names())
