import numpy as np
import matplotlib.pyplot as plt
import json


class Photokin_Thermal_Model:
    """General model for simulating thermal (only first order reactions) and photokinetics."""

    def __init__(self):

        self.initial_conditions = {}
        self.scheme = ""

        self.transitions = []  # list of dictionaries
        self.fpath = ""
        self.func = None  # function that returns dc_dt for current c and t for build model

    def add_transition(self, from_comp='a', to_comp='b', rate: float=1.0, kind='therm'):
        """kind can be 'therm' or 'photo'

        """
        d_transition = dict(from_comp=from_comp.upper(), to_comp=to_comp.upper(), rate=rate, kind=kind)

        if d_transition not in self.transitions:
            self.transitions.append(d_transition)

    def get_compartments(self):
        """
        Return the compartments names
        """
        l = []
        for trans in self.transitions:
            if trans['from_comp'] not in l:
                l.append(trans['from_comp'])
            if trans['to_comp'] not in l:
                l.append(trans['to_comp'])
        return l


    # @classmethod
    # def from_list(cls, list_of_transitions):
    #     model = cls()
    #     for tr in list_of_transitions:
    #         assert len(tr) >= 2
    #
    #         model.add_transition(tr[0], tr[1], tr[2] if len(tr) > 2 else 1)
    #
    #     return model

    def build_matrices(self):
        """ Builds the n x n K and Phi-matrix """

        comp = self.get_compartments()
        n = len(comp)
        idx_dict = dict(enumerate(comp))
        inv_idx = dict(zip(idx_dict.values(), idx_dict.keys()))
        K_matrix = np.zeros((n, n), dtype=np.float64)
        Phi_matrix = np.zeros((n, n), dtype=np.float64)

        for tr in self.transitions:
            i = inv_idx[tr['from_comp']]
            j = inv_idx[tr['to_comp']]
            matrix = K_matrix if tr['kind'] == 'therm' else Phi_matrix
            matrix[i, i] -= tr['rate']
            matrix[j, i] = tr['rate']

        return K_matrix, Phi_matrix

    def get_rate_names(self):
        names = []

        for tr in self.transitions:
            letter = 'k' if tr['kind'] == 'therm' else 'Phi'
            names.append(f"{letter}_{tr['from_comp']}{tr['to_comp']}")

        return names

    def set_rates(self, rates):
        assert len(rates) == len(self.transitions)

        for tr, rate in zip(self.transitions, rates):
            tr['rate'] = rate

    def plot_model(self, style='circle', radius=0.05, offset_alpha=0.20, offset_text=0.04, filepath=None, dpi=500,
                   transparent=False):
        """

        :param style: can be 'circle', 'linear' or 'random'
        :param radius:
        :param offset_alpha:
        :param offset_text:
        :param filepath:
        :param dpi:
        :param transparent:
        :return:
        """

        if self.transitions.__len__() == 0:
            return

        comps = self.get_compartments()
        idx_dict = dict(enumerate(comps))
        inv_idx = dict(zip(idx_dict.values(), idx_dict.keys()))
        n = len(comps)

        xy_pairs = []

        if style is 'linear':
            xy_pairs = [(x, 0.5) for x in np.linspace(0, 1, n)]
        elif style is 'circle':
            phase = np.pi
            r = 0.5
            for alpha in np.linspace(phase, 2*np.pi + phase, n, endpoint=False):
                xy_pairs.append((0.5 + r*np.cos(-alpha), 0.5 + r*np.sin(-alpha)))
        elif style is 'random':
            xy_pairs = [tuple(pair) for pair in np.random.random((n, 2))]

        figure, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'gray', 'yellow', 'pink', 'brown', 'cyan', 'magenta']

        for i, (cm, xy) in enumerate(zip(comps, xy_pairs)):
            circle = plt.Circle(xy, radius=radius, color=colors[i % len(colors)])
            ax.add_patch(circle)
            ax.annotate(cm, xy=xy, fontsize=13, ha="center", va="center")

        for i, tr in enumerate(self.transitions):
            _i = inv_idx[tr['from_comp']]
            _j = inv_idx[tr['to_comp']]
            xi, xj = xy_pairs[_i][0], xy_pairs[_j][0]
            yi, yj = xy_pairs[_i][1], xy_pairs[_j][1]

            hypotenuse = np.sqrt((xj - xi)**2 + (yj - yi)**2)
            hypotenuse = 1 if np.isclose(hypotenuse, 0) else hypotenuse

            sina = (xj - xi) / hypotenuse  # sin of angle
            cosa = (yj - yi) / hypotenuse  # cos of angle

            ## sin (a+b) = sin a cos b  +  cos a sin b
            ## cos (a+b) = cos a cos b  -  sin a sin b

            sin_ab = sina * np.cos(-offset_alpha) + cosa * np.sin(-offset_alpha)
            cos_ab = cosa * np.cos(-offset_alpha) - sina * np.sin(-offset_alpha)

            sin_ab_end = sina * np.cos(offset_alpha) + cosa * np.sin(offset_alpha)
            cos_ab_end = cosa * np.cos(offset_alpha) - sina * np.sin(offset_alpha)

            x = xi + radius * sin_ab
            y = yi + radius * cos_ab
            x_end = xj - radius * sin_ab_end
            y_end = yj - radius * cos_ab_end

            color = colors[i % len(colors)]

            ax.arrow(x, y, x_end - x, y_end - y, head_width=0.03,
                     length_includes_head=True, width=0.005, color=color,
                     shape='right')

            x_text = (x + x_end) / 2 - cosa * offset_text
            y_text = (y + y_end) / 2 + sina * offset_text

            ax.annotate(f"$k_{{\\mathrm{{{tr['from_comp']}{tr['to_comp']}}}}}$",
                        xy=(x_text, y_text), fontsize=10,
                        ha="center", va="center", color=color)

        ax.autoscale_view()
        plt.box(False)
        plt.show()

    @classmethod
    def load(cls, fpath='target models/target1.json'):
        t_model = cls()

        try:
            with open(fpath, "r") as file:
                t_model.transitions = json.load(file)
                t_model.fpath = fpath

        except Exception as ex:
            print('Error loading target model:\n' + ex.__str__())

        return t_model

    def save(self, fpath='target models/target1.json'):

        try:
            with open(fpath, "w") as file:
                json.dump(self.transitions, file, sort_keys=False, indent=4, separators=(',', ': '))

        except Exception as ex:
            print('Error saving target model:\n' + ex.__str__())

    def print_model(self):
        for tr in self.transitions:
            rate_text = 'rate' if tr['kind'] == 'therm' else 'QY'
            print(f"Transition:  {tr['from_comp']}\u2192{tr['to_comp']}, {rate_text}: {tr['rate']}")


if __name__ == '__main__':

    # model = TargetModel.load()
    # model.print_model()
    # print(model.get_rate_names())
    # model.plot_model()

    model = Photokin_Thermal_Model()

    compartments = list('ZE')
    n = len(compartments)
    # for i in range(n):
    #     j = (i + 1) % n
    #     k = (i + 2) % n
    #     model.add_transition(compartments[i], compartments[j], i*j + 1)
    #     # model.add_transition(compartments[i], compartments[k], i*k + 1)

    model.add_transition('Z', 'E', 1, kind='photo')
    model.add_transition('E', 'Z', 0.2, kind='photo')


    model.print_model()
    print(model.build_matrices())
    print(model.get_rate_names())
    # print(model.get_rate_names())
    # model.plot_model()
    # model.save('target models/11_com_cyclic.json')
    #
    # model.save()
    #
