# -*- coding: utf-8 -*-
'''
.. module:: popgen
   :synopsis: Viewing results
   :noindex:
   :copyright: Copyright 2014 by Tiago Antao
   :license: GNU Affero, see LICENSE for details

.. moduleauthor:: Tiago Antao <tra@popgen.net>

'''

import matplotlib.pyplot as plt


class View():
    def __init__(self, model):
        self.model = model
        model.register(self)

    def start(self):
        pass

    def end(self):
        pass

    def set_sim_id(self, sim_id):
        self._sim_id = sim_id

    def set_pop(self, pop):
        self.pop = pop

    def complete_sim(self):
        pass

    def complete_cycle(self, pop):
        pass

    @property
    def view_ops(self):
        view_ops = []
        for param in self.params:
            param.set_pop(self.pop)
            view_ops.extend(param.simupop_stats)
        return view_ops


class BasicView(View):
    def __init__(self, model, params):
        View.__init__(self, model)
        self.params = params
        info_fields = []
        for param in params:
            info_fields.extend(param.simupop_info)
        self.info_fields = list(set(info_fields))

    def start(self):
        self.results = {}
        for param in self.params:
            self._num_sims = len(self.model._sim_ids)
            self.results[param] = [[] for x in range(self._num_sims)]
        self.fig = plt.figure(figsize=(16, 9))

    def complete_cycle(self, pop):
        for param in self.params:
            vals = param.get_values(pop)
            if len(self.results[param][self._sim_id]) == 0:
                for i in range(len(vals)):
                    self.results[param][self._sim_id].append([vals[i]])
            else:
                for i in range(len(vals)):
                    self.results[param][self._sim_id][i].append(vals[i])

    def end(self):
        for i, param in enumerate(self.params):
            for sim_id, results in enumerate(self.results[param]):
                if sim_id == 0:
                    ax = self.fig.add_subplot(
                        len(self.params), self._num_sims,
                        1 + i * self._num_sims + sim_id)
                    ax1 = ax
                else:
                    ax = self.fig.add_subplot(
                        len(self.params), self._num_sims,
                        1 + i * self._num_sims + sim_id, sharey=ax1)
                for my_case in results:
                    ax.plot(my_case)
        return self.fig


class BasicViewTwo(View):
    def __init__(self, model, param):
        View.__init__(self, model)
        info_fields = []
        info_fields.extend(param.simupop_info)
        self.info_fields = list(set(info_fields))
        self.params = [param]
        self.param = param

    def start(self):
        self.results = {}
        self._num_sims = len(self.model._sim_ids)
        self.results[self.param] = [[] for x in range(self._num_sims)]
        vparams = list(self.model._variation_params.keys())
        vparams.sort()
        p1 = self.model._variation_params[vparams[0]]
        p2 = self.model._variation_params[vparams[1]]
        self.fig, self.axs = plt.subplots(len(p1), len(p2),
                                          sharex=True, sharey=True,
                                          figsize=(16, 9), squeeze=False)

    def complete_cycle(self, pop):
        vals = self.param.get_values(pop)
        if len(self.results[self.param][self._sim_id]) == 0:
            for i in range(len(vals)):
                self.results[self.param][self._sim_id].append([vals[i]])
        else:
            for i in range(len(vals)):
                self.results[self.param][self._sim_id][i].append(vals[i])

    def end(self):
        vparams = list(self.model._variation_params.keys())
        vparams.sort()
        p1 = self.model._variation_params[vparams[0]]
        p2 = self.model._variation_params[vparams[1]]
        for i in range(len(p1)):
            for j in range(len(p2)):
                sim_id = i * len(p2) + j
                my_vars = {k: v for k, v in
                           self.model._sim_ids[sim_id].items()
                           if k in self.model._variation_params}
                self.axs[i, j].set_title('%s' % my_vars)
                results = self.results[self.param][sim_id]
                for my_case in results:
                    self.axs[i, j].plot(my_case)
        return self.fig
