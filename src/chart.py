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

    def complete_sim(self):
        pass

    def complete_cycle(self, pop):
        pass


class BasicView(View):
    def __init__(self, model, param):
        View.__init__(self, model)
        self.param = param
        self.info_fields = param.simupop_info
        self.post_ops = param.simupop_stats

    def start(self):
        self._num_sims = len(self.model._sim_ids)
        self.results = [[] for x in range(self._num_sims)]
        self.fig, self._axs = plt.subplots(
            1, self._num_sims, squeeze=False, sharey=True,
            figsize=(16, 9))

    def complete_cycle(self, pop):
        vals = self.param.get_values(pop)
        if len(self.results[self._sim_id]) == 0:
            for i in range(len(vals)):
                self.results[self._sim_id].append([vals[i]])
        else:
            for i in range(len(vals)):
                self.results[self._sim_id][i].append(vals[i])

    def end(self):
        for sim_id, results in enumerate(self.results):
            ax = self._axs[0, sim_id]
            for my_case in results:
                ax.plot(my_case)
        return self.fig
