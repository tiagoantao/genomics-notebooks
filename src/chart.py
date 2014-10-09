# -*- coding: utf-8 -*-
'''
.. module:: popgen
   :synopsis: Viewing results
   :noindex:
   :copyright: Copyright 2014 by Tiago Antao
   :license: GNU Affero, see LICENSE for details

.. moduleauthor:: Tiago Antao <tra@popgen.net>

'''

from collections import defaultdict
import statistics
import os

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class View:
    def __init__(self, model, stats=[], max_y=None):
        self.model = model
        self.stats = stats
        self.max_y = max_y
        self.params = []
        self._sim_id = None
        self._pop = None
        model.register(self)

    def start(self):
        pass

    def end(self):
        pass

    @property
    def sim_id(self):
        return self._sim_id

    @sim_id.setter
    def sim_id(self, sim_id):
        self._sim_id = sim_id

    @property
    def pop(self):
        return self._pop

    @pop.setter
    def pop(self, pop):
        self._pop = pop

    def complete_sim(self):
        pass

    def complete_cycle(self, pop):
        pass

    @property
    def view_ops(self):
        view_ops = []
        for param in self.params:
            param.pop = self.pop
            view_ops.extend(param.simupop_stats)
        return view_ops

    @property
    def info_fields(self):
        info_fields = []
        for param in self.params:
            info_fields.extend(param.info_fields)
        return info_fields


class BasicView(View):
    def __init__(self, model, params, stats=[], max_y=None, with_model=False):
        View.__init__(self, model, stats, max_y)
        self.with_model = with_model
        self.params = params

    def start(self):
        self.results = {}
        self.stat_results = {}
        self._num_sims = len(self.model._sim_ids)
        for stat in self.stats:
            self.stat_results[stat] = {}
        for param in self.params:
            self.results[param.name] = [[] for x in range(self._num_sims)]
            for stat in self.stats:
                self.stat_results[stat][param.name] = [[] for x in
                                                       range(self._num_sims)]

    def complete_cycle(self, pop):
        for param in self.params:
            samp_size = self.model._sim_ids[self._sim_id]['sample_size']
            param.sample_size = samp_size
            vals = param.get_values(pop)
            if len(self.results[param.name][self._sim_id]) == 0:
                for i in range(len(vals)):
                    self.results[param.name][self._sim_id].append([vals[i]])
            else:
                for i in range(len(vals)):
                    self.results[param.name][self._sim_id][i].append(vals[i])
            for stat in self.stats:
                if stat == 'mean':
                    fun = statistics.mean
                stat_dict = self.stat_results[stat][param.name][self._sim_id]
                stat_dict.append(fun(vals))

    def end(self):
        try:
            vparam = list(self.model._variation_params.keys())[0]
        except IndexError:
            # Single parameter, lets show pop_size
            vparam = 'pop_size'
        fig, axs = plt.subplots(len(self.params) +
                                (1 if self.with_model else 0),
                                self._num_sims,
                                figsize=(16, 9), squeeze=False)
        if self.with_model:
            for i, param in enumerate(self.model._sim_ids):
                ax = axs[0, i]
                self.model._draw_sim(ax, param)
                cval = str(param[vparam])
                ax.set_title('%s: %s' % (vparam, cval))
                if i != 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                    plt.setp(ax.get_xticklabels(), visible=False)
        for i, param in enumerate(self.params):
            min_param = None
            max_param = None
            for sim_id, results in enumerate(self.results[param.name]):
                if self.with_model:
                    ax = axs[i + 1, sim_id]
                else:
                    ax = axs[i, sim_id]
                    if i == 0:
                        cval = str(self.model._sim_ids[sim_id][vparam])
                        ax.set_title('%s: %s' % (vparam, cval))
                plt.setp(ax.get_xticklabels(), visible=False)
                if sim_id != 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                for my_case in results:
                    ax.plot(my_case)
                for stat in self.stats:
                    stat_dict = self.stat_results[stat][param.name]
                    ax.plot(stat_dict[sim_id], 'k', lw=4)
                ymin, ymax = ax.get_ylim()
                if min_param is None or ymin < min_param:
                    min_param = ymin
                if max_param is None or ymax > max_param:
                    max_param = ymax
            for sim_id, results in enumerate(self.results[param.name]):
                if self.with_model:
                    ax = axs[i + 1, sim_id]
                else:
                    ax = axs[i, sim_id]
                ax.set_ylim(min_param, max_param)
        if not self.with_model:
            plt.setp(axs[len(self.params) - 1, 0].get_xticklabels(),
                     visible=True)
        fig.tight_layout()
        return fig


class BasicViewTwo(View):
    def __init__(self, model, param, stats=[], highlight=None):
        View.__init__(self, model, stats)
        info_fields = []
        info_fields.extend(param.info_fields)
        self.params = [param]
        self.param = param
        self.highlight = highlight

    def start(self):
        self.results = {}
        self._num_sims = len(self.model._sim_ids)
        self.results[self.param] = [[] for x in range(self._num_sims)]
        vparams = list(self.model._variation_params.keys())
        vparams.sort()

    def complete_cycle(self, pop):
        samp_size = self.model._sim_ids[self._sim_id]['sample_size']
        self.param.sample_size = samp_size
        vals = self.param.get_values(pop)
        if len(self.results[self.param][self._sim_id]) == 0:
            for i in range(len(vals)):
                self.results[self.param][self._sim_id].append([vals[i]])
        else:
            for i in range(len(vals)):
                self.results[self.param][self._sim_id][i].append(vals[i])

    def end(self):
        def get_sim_id(p1, v1, p2, v2):
            for i, sim_params in enumerate(self.model._sim_ids):
                if sim_params[p1] == v1 and sim_params[p2] == v2:
                    return i
        vparams = list(self.model._variation_params.keys())
        vparams.sort()
        p1 = self.model._variation_params[vparams[0]]
        p2 = self.model._variation_params[vparams[1]]
        fig, axs = plt.subplots(len(p2), len(p1),
                                sharex=True, sharey=True,
                                figsize=(16, 9), squeeze=False)
        for i in range(len(p1)):
            v1 = p1[i]
            for j in range(len(p2)):
                v2 = p2[j]
                ax = axs[j, i]
                plt.setp(ax.get_yticklabels(), visible=False)
                plt.setp(ax.get_xticklabels(), visible=False)
                sim_id = get_sim_id(vparams[0], v1, vparams[1], v2)
                if j == 0:
                    cval = str(self.model._sim_ids[sim_id][vparams[0]])
                    axs[j, i].set_title('%s: %s' % (vparams[0], cval))
                results = self.results[self.param][sim_id]
                for l, my_case in enumerate(results):
                    if self.highlight is None or l in self.highlight:
                        axs[j, i].plot(my_case)
                    else:
                        axs[j, i].plot(my_case, 'k', lw=0.5)
        plt.setp(axs[0, 0].get_yticklabels(), visible=True)
        plt.setp(axs[0, len(p1) - 1].get_xticklabels(), visible=True)
        fig.tight_layout()
        for j in range(len(p2)):
            ax = axs[j, len(p1) - 1]
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.text(xmax + 1, ymax, '%s: %s' % (vparams[1], p2[j]),
                    ha='left', va='top', rotation='vertical')
        return fig


class MetaVsDemeView(View):
    def __init__(self, model, meta_param, deme_param, stats=[], max_y=None):
        View.__init__(self, model, stats, max_y)
        self.meta_param = meta_param
        self.deme_param = deme_param
        self.params = [meta_param, deme_param]
        info_fields = []
        info_fields.extend(meta_param.info_fields)
        info_fields.extend(deme_param.info_fields)

    def start(self):
        # need to add stats
        self._num_sims = len(self.model._sim_ids)
        self.deme_results = [defaultdict(list) for x in range(self._num_sims)]
        self.meta_results = [[] for x in range(self._num_sims)]

    def complete_cycle(self, pop):
        # need to add stats
        samp_size = self.model._sim_ids[self._sim_id]['sample_size']
        self.deme_param.sample_size = samp_size
        self.meta_param.sample_size = samp_size
        for sub_pop in range(pop.numSubPop()):
            vals = self.deme_param.get_values(pop, sub_pop)
            if len(self.deme_results[self._sim_id][sub_pop]) == 0:
                for i in range(len(vals)):
                    self.deme_results[self._sim_id][sub_pop].append([vals[i]])
            else:
                for i in range(len(vals)):
                    self.deme_results[self._sim_id][sub_pop][i].append(vals[i])

        vals = self.meta_param.get_values(pop)
        if len(self.meta_results[self._sim_id]) == 0:
            for i in range(len(vals)):
                self.meta_results[self._sim_id].append([vals[i]])
        else:
            for i in range(len(vals)):
                self.meta_results[self._sim_id][i].append(vals[i])

    def end(self):
        max_sub_pops = max([len(self.deme_results[i]) for i in
                            range(len(self.deme_results))])
        fig, axs = plt.subplots(self._num_sims, 1 + max_sub_pops,
                                sharex=True, sharey=True,
                                squeeze=False, figsize=(16, 9))
        y_min = float('inf')
        y_max = float('-inf')
        for i, param in enumerate(range(self._num_sims)):
            n_sub_pops = len(self.deme_results[i])
            ax = axs[i, 0]
            ax.set_axis_bgcolor('0.9')
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_xticklabels(), visible=False)
            for my_case in self.meta_results[i]:
                ax.plot(my_case)
                if y_min > min(my_case):
                    y_min = min(my_case)
                if max(my_case) != float('inf') and y_max < max(my_case):
                    y_max = max(my_case)
            ax.set_xlim(0, len(my_case))
            for sp in range(n_sub_pops):
                ax = axs[i, sp + 1]
                plt.setp(ax.get_yticklabels(), visible=False)
                plt.setp(ax.get_xticklabels(), visible=False)
                for my_case in self.deme_results[i][sp]:
                    ax.plot(my_case)
                    if y_min > min(my_case):
                        y_min = min(my_case)
                    if max(my_case) != float('inf') and y_max < max(my_case):
                        y_max = max(my_case)

        ax = axs[self._num_sims - 1, 0]
        plt.setp(ax.get_yticklabels(), visible=True)
        plt.setp(ax.get_xticklabels(), visible=True)
        if self.max_y is None:
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(y_min, self.max_y)
        fig.subplots_adjust(hspace=0, wspace=0)

        return fig


def _plot_2d(ax, result, prev=None):
    mult_x = 1
    mult_y = 1
    x_0 = []
    y_0 = []
    for (sp, ind), dims in result.items():
        if sp != 0:
            continue
        x_0.append(dims[0])
        y_0.append(dims[1])
    x_0 = sorted(x_0)[len(x_0) // 2]
    y_0 = sorted(y_0)[len(y_0) // 2]

    if prev is not None:
        xp_0, yp_0 = prev
        if abs(xp_0 - x_0) > abs(xp_0 + x_0):
            mult_x = -1
        if abs(yp_0 - y_0) > abs(yp_0 + y_0):
            mult_y = -1
        x_0 *= mult_x
        y_0 *= mult_y
    sp_pos = defaultdict(list)
    for (sp, ind), dims in result.items():
        x = mult_x * dims[0]
        y = mult_y * dims[1]
        sp_pos[sp].append((x, y))
    sps = sorted(sp_pos.keys())
    for c, sp in enumerate(sps):
        vals = sp_pos[sp]
        x, y = zip(*vals)
        try:
            dot = ['.', '<', '>', 'v', '^', '8', 's', 'p',
                   '*'][c // 7]
        except IndexError:
            dot = '*'
        color = ['k', 'r', 'g', 'b', 'm', 'c', 'y'][c % 7]
        ax.plot(x, y, dot, color=color)
    return x_0, y_0


class IndividualView(View):
    def __init__(self, model, param, step=10, with_model=False):
        View.__init__(self, model, stats=[], max_y=None)
        self.params = [param]
        self.gen = 0
        self.step = step
        self.with_model = with_model

    def complete_sim(self):
        self.gen = 0

    def start(self):
        self.gen = 0
        self._num_sims = len(self.model._sim_ids)
        self.results = [[] for x in range(self._num_sims)]

    def complete_cycle(self, pop):
        if self.gen % self.step != 0:
            self.gen += 1
            return
        self.params[0].sample_size = self.model._sim_ids[
            self._sim_id]['sample_size']
        vals = self.params[0].get_values(pop)
        self.results[self._sim_id].append(vals)
        self.gen += 1

    def end(self):
        try:
            vparam = list(self.model._variation_params.keys())[0]
        except IndexError:
            # Single parameter, lets show pop_size
            vparam = 'pop_size'
        all_steps = max([len(x) for x in self.results])
        fig, axs = plt.subplots(all_steps +
                                (1 if self.with_model else 0),
                                self._num_sims,
                                figsize=(16, 4 * all_steps), squeeze=False)
        if self.with_model:
            for i, param in enumerate(self.model._sim_ids):
                ax = axs[0, i]
                self.model._draw_sim(ax, param)
                cval = str(param[vparam])
                ax.set_title('%s: %s' % (vparam, cval))
                if i != 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                    plt.setp(ax.get_xticklabels(), visible=False)
        for i, param in enumerate(range(all_steps)):
            for sim_id, results in enumerate(self.results):
                if self.with_model:
                    ax = axs[i + 1, sim_id]
                else:
                    ax = axs[i, sim_id]
                    if i == 0:
                        cval = str(self.model._sim_ids[sim_id][vparam])
                        ax.set_title('%s: %s' % (vparam, cval))
            for sim_id, results in enumerate(self.results):
                sp_pos = defaultdict(list)
                for (sp, ind), dims in results[i].items():
                    x = dims[0]
                    y = dims[1]
                    sp_pos[sp].append((x, y))
                if self.with_model:
                    ax = axs[i + 1, sim_id]
                else:
                    ax = axs[i, sim_id]
                for c, (sp, vals) in enumerate(sp_pos.items()):
                    x, y = zip(*vals)
                    try:
                        dot = ['.', '<', '>', 'v', '^', '8', 's', 'p',
                               '*'][c // 7]
                    except IndexError:
                        dot = ','
                    ax.plot(x, y, dot)
        if not self.with_model:
            plt.setp(axs[all_steps - 1, 0].get_xticklabels(),
                     visible=True)
        fig.tight_layout()
        return fig


class AnimatedIndividualView(View):
    '''Ánimated view of individuals over time. 2D

       Single simulation (no more)
       '''
    def __init__(self, model, param, step=10, pref='tmp'):
        View.__init__(self, model, stats=[], max_y=None)
        self.params = [param]
        self.gen = 0
        self.step = step
        self.pref = pref

    def complete_sim(self):
        pass  # Single simulation

    def start(self):
        self.gen = 0
        self.results = []

    def complete_cycle(self, pop):
        if self.gen % self.step != 0:
            self.gen += 1
            return
        self.params[0].sample_size = self.model._sim_ids[0]['sample_size']
        vals = self.params[0].get_values(pop)
        self.results.append(vals)
        self.gen += 1

    def _get_lims(self):
        xmax = ymax = float('-inf')
        for result in self.results:
            for pop_inf, dims in result.items():
                x = dims[0]
                y = dims[1]
                if abs(x) > xmax:
                    xmax = abs(x)
                if abs(y) > ymax:
                    ymax = abs(y)
        return xmax, ymax

    def end(self):
        num_frames = len(self.results)
        xmax, ymax = self._get_lims()
        fig = Figure(figsize=(16, 9))
        prev = None
        for i in range(num_frames):
            ax = plt.axes(xlim=(-xmax, xmax), ylim=(-ymax, ymax))
            prev = _plot_2d(ax, self.results[i], prev)
            canvas = FigureCanvasAgg(fig)
            fig.savefig('%s%04d.png' % (self.pref, i + 1))
        try:
            os.remove('%s.mp4' % self.pref)
        except FileNotFoundError:
            pass  # OK, does not exist
        os.system('avconv -r 2 -f image2 -i %s%%04d.png %s.mp4 -vcodec libx264'
                  % (self.pref, self.pref))
        for i in range(num_frames):
            os.remove('%s%04d.png' % (self.pref, i + 1))
        return '%s.mp4' % self.pref
