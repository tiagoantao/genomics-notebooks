# -*- coding: utf-8 -*-
'''
.. module:: genomics
   :synopsis: PopGen classes with simulations
   :noindex:
   :copyright: Copyright 2014 by Tiago Antao
   :license: GNU Affero, see LICENSE for details

.. moduleauthor:: Tiago Antao <tra@popgen.net>

'''

import copy
import inspect
import math

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

import simuOpt
simuOpt.setOptions(gui=False, quiet=True)
import simuPOP as sp
from simuPOP import demography

from IPython.core.pylabtools import print_figure
from IPython.display import Image


def _hook_view(pop, param):
    view = param
    view.complete_cycle(pop)
    return True


class Model:
    def __init__(self, gens):
        self._gens = gens
        self._views = []
        self.pop_size = 100
        self.num_msats = 10
        self.num_msat_alleles = 10
        self.mut_msat = None
        self.sample_size = None  # All individuals
        self._stats = set()
        self._info_fields = set()
        self._sim_ids = []
        self._sims = []

    def register(self, view):
        self._views.append(view)

    def add_stat(self, stat):
        self._stats.add(stat)

    def _repr__png_(self):
        param_order = list(self._variation_params.keys())
        param_order.sort()
        if len(self._variation_params) == 0:
            ys = 1
            xs = 1
        elif len(self._variation_params) == 1:
            p1 = param_order[0]
            vals = self._variation_params[p1]
            xs = min([3, len(vals)])
            ys = math.ceil(len(vals) / 3)
        else:
            p1 = param_order[0]
            p2 = param_order[1]
            xs = len(self._variation_params[p1])
            ys = len(self._variation_params[p2])
        fig, axs = plt.subplots(ys, xs, squeeze=False, figsize=(16, 9))
        for i, sim_params in enumerate(self._sim_ids):
            x = i % 3
            y = i // 3
            ax = axs[y, x]
            self._draw_sim(ax, sim_params)
        for i in range(i + 1, ys * xs):
            x = i % 3
            y = i // 3
            ax = axs[y, x]
            ax.set_axis_off()
        data = print_figure(fig, 'png')
        plt.close(fig)
        return data

    @property
    def png(self):
        return Image(self._repr__png_(), embed=True)

    def _create_snp_genome(self, num_snps, freq):
        init_ops = []
        loci = num_snps * [1]

        for snp in range(num_snps):
            init_ops.append(sp.InitGenotype(freq=[1 - freq, freq], loci=snp))

        return loci, init_ops

    def _create_genome(self, num_msats, mut=None, start_alleles=10):
        init_ops = []
        loci = num_msats * [1]
        pre_ops = []
        max_allele_msats = 100

        for msat in range(num_msats):
            diri = np.random.mtrand.dirichlet([1.0] * start_alleles)
            if type(diri[0]) == float:
                diri_list = diri
            else:
                diri_list = list(diri)

            init_ops.append(
                sp.InitGenotype(freq=[0.0] * ((max_allele_msats + 1 - 8) //
                                2) + diri_list + [0.0] *
                                ((max_allele_msats + 1 - 8) // 2),
                                loci=msat))
        if mut is not None:
            pre_ops.append(sp.StepwiseMutator(rates=mut))

        return loci, init_ops, pre_ops

    def _create_single_pop(self, pop_size, nloci):
        init_ops = []
        init_ops.append(sp.InitSex())
        pop = sp.Population(pop_size, ploidy=2, loci=[1] * nloci,
                            chromTypes=[sp.AUTOSOME] * nloci,
                            infoFields=list(self._info_fields))
        pre_ops = []
        post_ops = []
        return pop, init_ops, pre_ops, post_ops

    def _create_island(self, pop_sizes, mig, nloci):
        init_ops = []
        init_ops.append(sp.InitSex())
        pop = sp.Population(pop_sizes, ploidy=2, loci=[1] * nloci,
                            chromTypes=[sp.AUTOSOME] * nloci,
                            infoFields=list(self._info_fields))
        post_ops = [sp.Migrator(
            demography.migrIslandRates(mig, len(pop_sizes)))]
        pre_ops = []
        self._info_fields.add('migrate_to')
        return pop, init_ops, pre_ops, post_ops

    def _create_stepping_stone(self, pop_sizes, mig, nloci):
        if len(pop_sizes) == 1:
            flat_pop_sizes = pop_sizes[0]
            post_ops = [sp.Migrator(
                demography.migrSteppingStoneRates(mig, len(flat_pop_sizes)))]
        else:
            flat_pop_sizes = []
            for line in pop_sizes:
                flat_pop_sizes.extend(line)
            post_ops = [sp.Migrator(
                demography.migr2DSteppingStoneRates(mig,
                                                    len(pop_sizes),
                                                    len(pop_sizes[0])))]
        init_ops = []
        init_ops.append(sp.InitSex())
        pop = sp.Population(flat_pop_sizes, ploidy=2, loci=[1] * nloci,
                            chromTypes=[sp.AUTOSOME] * nloci,
                            infoFields=list(self._info_fields))
        pre_ops = []
        self._info_fields.add('migrate_to')
        return pop, init_ops, pre_ops, post_ops

    def prepare_sim_vars(self):
        fixed_params = {}
        variation_params = {}
        for name, val in inspect.getmembers(self):
            if inspect.ismethod(val) or name[0] == '_':
                continue
            if type(val) == list:
                variation_params[name] = val
            else:
                fixed_params[name] = val
        self._set_sim_ids(fixed_params, variation_params)
        self._variation_params = variation_params
        self._fixed_params = fixed_params

    def _set_sim_ids(self, fixed_params, variation_params):
        if len(variation_params) == 0:
            self._sim_ids.append(copy.copy(fixed_params))
        elif len(variation_params) == 1:
            for name, values in variation_params.items():  # just one, really
                for value in values:
                    sim_params = copy.copy(fixed_params)
                    sim_params[name] = value
                    self._sim_ids.append(sim_params)
        elif len(variation_params) == 2:
            n1, n2 = tuple(variation_params.keys())
            v1s = variation_params[n1]
            v2s = variation_params[n2]
            for v1 in v1s:
                for v2 in v2s:
                    sim_params = copy.copy(fixed_params)
                    sim_params[n1] = v1
                    sim_params[n2] = v2
                    self._sim_ids.append(sim_params)
        else:
            raise Exception('Maximum of 2 parameters varying')

    def prepare_sim(self, params):
        raise NotImplementedError('Use a concrete subclass')

    def _run(self, sim_id, params):
        pr = self.prepare_sim(params)
        sim = pr['sim']
        for view in self._views:
            view.sim_id = sim_id
        sim.evolve(initOps=pr['init_ops'],
                   preOps=pr['pre_ops'],
                   postOps=pr['post_ops'],
                   matingScheme=pr['mating_scheme'],
                   gen=self._gens)
        for view in self._views:
            view.complete_sim()

    def run(self):
        self.prepare_sim_vars()
        for view in self._views:
            view.start()
        for params in self._sim_ids:
            self._sims.append(self.prepare_sim(params))
        for i, params in enumerate(self._sim_ids):
            self._run(i, params)
        for view in self._views:
            view.end()


class SinglePop(Model):
    def prepare_sim(self, params):
        for view in self._views:
            for info in view.info_fields:
                self._info_fields.add(info)
        pop, init_ops, pre_ops, post_ops = \
            self._create_single_pop(params['pop_size'], params['num_msats'])
        loci, genome_init, gpre_ops = self._create_genome(
            params['num_msats'], mut=params['mut_msat'],
            start_alleles=params['num_msat_alleles'])
        view_ops = []
        for view in self._views:
            view.pop = pop
            view_ops.extend(view.view_ops)
        for view in self._views:
            post_ops.append(sp.PyOperator(func=_hook_view, param=view))
        post_ops = view_ops + post_ops
        sim = sp.Simulator(pop, 1, True)
        return {'sim': sim, 'init_ops': init_ops + genome_init,
                'pre_ops': pre_ops + gpre_ops, 'post_ops': post_ops,
                'mating_scheme': sp.RandomMating()}

    def _draw_sim(self, ax, sim_params):
        pop_size = sim_params['pop_size']
        ax.plot([0, self._gens], [0, 0], 'b')
        ax.plot([0, self._gens], [pop_size, pop_size], 'b')
        ax.set_xlim(0, self._gens)
        if type(self.pop_size) == list:
            pop_sizes = self.pop_size
        else:
            pop_sizes = [self.pop_size]
        ax.set_ylim(-10, 1.1 * max(pop_sizes))


class Bottleneck(Model):
    def prepare_sim(self, params):
        for view in self._views:
            for info in view.info_fields:
                self._info_fields.add(info)
        pop, init_ops, pre_ops, post_ops = \
            self._create_single_pop(params['start_size'], params['num_msats'])
        loci, genome_init, gpre_ops = self._create_genome(params['num_msats'],
            start_alleles=params['num_msat_alleles'])
        view_ops = []
        for view in self._views:
            view.pop = pop
            view_ops.extend(view.view_ops)
        for view in self._views:
            post_ops.append(sp.PyOperator(func=_hook_view, param=view))
        post_ops = view_ops + post_ops
        sim = sp.Simulator(pop, 1, True)
        pre_ops.append(sp.ResizeSubPops(
            proportions=(params['end_size'] / params['start_size'],),
            at=params['bgen']))
        return {'sim': sim, 'pop': pop, 'init_ops': init_ops + genome_init,
                'pre_ops': pre_ops + gpre_ops, 'post_ops': post_ops,
                'mating_scheme': sp.RandomMating()}

    def _draw_sim(self, ax, sim_params):
        start_size = sim_params['start_size']
        end_size = sim_params['end_size']
        bgen = sim_params['bgen']
        ax.plot([0, self._gens], [0, 0], 'b')
        ax.plot([0, bgen], [start_size, start_size], 'b')
        ax.plot([bgen, bgen], [start_size, end_size], 'b')
        ax.plot([bgen, self._gens], [end_size, end_size], 'b')
        ax.set_xlim(0, self._gens)
        if type(self.start_size) == list:
            pop_sizes = self.start_size
        else:
            pop_sizes = [self.start_size]
        if type(self.end_size) == list:
            pop_sizes.extend(self.end_size)
        else:
            pop_sizes.append(self.end_size)
        ax.set_ylim(-10, 1.1 * max(pop_sizes))


class SelectionPop(Model):
    def __init__(self, gens):
        Model.__init__(self, gens)
        self.sel = 0.01
        self.freq = 0.01
        self.neutral_loci = 0

    def prepare_sim(self, params):
        for view in self._views:
            for info in view.info_fields:
                self._info_fields.add(info)
        nloci = 1 + params['neutral_loci']
        pop, init_ops, pre_ops, post_ops = \
            self._create_single_pop(params['pop_size'], nloci)
        view_ops = []
        for view in self._views:
            view.pop = pop
            view_ops.extend(view.view_ops)
        for view in self._views:
            post_ops.append(sp.PyOperator(func=_hook_view, param=view))
        post_ops = view_ops + post_ops
        loci, genome_init = self._create_snp_genome(nloci, freq=params['freq'])
        sim = sp.Simulator(pop, 1, True)
        if params['sel_type'] == 'hz_advantage':
            ms = sp.MapSelector(loci=0, fitness={
                (0, 0): 1 - params['sel'],
                (0, 1): 1,
                (1, 1): 1 - params['sel']})
        elif params['sel_type'] == 'recessive':
            ms = sp.MapSelector(loci=0, fitness={
                (0, 0): 1 - params['sel'],
                (0, 1): 1 - params['sel'],
                (1, 1): 1})
        else:  # dominant
            ms = sp.MapSelector(loci=0, fitness={
                (0, 0): 1 - params['sel'],
                (0, 1): 1,
                (1, 1): 1})
        return {'sim': sim, 'pop': pop, 'init_ops': init_ops + genome_init,
                'pre_ops': pre_ops, 'post_ops': post_ops,
                'mating_scheme': sp.RandomMating(
                    ops=[sp.MendelianGenoTransmitter(), ms])}


class Island(Model):
    def __init__(self, gens):
        Model.__init__(self, gens)
        self.num_pops = 5
        self.mig = 0.01

    def prepare_sim(self, params):
        for view in self._views:
            for info in view.info_fields:
                self._info_fields.add(info)
        pop, init_ops, pre_ops, post_ops = \
            self._create_island([params['pop_size']] * params['num_pops'],
                                params['mig'], params['num_msats'])
        loci, genome_init, gpre_ops = self._create_genome(params['num_msats'],
            start_alleles=params['num_msat_alleles'])
        view_ops = []
        for view in self._views:
            view.pop = pop
            view_ops.extend(view.view_ops)
        for view in self._views:
            post_ops.append(sp.PyOperator(func=_hook_view, param=view))
        post_ops = view_ops + post_ops
        sim = sp.Simulator(pop, 1, True)
        return {'sim': sim, 'pop': pop, 'init_ops': init_ops + genome_init,
                'pre_ops': pre_ops, 'post_ops': post_ops,
                'mating_scheme': sp.RandomMating()}

    def _draw_sim(self, ax, sim_params):
        graph = nx.Graph()
        num_pops = sim_params['num_pops']
        gnames = ['P%d: %d' % (g + 1, sim_params['pop_size'], )
                  for g in range(num_pops)]
        for g in range(num_pops):
            graph.add_node(gnames[g])
        for g1 in range(num_pops - 1):
            for g2 in range(g1 + 1, num_pops):
                graph.add_edge(gnames[g1], gnames[g2])
        nx.draw_circular(graph, node_color='c', ax=ax)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        pos = ymin
        for var in self._variation_params:
            if var == 'mig':
                continue
            ax.text(xmin, pos, '%s: %d' % (var, int(sim_params[var])),
                    va='top', ha='left')
            pos = (ymax - ymin) / 2 + ymin
        ax.text(xmin, ymax, 'mig: %f' % sim_params['mig'],
                va='top', ha='left')


class SteppingStone(Model):
    def __init__(self, gens, two_d=False):
        Model.__init__(self, gens)
        self.num_pops_x = 5
        self.mig = 0.01
        self._two_d = two_d
        self.num_pops_y = None

    def prepare_sim(self, params):
        for view in self._views:
            for info in view.info_fields:
                self._info_fields.add(info)
            if self._two_d:
                pop, init_ops, pre_ops, post_ops = \
                    self._create_stepping_stone(
                        [[params['pop_size']] * params['num_pops_x']] *
                        params['num_pops_y'],
                        params['mig'], params['num_msats'])
            else:
                pop, init_ops, pre_ops, post_ops = \
                    self._create_stepping_stone(
                        [[params['pop_size']] * params['num_pops_x']],
                        params['mig'], params['num_msats'])
        loci, genome_init, gpre_ops = self._create_genome(params['num_msats'],
            start_alleles=params['num_msat_alleles'])
        view_ops = []
        for view in self._views:
            view.pop = pop
            view_ops.extend(view.view_ops)
        for view in self._views:
            post_ops.append(sp.PyOperator(func=_hook_view, param=view))
        post_ops = view_ops + post_ops
        sim = sp.Simulator(pop, 1, True)
        return {'sim': sim, 'pop': pop, 'init_ops': init_ops + genome_init,
                'pre_ops': pre_ops, 'post_ops': post_ops,
                'mating_scheme': sp.RandomMating()}

    def _draw_sim(self, ax, sim_params):
        if self._two_d:
            y = sim_params['num_pops_y']
        else:
            y = 1
        ax.set_axis_off()
        ax.set_ylim(0, 1 + y)
        ax.set_xlim(0, 1 + sim_params['num_pops_x'])
        for j in range(y):
            for i in range(sim_params['num_pops_x']):
                el = Ellipse((i + 1, j + 1), 0.5, 0.5, ec="none")
                ax.add_patch(el)
                if i > 0:
                    ax.plot([i + .25, i + .75], [j + 1, j + 1], 'k')
                if j > 0:
                    ax.plot([i + 1, i + 1], [j + .25, j + .75], 'k')
