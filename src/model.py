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

import simuOpt
simuOpt.setOptions(gui=False, quiet=True)
import simuPOP as sp
from simuPOP import demography
from simuPOP import sampling

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

    def _create_genome(self, num_msats, mut=None):
        init_ops = []
        loci = num_msats * [1]
        pre_ops = []
        max_allele_msats = 100
        start_alleles = 10

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
        pop = sp.Population(pop_sizes, ploidy=2, loci=[1] * nloci,
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
            view.set_sim_id(sim_id)
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
            params['num_msats'], params['mut_msat'])
        view_ops = []
        for view in self._views:
            view.set_pop(pop)
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
        loci, genome_init, gpre_ops = self._create_genome(params['num_msats'])
        view_ops = []
        for view in self._views:
            view.set_pop(pop)
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
            view.set_pop(pop)
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
        loci, genome_init, gpre_ops = self._create_genome(params['num_msats'])
        view_ops = []
        for view in self._views:
            view.set_pop(pop)
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
    def __init__(self, gens, two_d):
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
                        [params['pop_size']] * params['num_pops'],
                        params['mig'], params['num_msats'])
            else:
                pop, init_ops, pre_ops, post_ops = \
                    self._create_stepping_stone(
                        [[params['pop_size']] * params['num_pops_x']] *
                        params['num_pops_y'],
                        params['mig'], params['num_msats'])
        loci, genome_init, gpre_ops = self._create_genome(params['num_msats'])
        view_ops = []
        for view in self._views:
            view.set_pop(pop)
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


def _get_sub_sample(pop, size, sub_pop=None):
    if sub_pop is None:
        pop_s = pop
    else:
        pop_s = pop.extractSubPops(subPops=[sub_pop])
    if size is None:
        return pop_s
    pop_s = sampling.drawRandomSample(pop_s, sizes=size)
    return pop_s


class Parameter():
    def __init__(self, do_structured=False):
        self.name = None
        self.desc = None
        self.simupop_info = []
        self.do_structured = do_structured
        self._sample_size = None

    def get_values(self, pop, sub_pop=None):
        if self.do_structured:
            pop_ = _get_sub_sample(pop, self.sample_size, sub_pop)
        else:
            pop_ = _get_sub_sample(pop, self.sample_size)
        ind_values = self._get_values(pop_)
        return ind_values

    def set_pop(self, pop):
        # To set the operators (at start)
        self.pop = pop

    @property
    def simupop_stats(self):
        return []

    @property
    def sample_size(self):
        return self._sample_size

    @sample_size.setter
    def sample_size(self, value):
        self._sample_size = value


class ObsHe(Parameter):
    def __init__(self):
        Parameter.__init__(self)
        self.name = 'ObsHe'
        self.desc = 'Observed Heterozygozity'

    def _get_values(self, pop):
        st = sp.Stat(heteroFreq=True)
        st.apply(pop)
        loci = list(pop.dvars().heteroFreq.keys())
        loci.sort()
        return [pop.dvars().heteroFreq[l] for l in loci]


class ExpHe(Parameter):
    def __init__(self, **kwargs):
        Parameter.__init__(self, kwargs)
        self.name = 'ExpHe'
        self.desc = 'Expected Heterozygozity'

    def _get_values(self, pop):
        st = sp.Stat(alleleFreq=True)
        st.apply(pop)
        freqs = pop.dvars().alleleFreq
        loci = list(freqs.keys())
        loci.sort()
        expHe = []
        for locus in loci:
            afreqs = freqs[locus]
            expHo = 0
            for allele, freq in afreqs.items():
                expHo += freq*freq
            expHe.append(1 - expHo)
        return expHe


class NumAlleles(Parameter):
    def __init__(self, **kwargs):
        Parameter.__init__(self, kwargs)
        self.name = 'NumAlleles'
        self.desc = 'Number of Alleles'

    def _get_values(self, pop):
        st = sp.Stat(alleleFreq=True)
        st.apply(pop)
        anum = pop.dvars().alleleNum
        loci = list(anum.keys())
        loci.sort()
        anums = [len(anum[l]) for l in loci]
        return anums


class fst(Parameter):
    def __init__(self, **kwargs):
        StructuredParameter.__init__(self)
        self.name = 'fst'
        self.desc = 'FST per locus'

    def _get_values(self, pop):
        st = sp.Stat(structure=sp.ALL_AVAIL, vars=['f_st'])
        st.apply(pop)
        fsts = pop.dvars().f_st
        loci = list(fsts.keys())
        return [fsts[l] for l in loci]


class StructuredParameter(Parameter):
    def __init__(self, **kwargs):
        kwargs['do_structured'] = True
        Parameter.__init__(self, kwargs)


class FST(StructuredParameter):
    def __init__(self, **kwargs):
        StructuredParameter.__init__(self)
        self.name = 'FST'
        self.desc = 'FST'

    def _get_values(self, pop):
        st = sp.Stat(structure=sp.ALL_AVAIL)
        st.apply(pop)
        fst = pop.dvars().F_st
        return [fst]


class LDNe(Parameter):
    def __init__(self, pcrit=0.02, **kwargs):
        Parameter.__init__(self, kwargs)
        self.name = 'LDNe'
        self.desc = 'LDNe'
        self.pcrit = pcrit

    def _get_values(self, pop):
        st = sp.Stat(effectiveSize=sp.ALL_AVAIL, vars='Ne_LD')
        st.apply(pop)
        ne = pop.dvars().Ne_LD
        return ne[self.pcrit]


class FreqDerived(Parameter):
    def __init__(self, **kwargs):
        Parameter.__init__(self, kwargs)
        self.name = 'FreqDerived'
        self.desc = 'Frequency of the Derived Allele'

    def _get_values(self, pop):
        st = sp.Stat(alleleFreq=True)
        st.apply(pop)
        anum = pop.dvars().alleleFreq
        loci = list(anum.keys())
        loci.sort()
        anums = [anum[l][1] for l in loci]
        return anums
