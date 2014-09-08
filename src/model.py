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

import numpy as np

from IPython.core.pylabtools import print_figure
from IPython.display import Image

from matplotlib import pyplot as plt

import simuOpt
simuOpt.setOptions(gui=False, quiet=True)
import simuPOP as sp
from simuPOP import demography


def _hook_view(pop, param):
    view = param
    view.complete_cycle(pop)
    return True


class Model:
    def __init__(self, gens):
        self._gens = gens
        self._views = []
        self.pop_size = 100
        self.num_msats = 100
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
        if type(self.pop_size) == int:
            sizes = [self.pop_size]
        else:
            sizes = self.pop_size
        xs = 1 + len(sizes) // 3
        ys = len(sizes) % 3
        fig, axs = plt.subplots(xs, ys, squeeze=False)
        x_size = 2 * max(sizes)
        fig.suptitle('Single, constant sized populations')
        for i, size in enumerate(sizes):
            x = i // 3
            y = i % 3
            x_start = max(sizes) - size / 2
            x_end = max(sizes) + size / 2
            ax = axs[x, y]
            ax.plot([x_start, x_start], [0, 1], 'b')
            ax.plot([x_end, x_end], [0, 1], 'b')
            ax.text(x_size / 2, 0.5, 'Nc = %d' % size,
                    ha='center', va='top')
            ax.set_xlim(0, x_size)
            ax.set_ylim(0, 1)
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

    def _create_genome(self, num_msats):
        init_ops = []
        loci = num_msats * [1]
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

        return loci, init_ops

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
        if params['sample_size'] is None:
            pr['pop'].setVirtualSplitter(sp.ProportionSplitter(
                proportions=[1]))
        else:
            pr['pop'].setVirtualSplitter(sp.RangeSplitter(
                ranges=[[0, params['sample_size']]]))
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
        loci, genome_init = self._create_genome(params['num_msats'])
        view_ops = []
        for view in self._views:
            view_ops.extend(view.post_ops)
        for view in self._views:
            post_ops.append(sp.PyOperator(func=_hook_view, param=view))
        post_ops = view_ops + post_ops
        sim = sp.Simulator(pop, 1, True)
        return {'sim': sim, 'pop': pop, 'init_ops': init_ops + genome_init,
                'pre_ops': pre_ops, 'post_ops': post_ops,
                'mating_scheme': sp.RandomMating()}


class Bottleneck(Model):
    def prepare_sim(self, params):
        for view in self._views:
            for info in view.info_fields:
                self._info_fields.add(info)
        pop, init_ops, pre_ops, post_ops = \
            self._create_single_pop(params['start_size'], params['num_msats'])
        loci, genome_init = self._create_genome(params['num_msats'])
        view_ops = []
        for view in self._views:
            view_ops.extend(view.post_ops)
        for view in self._views:
            post_ops.append(sp.PyOperator(func=_hook_view, param=view))
        post_ops = view_ops + post_ops
        sim = sp.Simulator(pop, 1, True)
        pre_ops.append(sp.ResizeSubPops(
            proportions=(params['end_size'] / params['start_size'],),
            at=params['bgen']))
        return {'sim': sim, 'pop': pop, 'init_ops': init_ops + genome_init,
                'pre_ops': pre_ops, 'post_ops': post_ops,
                'mating_scheme': sp.RandomMating()}


class SelectionPop(Model):
    def __init__(self, gens):
        Model.__init__(self, gens)
        self.sel = 0.01
        self.freq = 0.01

    def prepare_sim(self, params):
        for view in self._views:
            for info in view.info_fields:
                self._info_fields.add(info)
        pop, init_ops, pre_ops, post_ops = \
            self._create_single_pop(params['pop_size'], 1)
        view_ops = []
        for view in self._views:
            view_ops.extend(view.post_ops)
        for view in self._views:
            post_ops.append(sp.PyOperator(func=_hook_view, param=view))
        post_ops = view_ops + post_ops
        loci, genome_init = self._create_snp_genome(1, freq=params['freq'])
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
        loci, genome_init = self._create_genome(params['num_msats'])
        view_ops = []
        for view in self._views:
            view_ops.extend(view.post_ops)
        for view in self._views:
            post_ops.append(sp.PyOperator(func=_hook_view, param=view))
        post_ops = view_ops + post_ops
        sim = sp.Simulator(pop, 1, True)
        return {'sim': sim, 'pop': pop, 'init_ops': init_ops + genome_init,
                'pre_ops': pre_ops, 'post_ops': post_ops,
                'mating_scheme': sp.RandomMating()}


class LociParameter():
    def __init__(self):
        self.name = None
        self.desc = None
        self.simupop_info = []
        self.simupop_stats = []

    def get_values(self, pop):
        raise NotImplementedError


class ObsHe(LociParameter):
    def __init__(self):
        LociParameter.__init__(self)
        self.name = 'ObsHe'
        self.desc = 'Observed Heterozygozity'
        self.simupop_stats = [sp.Stat(heteroFreq=True)]

    def get_values(self, pop):
        loci = list(pop.dvars().heteroFreq.keys())
        loci.sort()
        return [pop.dvars().heteroFreq[l] for l in loci]


class ExpHe(LociParameter):
    def __init__(self):
        LociParameter.__init__(self)
        self.name = 'ExpHe'
        self.desc = 'Expected Heterozygozity'
        self.simupop_stats = [sp.Stat(alleleFreq=True)]

    def get_values(self, pop):
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


class NumAlleles(LociParameter):
    def __init__(self):
        LociParameter.__init__(self)
        self.name = 'NumAlleles'
        self.desc = 'Number of Alleles'
        self.simupop_stats = [sp.Stat(alleleFreq=True)]

    def get_values(self, pop):
        anum = pop.dvars().alleleNum
        loci = list(anum.keys())
        loci.sort()
        anums = [len(anum[l]) for l in loci]
        return anums


class FST(LociParameter):
    def __init__(self):
        LociParameter.__init__(self)
        self.name = 'FST'
        self.desc = 'FST'
        self.simupop_stats = [sp.Stat(structure=sp.ALL_AVAIL)]

    def get_values(self, pop):
        fst = pop.dvars().F_st
        return [fst]


class LDNe(LociParameter):
    def __init__(self, pcrit=0.02):
        LociParameter.__init__(self)
        self.name = 'LDNe'
        self.desc = 'LDNe'
        self.pcrit = pcrit
        self.simupop_stats = [sp.Stat(effectiveSize=sp.ALL_AVAIL,
                                      vars='Ne_LD')]

    def get_values(self, pop):
        ne = pop.dvars().Ne_LD
        return [ne[self.pcrit]]


class FreqDerived(LociParameter):
    def __init__(self):
        LociParameter.__init__(self)
        self.name = 'FreqDerived'
        self.desc = 'Frequency of the Derived Allele'
        self.simupop_stats = [sp.Stat(alleleFreq=True)]

    def get_values(self, pop):
        anum = pop.dvars().alleleFreq
        loci = list(anum.keys())
        loci.sort()
        anums = [anum[0][1]]
        return anums
