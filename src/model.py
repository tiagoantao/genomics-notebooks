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
import simuOpt
simuOpt.setOptions(gui=False, quiet=True)
import simuPOP as sp


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
        self._stats = set()
        self._info_fields = set()
        self._sim_ids = []
        self._sims = []

    def register(self, view):
        self._views.append(view)

    def add_stat(self, stat):
        self._stats.add(stat)

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
        return pop, init_ops

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
        for view in self._views:
            for info in view.info_fields:
                self._info_fields.add(info)
        pop, init_ops = self._create_single_pop(params['pop_size'],
                                                params['num_msats'])
        loci, genome_init = self._create_genome(params['num_msats'])
        pre_ops = []
        view_ops = []
        post_ops = []
        for view in self._views:
            view_ops.extend(view.post_ops)
        for view in self._views:
            post_ops.append(sp.PyOperator(func=_hook_view, param=view))
        post_ops = view_ops + post_ops
        sim = sp.Simulator(pop, 1, True)
        return {'sim': sim, 'pop': pop, 'init_ops': init_ops + genome_init,
                'pre_ops': pre_ops, 'post_ops': post_ops}

    def _run(self, sim_id, params):
        pr = self.prepare_sim(params)
        sim = pr['sim']
        for view in self._views:
            view.set_sim_id(sim_id)
        sim.evolve(initOps=pr['init_ops'],
                   preOps=pr['pre_ops'],
                   postOps=pr['post_ops'],
                   matingScheme=sp.RandomMating(),
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


class LociParameter():
    def __init__(self):
        self.name = None
        self.simupop_info = []
        self.simupop_stats = []

    def get_values(self, pop):
        raise NotImplementedError


class ObsHe(LociParameter):
    def __init__(self):
        LociParameter.__init__(self)
        self.name = 'ObsHe'
        self.simupop_stats = [sp.Stat(heteroFreq=True)]

    def get_values(self, pop):
        loci = list(pop.dvars().heteroFreq.keys())
        loci.sort()
        return [pop.dvars().heteroFreq[l] for l in loci]


class ExpHe(LociParameter):
    def __init__(self):
        LociParameter.__init__(self)
        self.name = 'ExpHe'
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
