# -*- coding: utf-8 -*-
'''
.. module:: genomics
   :synopsis: PopGen classes with simulations
   :noindex:
   :copyright: Copyright 2014 by Tiago Antao
   :license: GNU Affero, see LICENSE for details

.. moduleauthor:: Tiago Antao <tra@popgen.net>

'''

from collections import defaultdict

import numpy as np

from sklearn.decomposition import PCA as pca

import simuOpt
simuOpt.setOptions(gui=False, quiet=True)
import simuPOP as sp
from simuPOP import sampling


def _get_sub_sample(pop, size, sub_pop=None):
    '''Gets a subsample of individuals.'''
    if sub_pop is None:
        pop_s = pop
    else:
        pop_s = pop.extractSubPops(subPops=[sub_pop])
    if size is None:
        return pop_s
    pop_s = sampling.drawRandomSample(pop_s, sizes=size)
    return pop_s


class Parameter:
    '''A simulation parameter. Absctract super-class.'''
    def __init__(self, do_structured=False):
        self.name = None
        self.desc = None
        self.do_structured = do_structured
        self._sample_size = None
        self._simupop_stats = []
        self._info_fields = []
        self._pop = None

    def _get_values(self, pop):
        '''Returns the parameter values for a certain subpopulation.

           Implemented on concrete class.
        '''
        raise NotImplementedError('Needs to be implemented')

    def get_values(self, pop, sub_pop=None):
        '''Returns the parameter values for a certain subpopulation.'''
        if self.do_structured:
            pop_ = _get_sub_sample(pop, self.sample_size, sub_pop)
        else:
            pop_ = _get_sub_sample(pop, self.sample_size)
        ind_values = self._get_values(pop_)
        return ind_values

    @property
    def pop(self, pop):
        '''Population'''
        self._pop = pop

    @pop.setter
    def pop(self, value):
        '''Population setter.'''
        self._pop = value

    @property
    def simupop_stats(self):
        '''Statistics that simupop needs to compute for this parameter.

        This is normally added to evolve postOps.'''
        return self._simupop_stats

    @property
    def sample_size(self):
        '''Parameter sample size.'''
        return self._sample_size

    @sample_size.setter
    def sample_size(self, value):
        '''Sample size setter.'''
        self._sample_size = value

    @property
    def info_fields(self):
        '''Fields that need to be available on the Population object'''
        return self._info_fields

    @info_fields.setter
    def info_fields(self, value):
        '''Info_fields setter.'''
        self._info_fields = value



class ObsHe(Parameter):
    '''Observed Heterozygosity'''
    def __init__(self):
        Parameter.__init__(self)
        self.name = 'ObsHe'
        self.desc = 'Observed Heterozygozity'

    def _get_values(self, pop):
        stat = sp.Stat(heteroFreq=True)
        stat.apply(pop)
        loci = list(pop.dvars().heteroFreq.keys())
        loci.sort()
        return [pop.dvars().heteroFreq[l] for l in loci]


class ExpHe(Parameter):
    '''Expected Heterozygosity'''
    def __init__(self, **kwargs):
        Parameter.__init__(self, kwargs)
        self.name = 'ExpHe'
        self.desc = 'Expected Heterozygozity'

    def _get_values(self, pop):
        stat = sp.Stat(alleleFreq=True)
        stat.apply(pop)
        freqs = pop.dvars().alleleFreq
        loci = list(freqs.keys())
        loci.sort()
        exp_he = []
        for locus in loci:
            afreqs = freqs[locus]
            exp_ho = 0
            for freq in afreqs.values():
                exp_ho += freq * freq
            exp_he.append(1 - exp_ho)
        return exp_he


class NumAlleles(Parameter):
    '''Number of Alleles'''
    def __init__(self, **kwargs):
        Parameter.__init__(self, kwargs)
        self.name = 'NumAlleles'
        self.desc = 'Number of Alleles'

    def _get_values(self, pop):
        stat = sp.Stat(alleleFreq=True)
        stat.apply(pop)
        anum = pop.dvars().alleleNum
        loci = list(anum.keys())
        loci.sort()
        anums = [len(anum[l]) for l in loci]
        return anums


class LDNe(Parameter):
    '''Estimating Ne according to LD (Waples)'''
    def __init__(self, pcrit=0.02, **kwargs):
        Parameter.__init__(self, kwargs)
        self.name = 'LDNe'
        self.desc = 'LDNe'
        self.pcrit = pcrit

    def _get_values(self, pop):
        stat = sp.Stat(effectiveSize=sp.ALL_AVAIL, vars='Ne_LD')
        stat.apply(pop)
        ne_est = pop.dvars().Ne_LD
        return ne_est[self.pcrit]


class FreqDerived(Parameter):
    '''Frequency of the derived allele.'''
    def __init__(self, **kwargs):
        Parameter.__init__(self, kwargs)
        self.name = 'FreqDerived'
        self.desc = 'Frequency of the Derived Allele'

    def _get_values(self, pop):
        stat = sp.Stat(alleleFreq=True)
        stat.apply(pop)
        anum = pop.dvars().alleleFreq
        loci = list(anum.keys())
        loci.sort()
        anums = [anum[l][1] for l in loci]
        return anums


class StructuredParameter(Parameter):
    '''A parameter that is applied to population structure.'''
    def __init__(self, **kwargs):
        kwargs['do_structured'] = True
        Parameter.__init__(self, kwargs)


class FST(StructuredParameter):
    '''Mean FST.'''
    def __init__(self, **kwargs):
        StructuredParameter.__init__(self)
        self.name = 'FST'
        self.desc = 'FST'

    def _get_values(self, pop):
        stat = sp.Stat(structure=sp.ALL_AVAIL)
        stat.apply(pop)
        my_fst = pop.dvars().F_st
        return [my_fst]


class fst(StructuredParameter):
    '''FST per locus.'''
    def __init__(self):
        StructuredParameter.__init__(self)
        self.name = 'fst'
        self.desc = 'FST per locus'

    def _get_values(self, pop):
        st = sp.Stat(structure=sp.ALL_AVAIL, vars=['f_st'])
        st.apply(pop)
        fsts = pop.dvars().f_st
        loci = list(fsts.keys())
        return [fsts[l] for l in loci]


class IndividualParameter(Parameter):
    '''A Parameter that returns a value per individual'''
    def __init__(self):
        Parameter.__init__(self)


class PCA(IndividualParameter):
    '''Principal Components Analysis.'''
    def __init__(self):
        IndividualParameter.__init__(self)
        self.info_fields = ['ind_id']

    def _get_values(self, pop):
        nsp = pop.numSubPop()
        all_alleles = []
        for subpop in range(nsp):
            for ind in pop.individuals(subPop=subpop):
                geno = ind.genotype()
                n_markers = len(geno) // 2
                for mi in range(n_markers):
                    if len(all_alleles) <= mi:
                        all_alleles.append(set())
                    a1 = geno[mi]
                    a2 = geno[mi + n_markers]
                    all_alleles[mi].add(a1)
                    all_alleles[mi].add(a2)
        for i, alleles in enumerate(all_alleles):
            all_alleles[i] = sorted(list(alleles))
        inds = defaultdict(list)
        for mi in range(n_markers):
            for subpop in range(nsp):
                for i, ind in enumerate(pop.individuals(subPop=subpop)):
                    geno = ind.genotype()
                    a1 = geno[mi]
                    a2 = geno[mi + n_markers]
                    for a in all_alleles[mi]:
                        inds[(subpop, i)].append([a1, a2].count(a))
        ind_order = sorted(list(inds.keys()))
        arr = []
        for ind in ind_order:
            arr.append(inds[ind])
        my_pca = pca(n_components=2)
        X = np.array(arr)
        X_r = my_pca.fit(X).transform(X)
        my_components = {}
        for i, ind in enumerate(ind_order):
            my_components[ind] = X_r[i]
        return my_components
