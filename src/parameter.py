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


class GenomicParameter(Parameter):
    def __init__(self, **kwargs):
        Parameter.__init__(self, kwargs)


class PCA(GenomicParameter):
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
