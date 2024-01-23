#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class PfPoroOperator(Operator):

    def __init__(self,
            Phis_test,
            measure,
            pf_val=None, pf_ini=None, pf_fin=None):

        self.measure = measure

        self.tv_pf = dmech.TimeVaryingConstant(
            val=pf_val, val_ini=pf_ini, val_fin=pf_fin)
        self.pf = self.tv_pf.val

        self.res_form = dolfin.inner(self.pf, Phis_test) * self.measure



    def set_value_at_t_step(self,
            t_step):

        self.tv_pf.set_value_at_t_step(t_step)
