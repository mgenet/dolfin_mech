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

class DirectionalDisplacementPenaltyOperator(Operator):

    def __init__(self,
            U,
            U_test,
            measure,
            N_val=None,  N_ini=None, N_fin=None,
            pen_val=None, pen_ini=None, pen_fin=None):

        self.measure = measure

        self.tv_N = dmech.TimeVaryingConstant(
            val=N_val, val_ini=N_ini, val_fin=N_fin)
        N = self.tv_N.val

        self.tv_pen = dmech.TimeVaryingConstant(
            val=pen_val, val_ini=pen_ini, val_fin=pen_fin)
        pen = self.tv_pen.val

        Pi = (pen/2) * dolfin.inner(U, N)**2 * self.measure
        self.res_form = dolfin.derivative(Pi, U, U_test)

        # self.res_form = pen * dolfin.inner(U     , N)\
        #                     * dolfin.inner(U_test, N) * self.measure



    def set_value_at_t_step(self,
            t_step):

        self.tv_N.set_value_at_t_step(t_step)
        self.tv_pen.set_value_at_t_step(t_step)
