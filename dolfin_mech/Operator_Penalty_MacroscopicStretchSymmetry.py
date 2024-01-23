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

class MacroscopicStretchSymmetryPenaltyOperator(Operator):

    def __init__(self,
            U_bar,
            sol,
            sol_test,
            measure,
            pen_val=None, pen_ini=None, pen_fin=None):

        self.measure = measure

        self.tv_pen = dmech.TimeVaryingConstant(
            val=pen_val, val_ini=pen_ini, val_fin=pen_fin)
        pen = self.tv_pen.val

        Pi = (pen/2) * dolfin.inner(U_bar.T - U_bar, U_bar.T - U_bar) * self.measure
        # self.res_form = dolfin.derivative(Pi, U_bar, U_bar_test) # MG20230106: Somehow this does not work… NotImplementedError("Cannot take length of non-vector expression.")
        self.res_form = dolfin.derivative(Pi, sol, sol_test)



    def set_value_at_t_step(self,
            t_step):

        self.tv_pen.set_value_at_t_step(t_step)
