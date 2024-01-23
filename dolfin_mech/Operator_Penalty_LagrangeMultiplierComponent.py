#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2020-2024                                    ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class LagrangeMultiplierComponentPenaltyOperator(Operator):

    def __init__(self,
            lambda_bar,
            lambda_bar_test,
            i, j,
            measure,
            pen_val=None, pen_ini=None, pen_fin=None):

        self.measure = measure

        self.tv_pen = dmech.TimeVaryingConstant(
            val=pen_val, val_ini=pen_ini, val_fin=pen_fin)
        pen = self.tv_pen.val

        self.res_form = pen * lambda_bar[i,j] * lambda_bar_test[i,j] * self.measure



    def set_value_at_t_step(self,
            t_step):

        self.tv_pen.set_value_at_t_step(t_step)
