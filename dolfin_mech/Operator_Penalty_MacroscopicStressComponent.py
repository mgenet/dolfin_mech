#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2020-2023                                    ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class MacroscopicStressComponentPenaltyOperator(Operator):

    def __init__(self,
            sigma_bar,
            sigma_bar_test,
            sol,
            sol_test,
            material,
            comp_i, comp_j,
            measure,
            comp_val=None, comp_ini=None, comp_fin=None,
            pen_val=None, pen_ini=None, pen_fin=None):

        self.material = material
        self.measure  = measure

        self.tv_comp = dmech.TimeVaryingConstant(
            val=comp_val, val_ini=comp_ini, val_fin=comp_fin)
        comp = self.tv_comp.val

        self.tv_pen = dmech.TimeVaryingConstant(
            val=pen_val, val_ini=pen_ini, val_fin=pen_fin)
        pen = self.tv_pen.val

        Pi = (pen/2) * (self.material.sigma[comp_i,comp_j] - comp)**2 * self.measure # MG20220426: Need to compute <sigma> properly, including fluid pressure
        # self.res_form = dolfin.derivative(Pi, sigma_bar[comp_i,comp_j], sigma_bar_test[comp_i,comp_j]) # MG20230106: This does not work…
        self.res_form = dolfin.derivative(Pi, sol, sol_test) # MG20230106: This works…

        # Pi = (pen/2) * (sigma_bar[comp_i,comp_j] - comp)**2 * self.measure # MG20230106: This does not work…
        # self.res_form = dolfin.derivative(Pi, sigma_bar[comp_i,comp_j], sigma_bar_test[comp_i,comp_j])
        # self.res_form = dolfin.derivative(Pi, sol, sol_test)



    def set_value_at_t_step(self,
            t_step):

        self.tv_comp.set_value_at_t_step(t_step)
        self.tv_pen.set_value_at_t_step(t_step)
