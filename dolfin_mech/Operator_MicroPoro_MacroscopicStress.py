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

class MacroscopicStressOperator(Operator):

    def __init__(self,
            sigma_bar,
            sigma_bar_test,
            vs,
            v,
            Vs0,
            kinematics,
            material,
            measure,
            P_val=None, P_ini=None, P_fin=None):

        self.material = material
        self.measure  = measure

        self.tv_P = dmech.TimeVaryingConstant(
            val=P_val, val_ini=P_ini, val_fin=P_fin)
        P = self.tv_P.val

        self.res_form = dolfin.inner(sigma_bar * v/Vs0 - self.material.sigma * kinematics.J + (v - vs)/Vs0 * P * dolfin.Identity(2), sigma_bar_test) * self.measure # MG20220426: Need to compute <sigma> properly, including fluid pressure


    def set_value_at_t_step(self,
        t_step):
        
        self.tv_P.set_value_at_t_step(t_step)