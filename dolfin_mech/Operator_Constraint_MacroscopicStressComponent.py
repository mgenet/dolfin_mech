#coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2024                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2024                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class MacroscopicStressComponentConstraintOperator(Operator):

    def __init__(self,
            U_bar, U_bar_test,
            kinematics,
            material,
            V0, Vs0,
            i, j,
            measure,
            N,
            sigma_bar_ij_val=None, sigma_bar_ij_ini=None, sigma_bar_ij_fin=None,
            pf_val=None, pf_ini=None, pf_fin=None):

        self.kinematics = kinematics
        self.material = material
        self.measure  = measure
        self.N = N

        self.tv_pf = dmech.TimeVaryingConstant(
            val=pf_val, val_ini=pf_ini, val_fin=pf_fin)
        pf = self.tv_pf.val

        self.tv_sigma_bar_ij = dmech.TimeVaryingConstant(
            val=sigma_bar_ij_val, val_ini=sigma_bar_ij_ini, val_fin=sigma_bar_ij_fin)
        sigma_bar_ij = self.tv_sigma_bar_ij.val

        dim = U_bar.ufl_shape[0]
        I_bar = dolfin.Identity(dim)
        F_bar = I_bar + U_bar
        J_bar = dolfin.det(F_bar)
        v = J_bar * V0

        sigma_tilde = self.material.sigma * self.kinematics.J - (v/Vs0 - self.kinematics.J) * pf * I_bar
        self.res_form = U_bar_test[i,j] * (sigma_tilde[i,j] - (v/Vs0) * sigma_bar_ij) * self.measure



    def set_value_at_t_step(self,
            t_step):

        self.tv_pf.set_value_at_t_step(t_step)
        self.tv_sigma_bar_ij.set_value_at_t_step(t_step)
