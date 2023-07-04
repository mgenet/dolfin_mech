#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2021-2023                                    ###
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
            mesh_V0,
            mesh_bbox_V0,
            sigma_bar, sigma_bar_test,
            sol, sol_test,
            material,
            measure,
            pen_val=None, pen_ini=None, pen_fin=None):

        self.material = material
        self.measure  = measure

        # self.tv_pen = dmech.TimeVaryingConstant(
        #     val=pen_val, val_ini=pen_ini, val_fin=pen_fin)
        # pen = self.tv_pen.val

        # Pi = (pen/2) * dolfin.inner((mesh_bbox_V0/mesh_V0) * sigma_bar - self.material.sigma, (mesh_bbox_V0/mesh_V0) * sigma_bar - self.material.sigma) * self.measure # MG20220426: Need to compute <sigma> properly, including fluid pressure # MG20230103: This does not work…
        # self.res_form = dolfin.derivative(Pi, sol, sol_test)

        self.res_form = dolfin.inner((mesh_bbox_V0/mesh_V0) * sigma_bar - self.material.sigma, sigma_bar_test) * self.measure # MG20220426: Need to compute <sigma> properly, including fluid pressure



    def set_value_at_t_step(self,
            t_step):

        self.tv_pen.set_value_at_t_step(t_step)
