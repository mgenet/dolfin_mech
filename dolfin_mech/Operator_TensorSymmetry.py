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

class TensorSymmetryOperator(Operator):

    def __init__(self,
            tensor,
            tensor_test,
            measure):

        self.measure = measure
        self.res_form = dolfin.inner(tensor.T - tensor, tensor_test) * self.measure
