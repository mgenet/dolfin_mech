#coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2023                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2023                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

# ################################################################################

class DeformedSolidVolumeOperator(Operator):

    def __init__(self,
            vs,
            vs_test,
            J,
            Vs0, 
            measure):

        self.Vs0 = dolfin.Constant(Vs0)
        self.measure = measure
        
        self.res_form = ((vs/self.Vs0 - J) * vs_test) * self.measure
