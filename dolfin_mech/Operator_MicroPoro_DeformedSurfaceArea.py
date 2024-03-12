#coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2023                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

# ################################################################################

class DeformedSurfaceAreaOperator(Operator):

    def __init__(self,
            S_area,
            S_area_test,
            kinematics,
            N,
            measure):

        self.measure = measure
        self.kinematics = kinematics
        self.N = N


        FmTN = dolfin.dot(dolfin.inv(self.kinematics.F).T, self.N)
        T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
        S0 = dolfin.assemble(1 * self.measure)
        
        self.res_form = ((S_area/S0 - T*self.kinematics.J) * S_area_test) * self.measure

