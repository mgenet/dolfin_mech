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

class DeformedFluidVolumeOperator(Operator):

    def __init__(self,
            vf,
            vf_test,
            kinematics,
            N,
            dS,
            U_tot,
            X, 
            measure):

        self.kinematics = kinematics
        self.U_tot = U_tot
        self.X = X
        self.N = N
        self.dS = dS
        self.measure = measure

        PN = self.kinematics.J * dolfin.dot(dolfin.inv(self.kinematics.F.T), self.N)
        
        self.res_form = ((vf + dolfin.assemble(dolfin.inner(self.U_tot + self.X, PN) * self.dS(0))/2) * vf_test) * self.measure # MG20230203: This is not correct. Need to check.
