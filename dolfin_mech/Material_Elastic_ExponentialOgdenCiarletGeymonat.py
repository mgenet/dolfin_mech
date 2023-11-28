#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Cécile Patte, 2019-2021                                              ###
###                                                                          ###
### INRIA, Palaiseau, France                                                 ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Material_Elastic import ElasticMaterial

################################################################################

class ExponentialOgdenCiarletGeymonatElasticMaterial(ElasticMaterial):



    def __init__(self,
            kinematics,
            parameters):

        self.kinematics = kinematics

        if ("alpha" in parameters) and ("gamma" in parameters):
            self.alpha = dolfin.Constant(parameters["alpha"])
            self.gamma = dolfin.Constant(parameters["gamma"])
        else:
            assert (0),\
                "No parameter found: \"+str(parameters)+\". Must provide alpha & gamma. Aborting."

        self.Psi = (self.alpha) * (dolfin.exp(self.gamma*(self.kinematics.J**2 - 1 - 2*dolfin.ln(self.kinematics.J))) - 1)

        self.Sigma = (self.alpha) * dolfin.exp(self.gamma*(self.kinematics.J**2 - 1 - 2*dolfin.ln(self.kinematics.J))) * (2*self.gamma) * (self.kinematics.J**2 - 1) * self.kinematics.C_inv

        # self.P = dolfin.diff(self.Psi, self.kinematics.F) # MG20220426: Cannot do that for micromechanics problems
        self.P = self.kinematics.F * self.Sigma

        self.sigma = self.P * self.kinematics.F.T / self.kinematics.J
