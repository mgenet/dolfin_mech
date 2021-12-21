#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Cécile Patte, 2019-2022                                              ###
###                                                                          ###
### INRIA, Palaiseau, France                                                 ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Material_Elastic_Bulk import BulkElasticMaterial

################################################################################

class PneumoBulkElasticMaterial(BulkElasticMaterial):



    def __init__(self,
            parameters):

        if ("alpha" in parameters) and ("gamma" in parameters):
            self.alpha = dolfin.Constant(parameters["alpha"])
            self.gamma = dolfin.Constant(parameters["gamma"])
        else:
            assert (0),\
                "No parameter found: \"+str(parameters)+\". Must provide alpha & gamma. Aborting."



    def get_free_energy(self,
            U=None,
            C=None):

        C  = self.get_C_from_U_or_C(U, C)
        JF = dolfin.sqrt(dolfin.det(C)) # MG20200207: Watch out! This is well defined for inverted elements!

        Psi = (self.alpha) * (dolfin.exp(self.gamma*(JF**2 - 1 - 2*dolfin.ln(JF))) - 1)

        C_inv = dolfin.inv(C)
        Sigma = (self.alpha) *  dolfin.exp(self.gamma*(JF**2 - 1 - 2*dolfin.ln(JF))) * (2*self.gamma) * (JF**2 - 1) * C_inv

        return Psi, Sigma
