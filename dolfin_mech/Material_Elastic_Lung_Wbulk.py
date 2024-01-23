#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
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

class WbulkLungElasticMaterial(ElasticMaterial):



    def __init__(self,
            Phis,
            Phis0,
            parameters):

        assert ('kappa' in parameters)
        self.kappa = dolfin.Constant(parameters['kappa'])

        Phis = dolfin.variable(Phis)
        self.Psi = self.kappa * (Phis/Phis0 - 1 - dolfin.ln(Phis/Phis0))
        self.dWbulkdPhis = dolfin.diff(self.Psi, Phis)
