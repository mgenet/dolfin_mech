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
###                                                                          ###
### And Colin Laville, 2021-2022                                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Material_Elastic import ElasticMaterial

################################################################################

class WporeLungElasticMaterial(ElasticMaterial):



    def __init__(self,
            Phif,
            Phif0,
            parameters):

        assert ('eta' in parameters)
        self.eta = dolfin.Constant(parameters['eta'])

        self.n = dolfin.Constant(parameters.get('n', 1))
        self.p = dolfin.Constant(parameters.get('p', 1))
        self.q = dolfin.Constant(parameters.get('q', 1))

        Phif = dolfin.variable(Phif)
        r = Phif/Phif0
        r_inf = Phif0**(self.p-1)
        r_sup = Phif0**(1/self.q-1)
        self.Psi = self.eta * dolfin.conditional(dolfin.lt(r, 0.), r/dolfin.Constant(0.), dolfin.conditional(dolfin.lt(r, r_inf), (r_inf/r - 1)**(self.n+1), dolfin.conditional(dolfin.gt(r, r_sup), (r/r_sup - 1)**(self.n+1), 0)))
        self.dWporedPhif = dolfin.diff(self.Psi, Phif)
