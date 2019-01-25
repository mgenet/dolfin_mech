#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_cm as dcm
from Material_Elastic import ElasticMaterial

################################################################################

class HookeElasticMaterial(ElasticMaterial):



    def __init__(self,
            parameters):

        if ("lambda" in parameters) and ("mu" in parameters):
            self.lmbda = dolfin.Constant(parameters["lambda"])
            self.mu    = dolfin.Constant(parameters["mu"])
        elif ("E" in parameters) and ("nu" in parameters):
            self.E  = dolfin.Constant(parameters["E"])
            self.nu = dolfin.Constant(parameters["nu"])
            self.lmbda = self.E*self.nu/(1+self.nu)/(1-2*self.nu) # MG20180516: in 2d, plane strain
            self.mu    = self.E/2/(1+self.nu)



    def get_free_energy(self,
            U=None,
            epsilon=None):

        if (epsilon is None):
            dim = U.ufl_shape[0]
            epsilon = dolfin.sym(dolfin.grad(U))
        else:
            assert (epsilon.ufl_shape[0] == epsilon.ufl_shape[1])
            dim = epsilon.ufl_shape[0]

        Psi = (self.lmbda/2) * dolfin.tr(epsilon)**2 + self.mu * dolfin.inner(epsilon, epsilon)

        I = dolfin.Identity(dim)
        Sigma = self.lmbda * dolfin.tr(epsilon) * I + 2 * self.mu * epsilon

        return Psi, Sigma