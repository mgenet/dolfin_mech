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

class PorousElasticMaterial(ElasticMaterial):



    def __init__(self,
            solid_material,
            scaling="no",
            Phis0=None):

        self.solid_material = solid_material

        if (scaling == "no"):
            scaling = dolfin.Constant(1)
            # self.Psi   = self.material.Psi
            # if (hasattr(self.material,       "Sigma")): self.Sigma       = self.material.Sigma
            # if (hasattr(self.material,           "P")): self.P           = self.material.P
            # if (hasattr(self.material,       "sigma")): self.sigma       = self.material.sigma
            # if (hasattr(self.material, "dWbulkdPhis")): self.dWbulkdPhis = self.material.dWbulkdPhis
        elif (scaling == "linear"):
            scaling = Phis0
            # self.Psi   = Phis0 * self.material.Psi
            # if (hasattr(self.material,       "Sigma")): self.Sigma       = Phis0 * self.material.Sigma
            # if (hasattr(self.material,           "P")): self.P           = Phis0 * self.material.P
            # if (hasattr(self.material,       "sigma")): self.sigma       = Phis0 * self.material.sigma
            # if (hasattr(self.material, "dWbulkdPhis")): self.dWbulkdPhis = Phis0 * self.material.dWbulkdPhis
        else:
            assert (0),\
                "scaling must be \"no\" or \"linear\". Aborting."

        for attr in ("Psi", "Sigma", "P", "sigma", "dWbulkdPhis", "dWporedPhif"):
            if (hasattr(self.solid_material, attr)):
                setattr(self, attr, scaling * getattr(self.solid_material, attr))
