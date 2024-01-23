#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class HyperElasticityOperator(Operator):

    def __init__(self,
            U,
            U_test,
            kinematics,
            material_model,
            material_parameters,
            measure,
            formulation="PK1"): # PK1 or PK2 or ener

        self.kinematics = kinematics
        self.material   = dmech.material_factory(kinematics, material_model, material_parameters)
        self.measure    = measure

        assert (formulation in ("PK1", "PK2", "ener")),\
            "\"formulation\" should be \"PK1\", \"PK2\" or \"ener\". Aborting."

        if (formulation == "ener"):
            self.res_form = dolfin.derivative(self.material.Psi, U, U_test) * self.measure
        elif (formulation == "PK2"):
            dE_test = dolfin.derivative(
                self.kinematics.E, U, U_test)
            self.res_form = dolfin.inner(self.material.Sigma, dE_test) * self.measure
        elif (formulation == "PK1"):
            dF_test = dolfin.derivative(
                self.kinematics.F, U, U_test)
            self.res_form = dolfin.inner(self.material.P, dF_test) * self.measure
