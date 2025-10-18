
import dolfin

import dolfin_mech as dmech
from .Operator import Operator

class WbulkPoroFlowOperator(Operator):

    def __init__(self,
            kinematics,
            U,
            U_test,
            Phis0,
            Phis,
            Phis_test,
            material_parameters,
            material_scaling,
            measure,
            pressure=None
            ):  # new input

        self.kinematics = kinematics
        self.solid_material = dmech.WbulkLungElasticMaterial(
            Phis=Phis,
            Phis0=Phis0,
            parameters=material_parameters)
        self.material = dmech.PorousElasticMaterial(
            solid_material=self.solid_material,
            scaling=material_scaling,
            Phis0=Phis0)
        self.measure = measure

        dE_test = dolfin.derivative(
            self.kinematics.E, U, U_test)

        # self.res_form = dolfin.inner(
        #     self.material.dWbulkdPhis * self.kinematics.J * self.kinematics.C_inv,
        #     dE_test) * self.measure

        #Add pressure coupling
        self.res_form =  dolfin.inner(
            pressure * self.kinematics.J * self.kinematics.C_inv,
            dE_test) * self.measure

        self.res_form += self.material.dWbulkdPhis * Phis_test * self.measure