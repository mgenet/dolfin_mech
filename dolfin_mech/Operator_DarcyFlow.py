#coding=utf8

################################################################################
###                                                                          ###
### Created by Haotian XIAO, 2024-2027                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class DarcyFlowOperator(Operator):
    def __init__(self,
                 kinematics,
                 p,
                 p_test,
                 K_l,
                 rho_l,
                 dx,
                 dx_in,
                 dx_out,
                 Theta_in=dolfin.Constant(0.0),
                 Theta_out=dolfin.Constant(0.0)):
        
        

        assert dx is not None, "You must provide a global measure dx."
        assert dx_in is not None and dx_out is not None, "You must provide inlet and outlet subdomain measures."

        self.measure = dx  # typically dx(0) or full domain
        self.kinematics = kinematics

        grad_p = dolfin.grad(p)
        grad_p_test = dolfin.grad(p_test)

        F = self.kinematics.F
        J = self.kinematics.J
        # K_l : permeability tensor in reference config (material)
        k_l = (1.0 / J) * F * K_l * F.T  # current configuration permeability
        self.K_l = K_l  # keep reference permeability for output
        self.k_l = k_l  # keep current permeability for output
        self.J = J

        grad_p = dolfin.grad(p)
        grad_p_test = dolfin.grad(p_test)

        # --- Darcy flow residual (standard diffusion-like form) ---
        self.res_form = rho_l * dolfin.inner(k_l * dolfin.inv(kinematics.F) * grad_p, grad_p_test) * dx
        if Theta_in != 0.0:
            self.res_form -= Theta_in * p_test * dx_in
        if Theta_out != 0.0:
            self.res_form += Theta_out * p_test * dx_out





class PlFieldOperator(Operator):
    def __init__(self,
                 pl,
                 unknown_porosity_test,
                 measure):
        self.measure = measure
        self.res_form = dolfin.inner(pl, unknown_porosity_test) * self.measure

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
            pl
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

        self.res_form =  dolfin.inner(
            pl * self.kinematics.J * self.kinematics.C_inv,
            dE_test) * self.measure

        self.res_form += self.material.dWbulkdPhis * Phis_test * self.measure