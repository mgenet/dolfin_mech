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
                 Theta_in=None,
                 Theta_out=None):

        if Theta_in is None:
            Theta_in = dolfin.Constant(0.0)
        if Theta_out is None:
            Theta_out = dolfin.Constant(0.0)
        
        

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

class WskelPoroFlowOperator(Operator):

    def __init__(self,
            kinematics,
            U,
            U_test,
            material_parameters,
            material_scaling,
            Phis0,
            measure):

        self.kinematics = kinematics
        self.solid_material = dmech.WskelLungElasticMaterial(
            kinematics=kinematics,
            parameters=material_parameters)
        self.material = dmech.PorousElasticMaterial(
            solid_material=self.solid_material,
            scaling=material_scaling,
            Phis0=Phis0)
        self.measure = measure

        dE_test = dolfin.derivative(
            self.kinematics.E, U, U_test)
        self.res_form = dolfin.inner(self.material.Sigma, dE_test) * self.measure

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
            -pl * self.kinematics.J * self.kinematics.C_inv,
            dE_test) * self.measure

        self.res_form += self.material.dWbulkdPhis * Phis_test * self.measure


class WbulkMicroPoroFlowOperator(Operator):

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
        self.res_form += self.material.dWbulkdPhis * Phis_test * self.measure

class MicroDarcyFlowOperator(Operator):
    def __init__(self,
                 kinematics,
                 U,
                 U_test,
                 X, 
                 X_0,
                 unknown_porosity_test,
                 p_tilde,
                 grad_p_bar_ini,
                 grad_p_bar_fin,
                 pl_bar,
                 p_test,
                 K_l,
                 rho_l,
                 dx,
                 dx_in,
                 dx_out,
                 Theta_in_ini=None,
                 Theta_in_fin=None,
                 Theta_out_ini=None,
                 Theta_out_fin=None):

        dE_test = dolfin.derivative(
            kinematics.E, U, U_test)

        gx_ini, gy_ini = grad_p_bar_ini
        gx_fin, gy_fin = grad_p_bar_fin

        print("DarcyFlowOperator: grad_p_bar_ini =", (gx_ini, gy_ini))
        print("DarcyFlowOperator: grad_p_bar_fin =", (gx_fin, gy_fin))

        # --- TimeVaryingConstant for Theta ---
        self.tv_Theta_in  = dmech.TimeVaryingConstant(val_ini=Theta_in_ini,  val_fin=Theta_in_fin)
        self.tv_Theta_out = dmech.TimeVaryingConstant(val_ini=Theta_out_ini, val_fin=Theta_out_fin)

        # --- TimeVaryingConstant for grad p_bar components ---
        self.tv_grad_p_bar_x = dmech.TimeVaryingConstant(val_ini=gx_ini, val_fin=gx_fin)
        self.tv_grad_p_bar_y = dmech.TimeVaryingConstant(val_ini=gy_ini, val_fin=gy_fin)

        # --- Assemble vector ∇p̄ (2D) ---
        self.grad_p_bar = dolfin.as_vector((
            self.tv_grad_p_bar_x.val,
            self.tv_grad_p_bar_y.val
        ))

        self.pl_bar= pl_bar
        self.pl_tot = ( self.pl_bar + dolfin.dot(self.grad_p_bar, X - X_0) + p_tilde)
        self.measure = dx  
        self.kinematics = kinematics

        self.grad_p_tilde = dolfin.grad(p_tilde)
        grad_p_test = dolfin.grad(p_test)

        F = self.kinematics.F
        J = self.kinematics.J
        k_l = (1.0 / J) * F * K_l * F.T  # current configuration permeability
        self.K_l = K_l  # keep reference permeability for output
        self.k_l = k_l  # keep current permeability for output
        self.J = J

        # --- Darcy flow residual (standard diffusion-like form) ---
        self.res_form = rho_l * dolfin.inner(k_l * dolfin.inv(kinematics.F) * (self.grad_p_bar+self.grad_p_tilde), grad_p_test) * dx
        # form pl_field operator#
        self.res_form += dolfin.inner(self.pl_tot, unknown_porosity_test) * self.measure
        # form wbulk operator#
        self.res_form +=  dolfin.inner(
            -self.pl_tot * self.kinematics.J * self.kinematics.C_inv,
            dE_test) * self.measure


        # if Theta_in != 0.0:
        #     self.res_form -= Theta_in * p_test * dx_in
        # if Theta_out != 0.0:
        #     self.res_form += Theta_out * p_test * dx_out




    def set_value_at_t_step(self, t_step):
        self.tv_grad_p_bar_x.set_value_at_t_step(t_step)
        self.tv_grad_p_bar_y.set_value_at_t_step(t_step)
        self.tv_Theta_in.set_value_at_t_step(t_step)
        self.tv_Theta_out.set_value_at_t_step(t_step)
