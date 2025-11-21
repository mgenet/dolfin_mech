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
                 p_bar,
                 K_l,
                 rho_l,
                 Theta_in,
                 Theta_out,
                 dx,
                 dx_in,
                 dx_out,
                 grad_p_bar_val, 
                 grad_p_bar_x_ini, 
                 grad_p_bar_x_fin,
                 grad_p_bar_y_ini, 
                 grad_p_bar_y_fin,
                 X,
                 X_0,
                 Phis_test):
        
        self.measure = dx
        
        # --- TimeVaryingConstant for grad p_bar components ---
        self.tv_grad_p_bar_x = dmech.TimeVaryingConstant(
            #val=grad_p_bar_x_val, 
            val_ini=grad_p_bar_x_ini, 
            val_fin=grad_p_bar_x_fin
        )

        self.tv_grad_p_bar_y = dmech.TimeVaryingConstant(
            #val=grad_p_bar_y_val, 
            val_ini=grad_p_bar_y_ini, 
            val_fin=grad_p_bar_y_fin
        )

        # --- Assemble vector ∇p̄ = (∂p/∂x , ∂p/∂y) ---
        self.grad_p_bar = dolfin.as_vector((
            self.tv_grad_p_bar_x.val,
            self.tv_grad_p_bar_y.val
        ))

        # macroscopic mean pressure (scalar constant)
        self.p_bar = p_bar

        # --- Total pressure ---
        self.p_tot = (
            self.p_bar 
            + dolfin.dot(self.grad_p_bar, X - X_0)
            + p
        )


        grad_p = dolfin.grad(p)
        grad_p_test = dolfin.grad(p_test)

        F = kinematics.F
        J = kinematics.J
        k_l = (1.0 / J) * F * K_l * F.T  
        self.K_l = K_l  
        self.k_l = k_l  
        self.J = J

        grad_p = dolfin.grad(p)
        grad_p_test = dolfin.grad(p_test)

        # --- Darcy flow residual with Phis_test term ---
        self.res_form = rho_l * dolfin.inner(k_l * (grad_p + self.grad_p_bar), grad_p_test) * dx + dolfin.inner(self.p_tot, Phis_test) * dx
        if Theta_in != 0.0:
            self.res_form -= Theta_in * p_test * dx_in
        if Theta_out != 0.0:
            self.res_form += Theta_out * p_test * dx_out

    def set_value_at_t_step(self, t_step):

        self.tv_grad_p_bar_x.set_value_at_t_step(t_step)
        self.tv_grad_p_bar_y.set_value_at_t_step(t_step)

class PfFieldOperator(Operator):
    def __init__(self,
                 p_tot,
                 pressure,
                 Phis_test,
                 measure):
        self.measure = measure
        self.pf = pressure  # pressure field (Function)
        #self.res_form = dolfin.inner(self.pf, Phis_test) * self.measure
        self.res_form = dolfin.inner(p_tot, Phis_test) * self.measure