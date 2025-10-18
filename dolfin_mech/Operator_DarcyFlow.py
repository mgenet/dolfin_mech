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

# class DarcyFlowOperator(Operator):

#     def __init__(self,
#             p,
#             p_test,
#             K_l,
#             rho_l,
#             Theta_0,
#             measure):

#         self.measure = measure

#         grad_p = dolfin.grad(p)
#         grad_p_test = dolfin.grad(p_test)

#         self.res_form = (
#             rho_l * dolfin.dot(K_l * grad_p, grad_p_test) * self.measure
#             - Theta_0 * p_test * self.measure
#         )

class DarcyFlowOperator(Operator):
    def __init__(self,
                 kinematics,
                 p,
                 p_test,
                 K_l,
                 rho_l,
                 Theta_in,
                 Theta_out,
                 dx,
                 dx_in,
                 dx_out):
        
        
        print("[DBG] Darcy dx =", dx)
        print("[DBG] Darcy dx_in =", dx_in)
        print("[DBG] Darcy dx_out =", dx_out)

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

        print("F shape =", F.ufl_shape)
        print("K_l shape =", K_l.ufl_shape)
        print("k_l shape =", k_l.ufl_shape)
        print("grad_p shape =", grad_p.ufl_shape)
        grad_p = dolfin.grad(p)
        grad_p_test = dolfin.grad(p_test)

        # --- Darcy flow residual (standard diffusion-like form) ---
        self.res_form = rho_l * dolfin.inner(k_l * grad_p, grad_p_test) * dx
        if Theta_in != 0.0:
            self.res_form -= Theta_in * p_test * dx_in
        if Theta_out != 0.0:
            self.res_form += Theta_out * p_test * dx_out

        vol_in = dolfin.assemble(1.0 * dx_in)
        vol_out = dolfin.assemble(1.0 * dx_out)
        print("Inlet region volume =", vol_in)
        print("Outlet region volume =", vol_out)
       
        Q_in  = dolfin.assemble(Theta_in * dx_in)
        Q_out = dolfin.assemble(Theta_out * dx_out)
        print("Injected volume rate =", Q_in)
        print("Extracted volume rate =", Q_out)



        #Darcy residual: standard diffusion + source and sink
        # self.res_form = (
        #     rho_l * dolfin.dot(K_l * grad_p, grad_p_test) * self.measure
        #     - Theta_in * p_test * dx_in
        #     + Theta_out * p_test * dx_out
        # )


class PfFieldOperator(Operator):
    def __init__(self,
                 pressure,
                 Phis_test,
                 measure):
        self.measure = measure
        self.pf = pressure  # pressure field (Function)
        self.res_form = dolfin.inner(self.pf, Phis_test) * self.measure