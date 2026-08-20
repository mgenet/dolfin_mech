################################################################################
###                                                                          ###
### Created by Haotian Xiao, 2024-2027                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################
class DarcyFlowOperator(Operator):
    """
    Darcy operator written in the reference configuration.

    Outputs:
      - k_l_intr : intrinsic permeability tensor (current config)
      - K_l      : pulled-back permeability tensor (reference config)
      - q_l      : Darcy flux (current config)
      - Q_l      : Piola flux (reference config)
    """

    def __init__(self,
                 # --- kinematics / mechanics coupling ---
                 kinematics,
                 U,
                 U_test,
                 X,
                 X_0,

                 # --- pressure unknown & test ---
                 p_tilde,
                 p_test,

                 # --- macroscopic loading (time-varying) ---
                 grad_p_bar_ini,
                 grad_p_bar_fin,
                 pl_bar_ini,
                 pl_bar_fin,

                 # --- material / constitutive parameters ---
                 k_l0,          # 2x2 tensor (baseline intrinsic permeability, current config)
                 Phis0,
                 kappa_val,
                 use_kozeny_carman,

                 # --- measures & (optional) boundary source terms ---
                 dx,
                 dx_in=None,
                 dx_out=None,
                 Theta_in_ini=0.0,
                 Theta_in_fin=0.0,
                 Theta_out_ini=0.0,
                 Theta_out_fin=0.0):

        self.kinematics = kinematics

        # Measures (also reused later for qois/fois)
        self.dx_measure = dx
        self.dx_in = dx_in
        self.dx_out = dx_out
        self.measure = dx

        # ---- time-varying macro loads ----
        gx_ini, gy_ini = grad_p_bar_ini
        gx_fin, gy_fin = grad_p_bar_fin

        self.tv_pl_bar = dmech.TimeVaryingConstant(val_ini=pl_bar_ini, val_fin=pl_bar_fin)
        self.tv_grad_p_bar_x = dmech.TimeVaryingConstant(val_ini=gx_ini, val_fin=gx_fin)
        self.tv_grad_p_bar_y = dmech.TimeVaryingConstant(val_ini=gy_ini, val_fin=gy_fin)

        # ---- time-varying boundary sources (safe defaults) ----
        if Theta_in_ini is None:  Theta_in_ini = 0.0
        if Theta_in_fin is None:  Theta_in_fin = Theta_in_ini
        if Theta_out_ini is None: Theta_out_ini = 0.0
        if Theta_out_fin is None: Theta_out_fin = Theta_out_ini

        self.tv_Theta_in  = dmech.TimeVaryingConstant(val_ini=Theta_in_ini,  val_fin=Theta_in_fin)
        self.tv_Theta_out = dmech.TimeVaryingConstant(val_ini=Theta_out_ini, val_fin=Theta_out_fin)

        # ---- macro pressure (reference) ----
        self.grad_p_bar = dolfin.as_vector((self.tv_grad_p_bar_x.val, self.tv_grad_p_bar_y.val))
        self.pl_bar = self.tv_pl_bar.val
        self.pl_affine = dolfin.dot(self.grad_p_bar, X - X_0)

        self.pl_tot = self.pl_bar + self.pl_affine + p_tilde

        # Reference gradients
        gradX_p_tot  = self.grad_p_bar + dolfin.grad(p_tilde)
        gradX_p_test = dolfin.grad(p_test)

        # Only keep invF locally (no caching of F/J)
        invF = dolfin.inv(self.kinematics.F)

        # Current gradient (for current flux output)
        gradx_p_tot = invF.T * gradX_p_tot

        # ---- porosity law + relative permeability factor ----
        Phis = Phis0 / (1.0 + (Phis0 / kappa_val) * self.pl_tot)
        self.Phis_expr = Phis
        self.Phif_expr = 1.0 - Phis

        if use_kozeny_carman:
            k_rel = k_rel_kozeny_carman_from_Phis(Phis=Phis, Phis0=Phis0)
        else:
            k_rel = dolfin.Constant(1.0)

        # Intrinsic permeability (current) and pull-back tensor (reference)
        self.k_l_intr = k_l0 * k_rel
        self.K_l = self.kinematics.J * invF * self.k_l_intr * invF.T

        # Fluxes for output
        self.q_l = - self.k_l_intr * gradx_p_tot
        self.Q_l = self.kinematics.J * invF * self.q_l

        # ---- Darcy residual (reference configuration) ----
        self.res_form = dolfin.inner(self.K_l * gradX_p_tot, gradX_p_test) * self.measure

        # ---- coupling to solid equilibrium (pressure stress) ----
        dE_test = dolfin.derivative(self.kinematics.E, U, U_test)
        Sigma_p = - self.pl_tot * self.kinematics.J * self.kinematics.C_inv
        self.res_form += dolfin.inner(Sigma_p, dE_test) * self.measure

        self.sigma_contrib = (1.0 / self.kinematics.J) * self.kinematics.F * Sigma_p * self.kinematics.F.T

        # ---- optional inlet/outlet source terms ----
        if self.dx_in is not None:
            self.res_form -= self.tv_Theta_in.val * p_test * self.dx_in
        if self.dx_out is not None:
            self.res_form += self.tv_Theta_out.val * p_test * self.dx_out

    def set_value_at_t_step(self, t_step):
        """Update all time-varying quantities for the current load step."""
        self.tv_grad_p_bar_x.set_value_at_t_step(t_step)
        self.tv_grad_p_bar_y.set_value_at_t_step(t_step)
        self.tv_pl_bar.set_value_at_t_step(t_step)
        self.tv_Theta_in.set_value_at_t_step(t_step)
        self.tv_Theta_out.set_value_at_t_step(t_step)

def k_rel_kozeny_carman_from_Phis(Phis, Phis0, eps_val=1e-12):
    """
    Kozeny–Carman-type relative permeability factor based on solid fraction.

    k_rel(Phis) = [(1-Phis)^3 / Phis^2] / [(1-Phis0)^3 / Phis0^2]
    with a small regularization eps to avoid division by zero.
    """
    Phif  = 1.0 - Phis
    Phif0 = 1.0 - Phis0
    eps = dolfin.Constant(eps_val)

    num = (Phif  + eps)**3 / ((1.0 - Phif  + eps)**2)   # = Phif^3 / Phis^2
    den = (Phif0 + eps)**3 / ((1.0 - Phif0 + eps)**2)   # = Phif0^3 / Phis0^2
    return num / den