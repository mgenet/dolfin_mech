#coding=utf8

################################################################################
###                                                                          ###
### Created by Alice Peyraut, 2023-2024                                      ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2024                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class PressureBalancingGravityLoadingOperator(Operator):

    def __init__(self,
            X,
            x0,
            x0_test,
            lbda,
            lbda_test,
            mu,
            mu_test,
            p,
            p_test,
            gamma,
            gamma_test,
            U,
            U_test,
            Phis0,
            rho_solid,
            kinematics,
            N,
            dS,
            dV,
            breathing_constant,
            P0_val=None, P0_ini=None, P0_fin=None,
            f_val=None, f_ini=None, f_fin=None):

        self.measure = dV
        self.dS = dS

        self.V0 = dolfin.assemble(dolfin.Constant(1)*self.measure)
        self.rho0 = dolfin.assemble(Phis0 * self.measure)

        self.tv_f = dmech.TimeVaryingConstant(
            val=f_val, val_ini=f_ini, val_fin=f_fin)
        f = self.tv_f.val

        self.tv_P0 = dmech.TimeVaryingConstant(
            val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
        P0 = self.tv_P0.val

        self.tv_grad = dmech.TimeVaryingConstant(
            val=None, val_ini=f_ini[2], val_fin=f_fin[2]) #### AP2023 - depends on the orientation of gravity: here, g = [0., 0., 9.81e3]
        grad = self.tv_grad.val

        nf = dolfin.dot(N, dolfin.inv(kinematics.F))
        nf_norm = dolfin.sqrt(dolfin.inner(nf,nf))
        n = nf/nf_norm

        x = X + U
        x_tilde = x-x0

        P_tilde = P0 + dolfin.Constant(rho_solid) * grad * ( x[2]- x0[2]) - dolfin.Constant(breathing_constant)*dolfin.Constant(rho_solid) * grad * ( x[1]- x0[1]) * ( x[1]- x0[1]) 

        grads_p = dolfin.dot(dolfin.grad(p-P_tilde), dolfin.inv(kinematics.F)) - n*(dolfin.dot(n,dolfin.dot(dolfin.grad(p-P_tilde), dolfin.inv(kinematics.F))))
        grads_p_test = dolfin.dot(dolfin.grad(p_test), dolfin.inv(kinematics.F)) - n*(dolfin.dot(n,dolfin.dot(dolfin.grad(p_test), dolfin.inv(kinematics.F))))

        self.res_form = dolfin.Constant(1e-8)*p*p_test * kinematics.J * self.measure
        self.res_form -= dolfin.inner(rho_solid*Phis0*f, U_test) * self.measure
        self.res_form -=  dolfin.inner(-p * n, U_test) * nf_norm * kinematics.J * dS
        self.res_form += dolfin.inner(rho_solid*Phis0*f, lbda_test) * self.measure
        self.res_form += dolfin.inner(-p * n, lbda_test) * nf_norm * kinematics.J * dS
        self.res_form += - dolfin.dot(lbda, n) * p_test * nf_norm * kinematics.J * self.dS
        self.res_form += - dolfin.dot(mu, dolfin.cross(x_tilde, n)) *  p_test * nf_norm * kinematics.J * self.dS
        self.res_form += gamma  *  p_test * nf_norm * kinematics.J * self.dS
        self.res_form +=  dolfin.inner(grads_p, grads_p_test) * nf_norm * kinematics.J * self.dS
        self.res_form += dolfin.inner(dolfin.cross(x_tilde, -p * n), mu_test) * nf_norm * kinematics.J * dS
        self.res_form += (p-P_tilde)*gamma_test * nf_norm * kinematics.J * dS
        self.res_form -= dolfin.inner((Phis0 * x / dolfin.Constant(self.rho0) - x0/self.V0), x0_test) * self.measure


    def set_value_at_t_step(self,
            t_step):

        self.tv_f.set_value_at_t_step(t_step)
        self.tv_P0.set_value_at_t_step(t_step)
        self.tv_grad.set_value_at_t_step(t_step)

################################################################################

class PressureBalancingGravity0LoadingOperator(Operator):

    def __init__(self,
            x,
            x0,
            u_test,
            lbda,
            lbda_test,
            mu,
            mu_test,
            p,
            p_test,
            gamma,
            gamma_test,
            rho_solid,
            phis,
            n,
            dS,
            dV,
            breathing_constant,
            P0_val=None, P0_ini=None, P0_fin=None,
            f_val=None, f_ini=None, f_fin=None):

        self.dS = dS
        self.measure = dV

        self.tv_f = dmech.TimeVaryingConstant(
            val=f_val, val_ini=f_ini, val_fin=f_fin)
        f = self.tv_f.val

        self.tv_P0 = dmech.TimeVaryingConstant(
            val=P0_val, val_ini=P0_ini, val_fin=P0_fin)
        P0 = self.tv_P0.val

        self.tv_grad = dmech.TimeVaryingConstant(
            val=None, val_ini=0, val_fin=f_fin[2]) #### AP2023 - depends on the orientation of gravity: here, g = [0., 0., 9.81e3]
        grad = self.tv_grad.val

        x_tilde = x-dolfin.Constant(x0)

        P_tilde = P0 + dolfin.Constant(rho_solid) * grad * ( x[2]- dolfin.Constant(x0[2])) - dolfin.Constant(breathing_constant)*dolfin.Constant(rho_solid) * grad* ( x[1]- dolfin.Constant(x0[1])) * ( x[1]- dolfin.Constant(x0[1])) 
        
        grads_p = dolfin.grad(p-P_tilde) - n*(dolfin.dot(n,dolfin.grad(p-P_tilde)))
        grads_p_test = dolfin.grad(p_test) - n*(dolfin.dot(n,dolfin.grad(p_test)))

        self.res_form = dolfin.Constant(1e-8)*p*p_test * self.measure
        self.res_form -= dolfin.inner(rho_solid*phis*f, u_test) * self.measure
        self.res_form -=  dolfin.inner(-p * n, u_test) * dS
        self.res_form += dolfin.inner(rho_solid*phis*f, lbda_test) * self.measure
        self.res_form += dolfin.inner(-p * n, lbda_test) * dS
        self.res_form += - dolfin.dot(lbda, n) *  p_test * self.dS
        self.res_form += - dolfin.dot(mu, dolfin.cross(x_tilde, n)) *  p_test * self.dS
        self.res_form += gamma  *  p_test * self.dS
        self.res_form +=  dolfin.inner(grads_p, grads_p_test) * self.dS
        self.res_form += dolfin.inner(dolfin.cross(x_tilde, -p * n), mu_test) * dS
        self.res_form += (p-P_tilde)*gamma_test * dS

    def set_value_at_t_step(self,
            t_step):
        self.tv_f.set_value_at_t_step(t_step)
        self.tv_P0.set_value_at_t_step(t_step)
        self.tv_grad.set_value_at_t_step(t_step)
