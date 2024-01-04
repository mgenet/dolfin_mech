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

class SurfaceTensionLoadingOperator(Operator):

    def __init__(self,
            # U,
            # U_test,
            kinematics,
            N,
            S_area,
            measure,
            U_tot_test,
            dS,
            tension_params=None,
            gamma_val=None, gamma_ini=None, gamma_fin=None):

        self.measure = measure

        self.tv_gamma = dmech.TimeVaryingConstant(
            val=gamma_val, val_ini=gamma_ini, val_fin=gamma_fin)
        gamma = self.tv_gamma.val

        # FmTN = dolfin.dot(dolfin.inv(kinematics.F).T, N)
        # T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
        # Pi = gamma * T * kinematics.J * self.measure
        # self.res_form = dolfin.derivative(Pi, U, U_test)


        self.dS = dS
        self.S0 = dolfin.assemble(dolfin.Constant(1)*dS(0))
        self.N = N
        self.kinematics=kinematics

        self.tv_gamma = dmech.TimeVaryingConstant(
            val=gamma_val, val_ini=gamma_ini, val_fin=gamma_fin)
        gamma = self.tv_gamma.val

        self.S_t = dmech.TimeVaryingConstant(0.)
        S = self.S_t.val
        
        S_hat = S_area/self.S0

        state = tension_params.get("loading_direction", None)
        d1 = tension_params.get("d1", 0)
        d2 = tension_params.get("d2", 0)
        d3 = tension_params.get("d3", 0)
        
        if state=="inf":
            gamma = gamma * (1 - d1/(1 + (S_hat/d2)**d3))
        elif state=="def":
            gamma = gamma * (d1 - d2*dolfin.exp(1 - dolfin.exp(d3*S_hat)))


        dim = U_tot_test.ufl_shape[0]
        I = dolfin.Identity(dim)
        FmTN = dolfin.dot(dolfin.inv(kinematics.F).T, N)
        T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
        n = FmTN/T
        P = I - dolfin.outer(n,n)
        taus = gamma * P

        self.res_form =  dolfin.inner(taus, dolfin.dot(P,(dolfin.grad(U_tot_test)))) *  self.kinematics.J * T * self.measure



    def set_value_at_t_step(self,
            t_step):

        self.tv_gamma.set_value_at_t_step(t_step)


    def set_dt(self,
            t_step):
        FmTN = dolfin.dot(dolfin.inv(self.kinematics.F).T, self.N)
        T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
        S_val = dolfin.assemble(T*self.kinematics.J * self.dS(0))
        self.S_t.set_value(S_val)

################################################################################

class SurfaceTension0LoadingOperator(Operator):

    def __init__(self,
            u,
            u_test,
            kinematics,
            N,
            measure,
            gamma_val=None, gamma_ini=None, gamma_fin=None):

        self.measure = measure

        self.tv_gamma = dmech.TimeVaryingConstant(
            val=gamma_val, val_ini=gamma_ini, val_fin=gamma_fin)
        gamma = self.tv_gamma.val

        dim = u.ufl_shape[0]
        I = dolfin.Identity(dim)
        Pi = gamma * (1 + dolfin.inner(
            kinematics.epsilon,
            I - dolfin.outer(N,N))) * self.measure
        self.res_form = dolfin.derivative(Pi, u, u_test) # MG20211220: Is that correct?!



    def set_value_at_t_step(self,
            t_step):

        self.tv_gamma.set_value_at_t_step(t_step)
