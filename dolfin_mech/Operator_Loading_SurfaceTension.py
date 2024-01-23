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
            kinematics,
            N,
            measure,
            U_test,
            tension_params={},
            gamma_val=None, gamma_ini=None, gamma_fin=None):

        self.measure = measure

        self.tv_gamma = dmech.TimeVaryingConstant(
            val=gamma_val, val_ini=gamma_ini, val_fin=gamma_fin)
        gamma = self.tv_gamma.val

        self.N = N
        self.kinematics=kinematics

        self.tv_gamma = dmech.TimeVaryingConstant(
            val=gamma_val, val_ini=gamma_ini, val_fin=gamma_fin)
        gamma = self.tv_gamma.val
        
        dim = U_test.ufl_shape[0]
        I = dolfin.Identity(dim)
        FmTN = dolfin.dot(dolfin.inv(kinematics.F).T, N)
        T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
        n = FmTN/T
        P = I - dolfin.outer(n,n)
        
        S_hat = kinematics.J * T

        surface_dependancy = tension_params.get("surface_dependancy", None)
        if surface_dependancy == 1:
            d1 = dolfin.Constant(tension_params.get("d1"))
            d2 = dolfin.Constant(tension_params.get("d2"))
            d3 = dolfin.Constant(tension_params.get("d3"))

            gamma = gamma * (d1/(1 + (S_hat/d2)**(d3)))


        taus = gamma * P

        self.res_form =  dolfin.inner(taus, dolfin.dot(P,(dolfin.grad(U_test)))) *  self.kinematics.J * T * self.measure



    def set_value_at_t_step(self,
            t_step):

        self.tv_gamma.set_value_at_t_step(t_step)



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
