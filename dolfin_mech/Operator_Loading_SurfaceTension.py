#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
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
            U_hat,
            U_hat_test,
            U_bar,
            U_bar_test,
            U_tot,
            U_tot_test,
            sol,
            sol_test,
            X,
            X_0,
            kinematics,
            N,
            measure,
            dS,
            gamma_val=None, gamma_ini=None, gamma_fin=None):

        self.measure = measure
        self.dS = dS
        self.S0 = dolfin.assemble(dolfin.Constant(1)*dS(0))
        self.N = N

        self.tv_gamma = dmech.TimeVaryingConstant(
            val=gamma_val, val_ini=gamma_ini, val_fin=gamma_fin)
        gamma = self.tv_gamma.val

        self.S_t = dmech.TimeVaryingConstant(0.)
        S = self.S_t.val

        # self.tv_gamma.surface_change_rate(kinematics, dt)
        # print("gamma =" +str(gamma))

        
        # gamma_S = gamma * (1 - 1.6*dolfin.exp(-0.5*S/self.S0))
        FmTN = dolfin.dot(dolfin.inv(kinematics.F).T, N)
        T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
        Pi = gamma * T * kinematics.J * self.measure
        # Pi = gamma * (T * kinematics.J - dolfin.Constant(1))* self.measure
        # self.res_form = dolfin.derivative(Pi, U, U_test)*dolfin.derivative(U_tot, U)% + dolfin.derivative(Pi, U_bar, U_bar_test)*dolfin.derivative(U_tot, U_bar)
        # self.res_form = dolfin.inner(dolfin.derivative(Pi, U, U_test), (X - X_0)) + dolfin.derivative(Pi, U_bar, U_bar_test)
        # self.res_form = dolfin.derivative(Pi, U_bar) * dolfin.derivative(U_tot, U_bar, U_bar_test) + dolfin.derivative(Pi, U_hat) * dolfin.derivative(U_tot, U_hat, U_hat_test)
        # self.res_form =  dolfin.derivative(Pi, U_hat, U_hat_test) +  dolfin.derivative(Pi, U_bar[0, 0], U_bar_test[0, 0]) +  dolfin.derivative(Pi, U_bar[1, 1], U_bar_test[1, 1]) +  dolfin.derivative(Pi, U_bar[1, 0], U_bar_test[1, 0])+  dolfin.derivative(Pi, U_bar[0, 1], U_bar_test[0, 1])
        # self.res_form = dolfin.derivative(Pi, U_tot)* (dolfin.derivative(U_tot, U_bar, U_bar_test) + dolfin.derivative(U_tot, U_hat, U_hat_test))
        # self.res_form = dolfin.derivative(Pi, U_hat, U_hat_test) #+ dolfin.derivative(Pi, U_bar, U_bar_test)
        # self.res_form = dolfin.derivative(Pi, U_hat, U_hat_test) 
        self.res_form = dolfin.derivative(Pi, sol, sol_test)

        self.kinematics=kinematics

    # def surface_change_rate(self,
    #         dt):

    #     self.tv_gamma.surface_change_rate(self.kinematics, dt)

    def set_value_at_t_step(self,
            t_step):

        self.tv_gamma.set_value_at_t_step(t_step)
        # print("t_step =" +str(t_step))
        # print("value at t_step = " +str(self.tv_gamma.set_value_at_t_step(t_step)))

    def returne_surface_rate(self):
        self.tv_gamma.surface_change_rate()


    def set_dt(self,
            t_step,
            dt):
        S = dolfin.assemble(dolfin.sqrt(dolfin.dot(dolfin.inv(self.kinematics.F.T), self.N)[0]**2+ dolfin.dot(dolfin.inv(self.kinematics.F.T), self.N)[1]**2)*self.kinematics.J * self.dS(0))
        # print("gamma_hat:" +str((1 - 2.5*dolfin.exp(-1*S/self.S0))))
        self.S_t.set_value(S)

################################################################################

class SurfaceTension0LoadingOperator(Operator):

    def __init__(self,
            U_tot,
            u,
            u_test,
            sol,
            sol_test,
            kinematics,
            N,
            measure,
            dS,
            gamma_val=None, gamma_ini=None, gamma_fin=None):

        self.measure = measure
        self.dS = dS
        self.S0 = dolfin.assemble(dolfin.Constant(1)*dS(0))
        self.N = N
        self.kinematics=kinematics


        self.tv_gamma = dmech.TimeVaryingConstant(
            val=gamma_val, val_ini=gamma_ini, val_fin=gamma_fin)
        gamma = self.tv_gamma.val

        self.S_t = dmech.TimeVaryingConstant(0.)
        S = self.S_t.val

        gamma_S = gamma * (1 - 1.6*dolfin.exp(-0.5*S/self.S0))

        dim = u.ufl_shape[0]
        I = dolfin.Identity(dim)
        Pi = gamma * (1 + dolfin.inner(
            self.kinematics.E,
            I - dolfin.outer(N,N))) * self.measure
        # self.res_form = dolfin.derivative(Pi, u, u_test) # MG20211220: Is that correct?!
        self.res_form = dolfin.derivative(Pi, sol, sol_test) # MG20211220: Is that correct?!



    def set_value_at_t_step(self,
            t_step):

        self.tv_gamma.set_value_at_t_step(t_step)

    def set_dt(self,
            t_step,
            dt):
        # S = dolfin.assemble(dolfin.sqrt(dolfin.dot(dolfin.inv(self.kinematics.F.T), self.N)[0]**2+ dolfin.dot(dolfin.inv(self.kinematics.F.T), self.N)[1]**2)*self.kinematics.J * self.dS(0))
        FmTN = dolfin.dot(dolfin.inv(self.kinematics.F).T, self.N)
        T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
        S_val = dolfin.assemble(T*self.kinematics.J * self.dS(0))
        # print("gamma_hat:" +str((1 - 2.5*dolfin.exp(-1*S/self.S0))))
        self.S_t.set_value(S_val)


################################################################################
class SurfaceTensionLoadingOperatorNew(Operator):

    def __init__(self,
            U_hat,
            U_hat_test,
            U_bar,
            U_bar_test,
            U_tot,
            U_tot_test,
            S_area,
            X,
            X_0,
            sol,
            sol_test,
            kinematics,
            N,
            measure,
            dS,
            gamma_val=None, gamma_ini=None, gamma_fin=None):

        self.measure = measure
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

        # gamma_S = gamma * (1 - 10*dolfin.exp(-2.35*S_area/self.S0))
        # gamma_S = gamma * (1 - 1.6*dolfin.exp(-0.5*S/self.S0))
        # gamma_S = gamma * (1 - 2.7*dolfin.exp(-1*S_area/self.S0))
        gamma_S_inf = gamma * (0.5*S_area/self.S0 - 0.5)
        gamma_S_inf = gamma * (1.012369 + (-0.03803987 - 1.012369)/(1 + (S_hat/1.723308)**5.623721))
        
        # S_hat = S/self.S0
        # a = 0.0523
        # b = 13.24
        # c = 2.31
        # d = 1.039
        # gamma_S = gamma*(d + (a - d)/(1 + (S_hat/c)**b))

        # y0 = -0.01111168
        # v0 = -0.005012312
        # K = -2.365393
        # gamma_S = gamma*(y0 - v0/K*(1 - dolfin.exp(-K*S_hat)))

        a = -1.42964750e-02
        b = 2.19311234e-03
        c = 2.35165948e+00
        gamma_S_def = gamma*(a + b*dolfin.exp(c*S_hat))

        # a = 11058820000
        # b = 21.839
        # c = 2.8277
        # gamma_S = gamma* a * dolfin.exp(- ((S_hat - b)**2)/2*c**2)

        # gamma_S = gamma*(0.3*S_hat - 0.25)
        # gamma_S = gamma*(0.5*S_hat - 0.5)

        dim = U_hat.ufl_shape[0]
        I = dolfin.Identity(dim)
        FmTN = dolfin.dot(dolfin.inv(kinematics.F).T, N)
        T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
        n = FmTN/T
        P = I - dolfin.outer(n,n)
        taus = gamma_S_def * P
        fs = dolfin.dot(P, dolfin.div(taus))
        S0 = dolfin.assemble(dolfin.Constant(1)*self.measure)

        # self.res_form =  dolfin.inner(taus, dolfin.dot(P,(dolfin.grad(U_hat_test)))) *  self.kinematics.J * T * self.measure
        self.res_form =  dolfin.inner(taus, dolfin.dot(P,(dolfin.grad(U_tot_test)))) *  self.kinematics.J * T * self.measure
        # self.res_form = dolfin.inner(taus, dolfin.grad(U_tot_test)) *  self.kinematics.J * T * self.measure
        # self.res_form = dolfin.inner(taus, dolfin.grad(U_hat_test)) *  self.kinematics.J * T * self.measure
        # self.res_form = - dolfin.dot(fs, U_tot_test) *  self.kinematics.J * T  * self.measure
        # self.res_form = - dolfin.inner(fs, U_hat_test) *  self.kinematics.J * T * self.measure



    def set_value_at_t_step(self,
            t_step):

        self.tv_gamma.set_value_at_t_step(t_step)


    def returne_surface_rate(self):
        self.tv_gamma.surface_change_rate()


    def set_dt(self,
            t_step,
            dt):
        # S_val = dolfin.assemble(dolfin.sqrt(dolfin.dot(dolfin.inv(self.kinematics.F.T), self.N)[0]**2+ dolfin.dot(dolfin.inv(self.kinematics.F.T), self.N)[1]**2)*self.kinematics.J * self.dS(0))
        FmTN = dolfin.dot(dolfin.inv(self.kinematics.F).T, self.N)
        T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
        S_val = dolfin.assemble(T*self.kinematics.J * self.dS(0))
        S_hat = S_val/self.S0
        # a = 0.0523
        # b = 13.24
        # c = 2.31
        # d = 1.039
        # gamma_hat = (d + (a - d)/(1 + (S_hat/c)**b))
        # gamma_hat = (1 - 10*dolfin.exp(-2.35*S_hat))
        # print("gamma_hat:" +str(gamma_hat))
        self.S_t.set_value(S_val)


