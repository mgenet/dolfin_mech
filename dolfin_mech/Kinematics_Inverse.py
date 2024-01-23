#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_mech as dmech

################################################################################

class InverseKinematics():



    def __init__(self,
            u,
            u_old=None):

        self.u = u

        self.dim = self.u.ufl_shape[0]
        self.I = dolfin.Identity(self.dim)

        self.f     = self.I + dolfin.grad(self.u)
        self.F     = dolfin.inv(self.f)
        self.F     = dolfin.variable(self.F)
        self.J     = dolfin.det(self.F)
        self.C     = self.F.T * self.F
        self.C     = dolfin.variable(self.C)
        self.C_inv = dolfin.inv(self.C)
        self.IC    = dolfin.tr(self.C)
        self.IIC   = (dolfin.tr(self.C)*dolfin.tr(self.C) - dolfin.tr(self.C*self.C))/2
        self.E     = (self.C - self.I)/2
        self.E     = dolfin.variable(self.E)

        self.F_bar   = self.J**(-1/self.dim) * self.F
        self.C_bar   = self.F_bar.T * self.F_bar
        self.IC_bar  = dolfin.tr(self.C_bar)
        self.IIC_bar = (dolfin.tr(self.C_bar)*dolfin.tr(self.C_bar) - dolfin.tr(self.C_bar*self.C_bar))/2
        self.E_bar   = (self.C_bar - self.I)/2

        if (u_old is not None):
            self.u_old = u_old

            self.f_old = self.I + dolfin.grad(self.u_old)
            self.F_old = dolfin.inv(self.f_old)
            self.J_old = dolfin.det(self.F_old)
            self.C_old = self.F_old.T * self.F_old
            self.E_old = (self.C_old - self.I)/2

            self.F_bar_old = self.J_old**(-1/self.dim) * self.F_old
            self.C_bar_old = self.F_bar_old.T * self.F_bar_old
            self.E_bar_old = (self.C_bar_old - self.I)/2
