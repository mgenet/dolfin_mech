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

class Kinematics():



    def __init__(self,
            U,
            U_old=None,
            Q_expr=None):

        self.U = U

        self.dim = self.U.ufl_shape[0]
        self.I = dolfin.Identity(self.dim)

        self.F     = self.I + dolfin.grad(self.U)
        self.F     = dolfin.variable(self.F)
        self.J     = dolfin.det(self.F)
        self.C     = self.F.T * self.F
        self.C     = dolfin.variable(self.C)
        self.C_inv = dolfin.inv(self.C)
        self.IC    = dolfin.tr(self.C)
        self.IIC   = (dolfin.tr(self.C)*dolfin.tr(self.C) - dolfin.tr(self.C*self.C))/2
        self.E     = (self.C - self.I)/2
        self.E     = dolfin.variable(self.E)

        self.E_sph = dolfin.tr(self.E)/self.dim * self.I
        self.E_dev = self.E - self.E_sph

        self.F_bar     = self.J**(-1/self.dim) * self.F
        self.C_bar     = self.F_bar.T * self.F_bar
        self.C_bar_inv = dolfin.inv(self.C_bar)
        self.IC_bar    = dolfin.tr(self.C_bar)
        self.IIC_bar   = (dolfin.tr(self.C_bar)*dolfin.tr(self.C_bar) - dolfin.tr(self.C_bar*self.C_bar))/2
        self.E_bar     = (self.C_bar - self.I)/2

        if (U_old is not None):
            self.U_old = U_old

            self.F_old = self.I + dolfin.grad(U_old)
            self.J_old = dolfin.det(self.F_old)
            self.C_old = self.F_old.T * self.F_old
            self.E_old = (self.C_old - self.I)/2

            self.F_bar_old = self.J_old**(-1/self.dim) * self.F_old
            self.C_bar_old = self.F_bar_old.T * self.F_bar_old
            self.E_bar_old = (self.C_bar_old - self.I)/2

            self.F_mid = (self.F_old + self.F)/2
            self.J_mid = (self.J_old + self.J)/2
            self.C_mid = (self.C_old + self.C)/2
            self.E_mid = (self.E_old + self.E)/2

            self.F_bar_mid = (self.F_bar_old + self.F_bar)/2
            self.C_bar_mid = (self.C_bar_old + self.C_bar)/2
            self.E_bar_mid = (self.E_bar_old + self.E_bar)/2

        if (Q_expr is not None):
            self.Q_expr = Q_expr

            self.E_loc     = self.Q_expr * self.E     * self.Q_expr.T # MG20211215: This should work, right?
            self.E_bar_loc = self.Q_expr * self.E_bar * self.Q_expr.T # MG20211215: This should work, right?
            # self.E_loc     = dolfin.dot(dolfin.dot(self.Q_expr, self.E    ), self.Q_expr.T)
            # self.E_bar_loc = dolfin.dot(dolfin.dot(self.Q_expr, self.E_bar), self.Q_expr.T)

            if (U_old is not None):
                self.E_old_loc     = self.Q_expr * self.E_old     * self.Q_expr.T # MG20211215: This should work, right?
                self.E_bar_old_loc = self.Q_expr * self.E_bar_old * self.Q_expr.T # MG20211215: This should work, right?
                # self.E_old_loc     = dolfin.dot(dolfin.dot(self.Q_expr, self.E_old    ), self.Q_expr.T)
                # self.E_bar_old_loc = dolfin.dot(dolfin.dot(self.Q_expr, self.E_bar_old), self.Q_expr.T)
