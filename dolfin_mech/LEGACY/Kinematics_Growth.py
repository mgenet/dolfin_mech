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

class GrowthKinematics():



    def __init__(self,
            dim,
            U,
            U_old,
            Q_expr=None,
            w_growth=False,
            Fg=None,
            Fg_old=None,
            w_relaxation=False,
            Fr=None,
            Fr_old=None):

        self.I = dolfin.Identity(dim)

        self.Ft     = self.I + dolfin.grad(U)
        self.Ft_old = self.I + dolfin.grad(U_old)

        self.Jt     = dolfin.det(self.Ft    )
        self.Jt_old = dolfin.det(self.Ft_old)

        self.Ct     = self.Ft.T     * self.Ft
        self.Ct_old = self.Ft_old.T * self.Ft_old

        self.Et     = (self.Ct     - self.I)/2
        self.Et_old = (self.Ct_old - self.I)/2

        self.Fe     = self.Ft
        self.Fe_old = self.Ft_old
        if (w_growth):
            self.Fe     = dolfin.dot(self.Fe    , dolfin.inv(Fg    ))
            self.Fe_old = dolfin.dot(self.Fe_old, dolfin.inv(Fg_old))
        if (w_relaxation):
            self.Fe     = dolfin.dot(self.Fe    , dolfin.inv(Fr    ))
            self.Fe_old = dolfin.dot(self.Fe_old, dolfin.inv(Fr_old))

        self.Je     = dolfin.det(self.Fe    )
        self.Je_old = dolfin.det(self.Fe_old)

        self.Ce     = self.Fe.T     * self.Fe
        self.Ce_old = self.Fe_old.T * self.Fe_old
        self.Ce_inv = dolfin.inv(self.Ce)
        self.ICe    = dolfin.tr(self.Ce)

        self.Ee     = (self.Ce     - self.I)/2
        self.Ee_old = (self.Ce_old - self.I)/2

        if (Q_expr is not None):
            self.Ee_loc = dolfin.dot(dolfin.dot(Q_expr, self.Ee), Q_expr.T)

        self.Fe_mid = (self.Fe_old + self.Fe)/2
        self.Ce_mid = (self.Ce_old + self.Ce)/2
        self.Ee_mid = (self.Ee_old + self.Ee)/2

        self.Fe_bar  = self.Je**(-1/self.dim) * self.Fe
        self.Ce_bar  = self.Fe_bar.T * self.Fe_bar
        self.ICe_bar = dolfin.tr(self.Ce_bar)
