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

class SphericalKinematics():



    def __init__(self,
            R,
            Rho,
            Rho_old,
            dim=3):

        self.I = dolfin.Identity(dim)

        # self.U = dolfin.as_vector([R*Rho, 0., 0.])

        self.Ft = dolfin.as_matrix([
            [1+Rho+R*Rho.dx(0),  0   ,  0   ],
            [     0           , 1+Rho,  0   ],
            [     0           ,  0   , 1+Rho]])
        self.Ft_old = dolfin.as_matrix([
            [1+Rho_old+R*Rho_old.dx(0),  0       ,  0       ],
            [         0               , 1+Rho_old,  0       ],
            [         0               ,  0       , 1+Rho_old]])

        self.Jt     = dolfin.det(self.Ft    )
        self.Jt_old = dolfin.det(self.Ft_old)

        self.Ct     = self.Ft.T     * self.Ft
        self.Ct_old = self.Ft_old.T * self.Ft_old

        self.Et     = (self.Ct     - self.I)/2
        self.Et_old = (self.Ct_old - self.I)/2

        self.Fe     = self.Ft
        self.Fe_old = self.Ft_old

        self.Je     = dolfin.det(self.Fe    )
        self.Je_old = dolfin.det(self.Fe_old)

        self.Ce     = self.Fe.T     * self.Fe
        self.Ce_old = self.Fe_old.T * self.Fe_old
        self.Ce_inv = dolfin.inv(self.Ce)
        self.ICe    = dolfin.tr(self.Ce)

        self.Ee     = (self.Ce     - self.I)/2
        self.Ee_old = (self.Ce_old - self.I)/2

        self.Fe_mid = (self.Fe_old + self.Fe)/2
        self.Ce_mid = (self.Ce_old + self.Ce)/2
        self.Ee_mid = (self.Ee_old + self.Ee)/2
