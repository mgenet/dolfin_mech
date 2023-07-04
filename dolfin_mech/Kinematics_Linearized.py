#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_mech as dmech

################################################################################

class LinearizedKinematics():



    def __init__(self,
            u,
            u_old=None):

        self.u = u

        self.dim = self.u.ufl_shape[0]
        self.I = dolfin.Identity(self.dim)

        self.epsilon = dolfin.sym(dolfin.grad(self.u))
        self.epsilon = dolfin.variable(self.epsilon)

        self.epsilon_sph = dolfin.tr(self.epsilon)/self.dim * self.I
        self.epsilon_dev = self.epsilon - self.epsilon_sph

        if (u_old is not None):
            self.u_old = u_old
            
            self.epsilon_old = dolfin.sym(dolfin.grad(self.u_old))

            self.epsilon_sph_old = dolfin.tr(self.epsilon_old)/self.dim * self.I
            self.epsilon_dev_old = self.epsilon_old - self.epsilon_sph_old

            self.epsilon_mid = (self.epsilon_old + self.epsilon)/2
