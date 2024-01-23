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
from .Material_Elastic import ElasticMaterial

################################################################################

class NeoHookeanElasticMaterial(ElasticMaterial):



    def __init__(self,
            kinematics,
            parameters,
            decoup=False):

        self.kinematics = kinematics

        self.C1 = self.get_C1_from_parameters(parameters)

        if (decoup):
            if   (self.kinematics.dim == 2):
                # self.Psi      =   self.C1 * (self.kinematics.J**(-2/3) * (self.kinematics.J * self.kinematics.IC_bar + 1) - 3) # MG20200206: Plane strain, written with IC_bar_2D (I prefer the expression with IC_2D)
                self.Psi      =   self.C1 * (self.kinematics.J**(-2/3) * (self.kinematics.IC + 1) - 3) # MG20200206: Plane strain, written with IC_2D
                self.Sigma    = 2*self.C1 * self.kinematics.J**(-2/3) * (self.kinematics.I - (self.kinematics.IC + 1)/3 * self.kinematics.C_inv) # MG20200206: Cannot differentiate Psi wrt to C because J is not defined as a function of C
                self.Sigma_ZZ = 2*self.C1 * self.kinematics.J**(-2/3) * (1 - (self.kinematics.IC + 1)/3)
            elif (self.kinematics.dim == 3):
                self.Psi   =   self.C1 * (self.kinematics.IC_bar - 3)
                self.Sigma = 2*self.C1 * self.kinematics.J**(-2/3) * (self.kinematics.I - self.kinematics.IC/3 * self.kinematics.C_inv) # MG20200206: Cannot differentiate Psi wrt to C because J is not defined as a function of C
        else:
            if   (self.kinematics.dim == 2):
                self.Psi      =   self.C1 * (self.kinematics.IC - 2 - 2*dolfin.ln(self.kinematics.J)) # MG20200206: Plane strain
                self.Sigma    = 2*self.C1 * (self.kinematics.I - self.kinematics.C_inv) # MG20200206: Cannot differentiate Psi wrt to C because J is not defined as a function of C
                self.Sigma_ZZ = dolfin.Constant(0.)
            elif (self.kinematics.dim == 3):
                self.Psi   =   self.C1 * (self.kinematics.IC - 3 - 2*dolfin.ln(self.kinematics.J))
                self.Sigma = 2*self.C1 * (self.kinematics.I - self.kinematics.C_inv) # MG20200206: Cannot differentiate Psi wrt to C because J is not defined as a function of C

        # self.P = dolfin.diff(self.Psi, self.kinematics.F) # MG20220426: Cannot do that for micromechanics problems
        self.P = self.kinematics.F * self.Sigma

        self.sigma = self.P * self.kinematics.F.T / self.kinematics.J



    # def get_free_energy(self,
    #         U=None,
    #         C=None):

    #     C     = self.get_C_from_U_or_C(U,C)
    #     IC    = dolfin.tr(C)
    #     JF    = dolfin.sqrt(dolfin.det(C)) # MG20200207: Watch out! This is well defined for inverted elements!
    #     # C_inv = dolfin.inv(C)

    #     assert (C.ufl_shape[0] == C.ufl_shape[1])
    #     dim = C.ufl_shape[0]
    #     # I = dolfin.Identity(dim)

    #     if   (dim == 2):
    #         Psi   =   self.C1 * (IC - 2 - 2*dolfin.ln(JF)) # MG20200206: plane strain
    #         # Sigma = 2*self.C1 * (I - C_inv)
    #     elif (dim == 3):
    #         Psi   =   self.C1 * (IC - 3 - 2*dolfin.ln(JF))
    #         # Sigma = 2*self.C1 * (I - C_inv)
    #     Sigma = 2*dolfin.diff(Psi, C)

    #     return Psi, Sigma
