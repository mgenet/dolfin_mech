#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Material import Material

################################################################################


class Material_Perm_KClaw:
 
    def __init__(self, C=dolfin.Constant(1.0), dim=3):
       
        self.C = C
        self.dim = dim

    # ---------------------------------------------------------------------- #
    def K_ref(self, J, Phi_s):

        I = dolfin.Identity(self.dim)
        return self.C * ((J - Phi_s)**3) / (J * Phi_s**2) * I