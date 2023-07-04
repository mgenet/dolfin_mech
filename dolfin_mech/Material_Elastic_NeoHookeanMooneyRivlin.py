#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Material_Elastic import ElasticMaterial

################################################################################

class NeoHookeanMooneyRivlinElasticMaterial(ElasticMaterial):



    def __init__(self,
            kinematics,
            parameters,
            decoup=False):

        self.kinematics = kinematics

        C1,C2 = self.get_C1_and_C2_from_parameters(parameters) # MG20220318: This is different from computing C1 & C2 separately…
        parameters["C1"] = C1
        parameters["C2"] = C2

        self.nh = dmech.NeoHookeanElasticMaterial(kinematics, parameters, decoup)
        self.mr = dmech.MooneyRivlinElasticMaterial(kinematics, parameters, decoup)

        self.Psi   = self.nh.Psi   + self.mr.Psi
        self.Sigma = self.nh.Sigma + self.mr.Sigma
        if (self.kinematics.dim == 2):
            self.Sigma_ZZ = self.nh.Sigma_ZZ + self.mr.Sigma_ZZ
        self.P     = self.nh.P     + self.mr.P
        self.sigma = self.nh.sigma + self.mr.sigma



    # def get_free_energy(self, *args, **kwargs):

    #     Psi_nh, Sigma_nh = self.nh.get_free_energy(*args, **kwargs)
    #     Psi_mr, Sigma_mr = self.mr.get_free_energy(*args, **kwargs)

    #     Psi   = Psi_nh   + Psi_mr
    #     Sigma = Sigma_nh + Sigma_mr

    #     return Psi, Sigma
