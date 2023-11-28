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

class WskelLungElasticMaterial(ElasticMaterial):



    def __init__(self,
            kinematics,
            parameters):

        self.kinematics = kinematics

        self.bulk = dmech.ExponentialOgdenCiarletGeymonatElasticMaterial(kinematics, parameters)
        self.dev  = dmech.NeoHookeanMooneyRivlinElasticMaterial(kinematics, parameters)

        self.Psi   = self.bulk.Psi   + self.dev.Psi
        self.Sigma = self.bulk.Sigma + self.dev.Sigma
        self.P     = self.bulk.P     + self.dev.P
        self.sigma = self.bulk.sigma + self.dev.sigma
