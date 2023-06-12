#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin_mech as dmech
from .Material_Elastic import ElasticMaterial

################################################################################

class OgdenCiarletGeymonatNeoHookeanElasticMaterial(ElasticMaterial):



    def __init__(self,
            kinematics,
            parameters,
            decoup=False):

        self.kinematics = kinematics

        self.bulk = dmech.OgdenCiarletGeymonatElasticMaterial(kinematics, parameters, decoup)
        self.dev  = dmech.NeoHookeanElasticMaterial(kinematics, parameters, decoup)

        self.Psi   = self.bulk.Psi   + self.dev.Psi
        self.Sigma = self.bulk.Sigma + self.dev.Sigma
        if (self.kinematics.dim == 2):
            self.Sigma_ZZ = self.bulk.Sigma_ZZ + self.dev.Sigma_ZZ
        self.P     = self.bulk.P     + self.dev.P
        self.sigma = self.bulk.sigma + self.dev.sigma
        self.sigma_old = self.bulk.sigma_old + self.dev.sigma_old

        self.Sigma_zz = self.bulk.Sigma_zz + self.dev.Sigma_zz

        # self.p_hydro = -(sum((self.Sigma*self.kinematics.C)[i, i] for i in range(self.kinematics.dim))+ self.Sigma_zz)/3/self.kinematics.J
        self.p_hydro = -(dolfin.tr(self.Sigma.T*self.kinematics.C)+ self.Sigma_zz)/3/self.kinematics.J

        # self.Sigma_dev = self.Sigma - self.p_hydro * self.kinematics.I
        self.Sigma_dev = self.Sigma + self.p_hydro * self.kinematics.J * self.kinematics.C_inv

        # self.Sigma_VM = dolfin.sqrt(1.5 *  (sum((self.Sigma_dev.T*self.Sigma_dev)[i, i] for i in range(self.kinematics.dim))))
        self.Sigma_VM = dolfin.sqrt(1.5 *dolfin.tr(self.Sigma_dev.T*self.Sigma_dev))



    # def get_free_energy(self, *args, **kwargs):

    #     Psi_bulk, Sigma_bulk = self.bulk.get_free_energy(*args, **kwargs)
    #     Psi_dev , Sigma_dev  = self.dev.get_free_energy(*args, **kwargs)

    #     Psi   = Psi_bulk   + Psi_dev
    #     Sigma = Sigma_bulk + Sigma_dev

    #     return Psi, Sigma



    # def get_PK2_stress(self, *args, **kwargs):

    #     Sigma_bulk = self.bulk.get_PK2_stress(*args, **kwargs)
    #     Sigma_dev  = self.dev.get_PK2_stress(*args, **kwargs)

    #     Sigma = Sigma_bulk + Sigma_dev

    #     return Sigma



    # def get_PK1_stress(self, *args, **kwargs):

    #     P_bulk = self.bulk.get_PK1_stress(*args, **kwargs)
    #     P_dev  = self.dev.get_PK1_stress(*args, **kwargs)

    #     P = P_bulk + P_dev

    #     return P