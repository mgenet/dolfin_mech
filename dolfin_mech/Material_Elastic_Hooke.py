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

class HookeElasticMaterial(ElasticMaterial):



    def __init__(self,
            kinematics,
            parameters):

        self.kinematics = kinematics

        self.lmbda = self.get_lambda_from_parameters(parameters)
        self.mu    = self.get_mu_from_parameters(parameters)

        self.psi   = (self.lmbda/2) * dolfin.tr(self.kinematics.epsilon)**2 + self.mu * dolfin.inner(self.kinematics.epsilon, self.kinematics.epsilon)

        self.sigma = dolfin.diff(self.psi, self.kinematics.epsilon)
        # self.sigma = self.lmbda * dolfin.tr(self.kinematics.epsilon) * self.kinematics.I + 2 * self.mu * self.kinematics.epsilon

        self.Sigma = self.sigma
        self.P     = self.sigma

        if (self.kinematics.dim == 2):
            self.sigma_ZZ = self.lmbda * dolfin.tr(self.kinematics.epsilon)



    # def get_free_energy(self,
    #         U=None,
    #         epsilon=None):

    #     epsilon = self.get_epsilon_from_U_or_epsilon(
    #         U, epsilon)

    #     psi = (self.lmbda/2) * dolfin.tr(epsilon)**2 + self.mu * dolfin.inner(epsilon, epsilon)
    #     sigma = dolfin.diff(psi, epsilon)

    #     # assert (epsilon.ufl_shape[0] == epsilon.ufl_shape[1])
    #     # dim = epsilon.ufl_shape[0]
    #     # I = dolfin.Identity(dim)
    #     # sigma = self.lmbda * dolfin.tr(epsilon) * I + 2 * self.mu * epsilon

    #     return psi, sigma

################################################################################

class HookeBulkElasticMaterial(ElasticMaterial):



    def __init__(self,
            kinematics,
            parameters):

        self.kinematics = kinematics

        # self.K = self.get_K_from_parameters(parameters)
        self.lmbda, self.mu = self.get_lambda_and_mu_from_parameters(parameters)
        self.K = (self.kinematics.dim*self.lmbda + 2*self.mu)/self.kinematics.dim

        self.psi   = (self.kinematics.dim*self.K/2) * dolfin.tr(self.kinematics.epsilon_sph)**2
        self.sigma =  self.kinematics.dim*self.K    *           self.kinematics.epsilon_sph

        self.Sigma = self.sigma
        self.P     = self.sigma

        if (self.kinematics.dim == 2):
            self.sigma_ZZ = self.K * dolfin.tr(self.kinematics.epsilon)



    # def get_free_energy(self,
    #         U=None,
    #         epsilon=None,
    #         epsilon_sph=None):

    #     epsilon_sph = self.get_epsilon_sph_from_U_epsilon_or_epsilon_sph(
    #         U, epsilon, epsilon_sph)
    #     assert (epsilon_sph.ufl_shape[0] == epsilon_sph.ufl_shape[1])
    #     dim = epsilon_sph.ufl_shape[0]

    #     psi   = (dim*self.K/2) * dolfin.tr(epsilon_sph)**2
    #     sigma =  dim*self.K    *           epsilon_sph

    #     return psi, sigma



    # def get_Cauchy_stress(self,
    #         U=None,
    #         epsilon=None,
    #         epsilon_sph=None):

    #     epsilon_sph = self.get_epsilon_sph_from_U_epsilon_or_epsilon_sph(
    #         U, epsilon, epsilon_sph)
    #     assert (epsilon_sph.ufl_shape[0] == epsilon_sph.ufl_shape[1])
    #     dim = epsilon_sph.ufl_shape[0]

    #     sigma = dim * self.K * epsilon_sph

    #     return sigma

################################################################################

class HookeDevElasticMaterial(ElasticMaterial):



    def __init__(self,
            kinematics,
            parameters):

        self.kinematics = kinematics

        self.G = self.get_G_from_parameters(parameters)

        self.psi   =   self.G * dolfin.inner(self.kinematics.epsilon_dev, self.kinematics.epsilon_dev)
        self.sigma = 2*self.G *              self.kinematics.epsilon_dev

        self.Sigma = self.sigma
        self.P     = self.sigma

        if (self.kinematics.dim == 2):
            self.sigma_ZZ = -2*self.G/3 * dolfin.tr(self.kinematics.epsilon)



    # def get_free_energy(self,
    #         U=None,
    #         epsilon=None,
    #         epsilon_dev=None):

    #     epsilon_dev = self.get_epsilon_dev_from_U_epsilon_or_epsilon_dev(
    #         U, epsilon, epsilon_dev)
        
    #     psi   =   self.G * dolfin.inner(epsilon_dev, epsilon_dev)
    #     sigma = 2*self.G *              epsilon_dev

    #     return psi, sigma



    # def get_Cauchy_stress(self,
    #         U=None,
    #         epsilon=None,
    #         epsilon_dev=None):

    #     epsilon_dev = self.get_epsilon_dev_from_U_epsilon_or_epsilon_dev(
    #         U, epsilon, epsilon_dev)
        
    #     sigma = 2*self.G * epsilon_dev

    #     return sigma
