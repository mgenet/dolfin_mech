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

class LinearizedSubDomain():



    def __init__(self,
            problem,
            elastic_behavior=None,
            elastic_behavior_dev=None,
            elastic_behavior_bulk=None,
            id=None):

        self.problem = problem
        self.id = id

        if (elastic_behavior is not None):
            self.psi, self.sigma = elastic_behavior.get_free_energy(
                epsilon=self.problem.kinematics.epsilon)
        else:
            if (self.problem.w_incompressibility):
                self.psi_bulk   = -self.problem.subsols["p"].subfunc * dolfin.tr(self.problem.kinematics.epsilon)
                self.sigma_bulk = -self.problem.subsols["p"].subfunc *           self.problem.kinematics.I
            else:
                self.psi_bulk, self.sigma_bulk = elastic_behavior_bulk.get_free_energy(
                    epsilon=self.problem.kinematics.epsilon)
            self.psi_dev, self.sigma_dev = elastic_behavior_dev.get_free_energy(
                epsilon=self.problem.kinematics.epsilon)
            self.psi   = self.psi_bulk   + self.psi_dev
            self.sigma = self.sigma_bulk + self.sigma_dev
