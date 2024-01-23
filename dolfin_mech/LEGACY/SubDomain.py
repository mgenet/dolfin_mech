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

class SubDomain():



    def __init__(self,
            problem,
            elastic_behavior=None,
            elastic_behavior_dev=None,
            elastic_behavior_bulk=None,
            id=None):

        self.problem = problem
        self.id = id

        if (elastic_behavior is not None):
            self.Psi, self.Sigma = elastic_behavior.get_free_energy(
                C=self.problem.kinematics.Ce)
        else:
            if (self.problem.w_incompressibility):
                self.Psi_bulk   = -self.problem.subsols["P"].subfunc * (self.problem.kinematics.Je - 1)
                self.Sigma_bulk = -self.problem.subsols["P"].subfunc *  self.problem.kinematics.Je      * self.problem.kinematics.Ce_inv
            else:
                self.Psi_bulk, self.Sigma_bulk = elastic_behavior_bulk.get_free_energy(
                    C=self.problem.kinematics.Ce)
            self.Psi_dev, self.Sigma_dev = elastic_behavior_dev.get_free_energy(
                C=self.problem.kinematics.Ce)
            self.Psi   = self.Psi_bulk   + self.Psi_dev
            self.Sigma = self.Sigma_bulk + self.Sigma_dev

        self.PK1   = self.problem.kinematics.Ft * self.Sigma
        self.sigma = (1./self.problem.kinematics.Jt) * self.PK1 * self.problem.kinematics.Ft.T

        # self.problem.kinematics.Ee = dolfin.variable(self.problem.kinematics.Ee) # MG20180412: Works here,
        # self.Sigma = dolfin.diff(self.Psi, self.problem.kinematics.Ee)           # MG20180412: but fails at project…
