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
from .Material_Inelastic import InelasticMaterial

################################################################################

class RelaxationInelasticMaterial(InelasticMaterial):



    def __init__(self,
            kinematics,
            parameters):

        if ("taur" in parameters):
            self.taur = parameters["taur"]
        else:
            assert (0),\
                "No parameter found: \"+str(parameters)+\". Must provide taur. Aborting."   



###################################################### for mixed formulation ###



    def set_internal_variables_mixed(self,
            problem):

        self.I = dolfin.Identity(problem.dim)

        self.Fr     = problem.subsols["Fr"].subfunc
        self.Fr_old = problem.subsols["Fr"].func_old



    def get_res_term(self,
            problem,
            dt):

        # print(problem)
        # print(problem.unloaded_kinematics)
        if (hasattr(problem, "unloaded_kinematics")):
            Fr_dot = problem.unloaded_kinematics.Ee_mid / dolfin.Constant(self.taur)
        else:
            Fr_dot = problem.kinematics.Ee_mid / dolfin.Constant(self.taur)
        # Fr_dot = (problem.kinematics.Fe_mid - self.I) / dolfin.Constant(self.taur)

        Fr_new = self.Fr_old + Fr_dot * dolfin.Constant(dt)

        res_form = dolfin.inner(
            problem.subsols["Fr"].subfunc - Fr_new,
            problem.subsols["Fr"].dsubtest) * problem.dV

        return res_form



########################################## for internal variable formulation ###



    def set_internal_variables(self,
            problem):

        assert (0)



    def update_internal_variables_at_t(self,
            problem):

        assert (0)



    def update_internal_variables_after_solve(self,
            problem,
            dt, t):

        assert (0)
