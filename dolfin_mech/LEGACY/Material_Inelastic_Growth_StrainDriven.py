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
from .Material_Inelastic_Growth import GrowthInelasticMaterial

################################################################################

class StrainDrivenGrowthInelasticMaterial(GrowthInelasticMaterial):



    def __init__(self,
            problem,
            parameters):

        self.problem = problem

        if ("thetag_max" in parameters) and ("Ee_thr" in parameters) and ("taug" in parameters):
            self.thetag_max = parameters["thetag_max"]
            self.Ee_thr = parameters["Ee_thr"]
            self.taug = parameters["taug"]
        else:
            assert (0),\
                "No parameter found: \"+str(parameters)+\". Must provide thetag_max & Ee_thr & taug. Aborting."


    # growth kinematics
    def get_Fg(self,
            thetag):

        self.I = dolfin.Identity(self.problem.dim)
        Fg = (1+thetag) * self.I

        return Fg



    # growth kinetics
    def get_thetag_dot(self):

        # deltaE = (dolfin.sqrt(dolfin.inner(
        #     self.problem.kinematics.Ee_mid,
        #     self.problem.kinematics.Ee_mid)) - dolfin.Constant(self.Ee_thr)) / dolfin.Constant(self.Ee_thr)
        # deltaE_pos = dolfin.conditional(dolfin.ge(deltaE, 0.), deltaE, 0.)

        assert (self.problem.dim==2),\
            "Eigenvalue decomposition only works in 2D. Aborting."
        Ee_mid = self.problem.kinematics.Ee_mid
        Ee_mid_lp = (dolfin.tr(Ee_mid) + dolfin.sqrt(dolfin.tr(Ee_mid)**2 - 4*dolfin.det(Ee_mid)))/2
        Ee_mid_lm = (dolfin.tr(Ee_mid) - dolfin.sqrt(dolfin.tr(Ee_mid)**2 - 4*dolfin.det(Ee_mid)))/2
        Ee_mid_lp_pos = dolfin.conditional(dolfin.ge(Ee_mid_lp, 0.), Ee_mid_lp, 0.)
        Ee_mid_lm_pos = dolfin.conditional(dolfin.ge(Ee_mid_lm, 0.), Ee_mid_lm, 0.)
        deltaE = (dolfin.sqrt(Ee_mid_lp_pos**2 + Ee_mid_lm_pos**2) - dolfin.Constant(self.Ee_thr)) / dolfin.Constant(self.Ee_thr)
        deltaE_pos = dolfin.conditional(dolfin.ge(deltaE, 0.), deltaE, 0.)

        thetag_mid = (self.problem.subsols["thetag"].func_old + self.problem.subsols["thetag"].subfunc)/2
        delatthetag = ((dolfin.Constant(self.thetag_max) - thetag_mid) / dolfin.Constant(self.thetag_max))
        delatthetag_pos = dolfin.conditional(dolfin.ge(delatthetag, 0.), delatthetag, 0.)

        thetag_dot = delatthetag_pos**2 * deltaE_pos**2 / dolfin.Constant(self.taug)

        return thetag_dot



###################################################### for mixed formulation ###



    def set_internal_variables_mixed(self):

        self.Fg     = self.get_Fg(thetag=self.problem.subsols["thetag"].subfunc    )
        self.Fg_old = self.get_Fg(thetag=self.problem.subsols["thetag"].subfunc_old)

        self.problem.add_foi(
            expr=self.Fg,
            fs=self.problem.mfoi_fs,
            name="Fg")



    def get_res_term(self,
            dt):

        thetag_dot = self.get_thetag_dot()

        thetag_new = self.problem.subsols["thetag"].func_old + thetag_dot * dolfin.Constant(dt)

        res_form = dolfin.inner(
            self.problem.subsols["thetag"].subfunc - thetag_new,
            self.problem.subsols["thetag"].dsubtest) * self.problem.dV

        return res_form



########################################## for internal variable formulation ###



    def set_internal_variables_internal(self):

        assert (0)



    def update_internal_variables_at_t(self,
            t):

        assert (0)



    def update_internal_variables_after_solve(self,
            dt, t):

        assert (0)
