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
from .Material_Inelastic_Growth import GrowthInelasticMaterial

################################################################################

class TimeDrivenGrowthInelasticMaterial(GrowthInelasticMaterial):



    def __init__(self,
            problem,
            parameters):

        self.problem = problem

        if ("taug" in parameters):
            self.taug = parameters["taug"]
        else:
            assert (0),\
                "No parameter found: \"+str(parameters)+\". Must provide taug. Aborting."        


    # growth kinematics
    def get_Fg(self,
            thetag):

        return (1+thetag) * numpy.eye(self.problem.dim)



    def get_Fg_expr(self,
            thetag):

        return dolfin.Constant(self.get_Fg(thetag=thetag))



    # growth kinetics
    def get_thetag_dot(self):

        return 1./self.taug



    def get_thetag_dot_expr(self):

        return dolfin.Constant(self.get_thetag_dot())



    def get_thetag_at_t(self,
            t):

        return self.get_thetag_dot()*t



    def get_thetag_at_t_expr(self,
            t):

        return dolfin.Constant(self.get_thetag_at_t())



###################################################### for mixed formulation ###



    def set_internal_variables_mixed(self):

        self.Fg     = self.get_Fg_expr(thetag=self.problem.subsols["thetag"].subfunc )
        self.Fg_old = self.get_Fg_expr(thetag=self.problem.subsols["thetag"].func_old)

        self.problem.add_foi(
            expr=self.Fg,
            fs=self.problem.mfoi_fs,
            name="Fg")



    def get_res_term(self,
            dt):

        thetag_dot = self.get_thetag_dot_expr()

        thetag_new = self.problem.subsols["thetag"].func_old + thetag_dot * dolfin.Constant(dt)

        res_form = dolfin.inner(
            self.problem.subsols["thetag"].subfunc - thetag_new,
            self.problem.subsols["thetag"].dsubtest) * self.problem.dV

        return res_form



########################################## for internal variable formulation ###



    def set_internal_variables_internal(self):

        self.thetag     = self.get_thetag_at_t_expr(t=0.)
        self.thetag_old = self.get_thetag_at_t_expr(t=0.)

        self.Fg     = self.get_Fg_expr(thetag=self.thetag    )
        self.Fg_old = self.get_Fg_expr(thetag=self.thetag_old)

        self.problem.add_foi(
            expr=self.thetag,
            fs=self.problem.sfoi_fs,
            name="thetag")
        self.problem.add_foi(
            expr=self.Fg,
            fs=self.problem.mfoi_fs,
            name="Fg")



    def update_internal_variables_at_t(self,
            t):

        self.thetag_old.assign(self.thetag)
        self.Fg_old.assign(self.Fg)
        self.thetag.assign(self.get_thetag_at_t(t=t))
        self.Fg.assign(self.get_Fg_expr(thetag=self.thetag.values()[0]))



    def update_internal_variables_after_solve(self,
            dt, t):

        pass
