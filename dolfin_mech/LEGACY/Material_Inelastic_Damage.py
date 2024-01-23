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
from .Material_Inelastic import InelasticMaterial

################################################################################

class DamageInelasticMaterial(InelasticMaterial):



    def __init__(self,
            problem,
            parameters):

        self.problem = problem

        if ("epsilon0" in parameters) and ("epsilon1" in parameters) and ("gamma" in parameters):
            self.epsilon0 = parameters["epsilon0"]
            self.epsilon1 = parameters["epsilon1"]
            self.gamma    = parameters["gamma"]
        else:
            assert (0),\
                "No parameter found: \"+str(parameters)+\". Must provide epsilon0 & epsilon1 & gamma. Aborting."



    def get_d(self):

        d = ((dolfin.tr(self.problem.kinematics.epsilon) - dolfin.Constant(self.epsilon0))/(dolfin.Constant(self.epsilon1)-dolfin.Constant(self.epsilon0)))**dolfin.Constant(self.gamma)
        d = dolfin.conditional(dolfin.ge(d, 0.), d, 0.)
        d = dolfin.conditional(dolfin.le(d, 1.), d, 1.)
        # d = dolfin.conditional(dolfin.ge(d, self.d_old), d, self.d_old) # MG20200112: Does not work with mixed formulation?

        return d



###################################################### for mixed formulation ###



    def set_internal_variables_mixed(self):

        self.d     = self.problem.subsols["d"].func # MG20200117: why not subfunc?
        self.d_old = self.problem.subsols["d"].func_old

        self.problem.psi   = (1 - self.d) * self.problem.psi_eff
        self.problem.sigma = (1 - self.d) * self.problem.sigma_eff



    def get_res_term(self,
            dt):

        d = self.get_d()

        res_form = dolfin.inner(
            self.problem.subsols["d"].subfunc - d,
            self.problem.subsols["d"].dsubtest) * self.problem.dV

        return res_form



########################################## for internal variable formulation ###



    def set_internal_variables_internal(self):

        self.d_fs = self.problem.sfoi_fs

        self.d      = dolfin.Function(self.d_fs)
        self.d_old  = dolfin.Function(self.d_fs)

        self.problem.add_foi(expr=self.d, fs=self.d_fs, name="d")

        self.problem.psi   = (1 - self.d) * self.problem.psi_eff
        self.problem.sigma = (1 - self.d) * self.problem.sigma_eff

        self.d_test = dolfin.TestFunction(self.d_fs)
        self.d_tria = dolfin.TrialFunction(self.d_fs)

        d = self.get_d()

        self.a_expr = dolfin.inner(
            self.d_tria,
            self.d_test) * self.problem.dV
        self.b_expr = dolfin.inner(
            d,
            self.d_test) * self.problem.dV
        self.local_solver = dolfin.LocalSolver(
            self.a_expr,
            self.b_expr)
        self.local_solver.factorize()



    def update_internal_variables_at_t(self,
            t):

        self.d_old.vector()[:] = self.d.vector()[:]



    def get_jac_term(self):

        d = self.get_d()

        jac_form = dolfin.inner(
            dolfin.diff(
                self.problem.sigma,
                dolfin.variable(self.d)),
            dolfin.derivative(
                self.problem.kinematics.epsilon,
                self.problem.subsols["u"].subfunc,
                self.problem.subsols["u"].dsubtest)) * dolfin.inner(
            dolfin.diff(
                d,
                dolfin.variable(self.problem.kinematics.epsilon)),
            dolfin.derivative(
                self.problem.kinematics.epsilon,
                self.problem.subsols["u"].subfunc,
                self.problem.subsols["u"].dsubtria)) * self.problem.dV

        return jac_form



    def update_internal_variables_after_solve(self,
            dt, t):

        self.local_solver.solve_local_rhs(self.d)



    def restore_old_value(self):

        self.d.vector()[:] = self.d_old.vector()[:]
