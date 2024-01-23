#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Cécile Patte, 2019-2021                                              ###
###                                                                          ###
### INRIA, Palaiseau, France                                                 ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_mech as dmech
from .Problem_InverseHyperelasticity import InverseHyperelasticityProblem

################################################################################

class TwoFormulationsInversePoroProblem(InverseHyperelasticityProblem):



    def __init__(self,
            eta,
            kappa,
            n,
            w_contact = 1,
            type_porosity = 'mixed',
            p0 = 0):

        InverseHyperelasticityProblem.__init__(self,w_incompressibility=False)
        self.eta                 = dolfin.Constant(eta)
        self.kappa               = dolfin.Constant(kappa)
        self.n                   = dolfin.Constant(n)
        self.p0                  = dolfin.Constant(p0)
        self.w_contact           = w_contact
        self.inertia             = None
        self.porosity_init_val   = None
        self.porosity_init_field = None
        self.porosity_given      = None
        self.config_porosity     = None
        self.type_porosity       = type_porosity

        assert (w_contact == 0) or (eta == 0)



    def add_porosity_subsol(self,
            degree):

        if self.porosity_init_val is not None:
            init_val = numpy.array([self.porosity_init_val])
        else:
            init_val = self.porosity_init_val

        if (degree == 0):
            self.add_scalar_subsol(
                name="Phi0",
                family="DG",
                degree=0,
                init_val=init_val,
                init_field=self.porosity_init_field)
        else:
            self.add_scalar_subsol(
                name="Phi0",
                family="CG",
                degree=degree,
                init_val=init_val,
                init_field=self.porosity_init_field)



    def get_Phi0(self):

        Phi0 = 1 - (1 - self.Phi) * self.kinematics.Je

        return Phi0



    def set_internal_variables_internal(self):

        self.Phi0_fs  = self.sfoi_fs

        self.Phi0     = dolfin.Function(self.Phi0_fs)
        self.Phi0_old = dolfin.Function(self.Phi0_fs)

        self.add_foi(expr=self.Phi0, fs=self.Phi0_fs, name="Phi0")

        self.Phi0_test = dolfin.TestFunction(self.Phi0_fs)
        self.Phi0_tria = dolfin.TrialFunction(self.Phi0_fs)

        if self.porosity_init_val is not None:
            c_expr = dolfin.inner(
                self.Phi0_tria,
                self.Phi0_test) * self.dV
            d_expr = dolfin.inner(
                self.porosity_init_val,
                self.Phi0_test) * self.dV
            local_solver = dolfin.LocalSolver(
                c_expr,
                d_expr)
            local_solver.factorize()
            local_solver.solve_local_rhs(self.Phi0)
        elif self.porosity_init_field is not None:
            self.Phi0.vector()[:] = self.porosity_init_field.array()[:]

        Phi0 = self.get_Phi0()

        self.a_expr = dolfin.inner(
            self.Phi0_tria,
            self.Phi0_test) * self.dV
        self.b_expr = dolfin.inner(
            Phi0,
            self.Phi0_test) * self.dV
        self.local_solver = dolfin.LocalSolver(
            self.a_expr,
            self.b_expr)
        self.local_solver.factorize()



    def update_internal_variables_at_t(self,
            t):

        self.Phi0_old.vector()[:] = self.Phi0.vector()[:]



    def update_internal_variables_after_solve(self,
            dt, t):

        self.local_solver.solve_local_rhs(self.Phi0)



    def restore_old_value(self):

        self.Phi0.vector()[:] = self.Phi0_old.vector()[:]



    def set_subsols(self,
            U_degree=1):

        self.add_displacement_subsol(
            degree=U_degree)

        if self.type_porosity == 'mixed':
            self.add_porosity_subsol(
                degree=U_degree-1)



    def get_porosity_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="Phi0")



    def set_Phi0_and_Phi(self):

        if self.config_porosity == 'reference':
            assert(0)
        elif self.config_porosity == 'deformed':
            self.Phi    = self.porosity_given
            self.Phipos = dolfin.conditional(dolfin.gt(self.Phi, 0), self.Phi, 0)
            self.Phibin = dolfin.conditional(dolfin.gt(self.Phi, 0),        1, 0)
            if self.type_porosity == 'mixed':
                self.Phi0    = self.subsols["Phi0"].subfunc
                self.Phi0pos = dolfin.conditional(dolfin.gt(self.Phi0, 0), self.Phi0, 0)
                self.Phi0bin = dolfin.conditional(dolfin.gt(self.Phi0, 0),         1, 0)
            elif self.type_porosity == 'internal':
                self.set_internal_variables_internal()
                self.inelastic_behaviors_internal += [self]
                Phi0 = self.get_Phi0()
                self.Phi0pos = dolfin.conditional(dolfin.gt(Phi0, 0), Phi0, 0)
                self.Phi0bin = dolfin.conditional(dolfin.gt(Phi0, 0),    1, 0)



    def set_kinematics(self):

        InverseHyperelasticityProblem.set_kinematics(self)

        self.set_Phi0_and_Phi()
        if self.type_porosity == 'internal':
            # self.kinematics.Js = self.kinematics.Je * (1 - self.get_Phi())
            self.kinematics.Js = self.kinematics.Je * (1 - self.Phi)
        elif self.type_porosity == 'mixed':
            self.kinematics.Js = self.kinematics.Je * (1 - self.Phi)
            self.kinematics.Js_pos = self.kinematics.Je * (1 - self.Phipos)



    def set_materials(self,
            elastic_behavior=None,
            elastic_behavior_dev=None,
            elastic_behavior_bulk=None,
            subdomain_id=None):

        InverseHyperelasticityProblem.set_materials(self,
            elastic_behavior=elastic_behavior,
            elastic_behavior_dev=elastic_behavior_dev,
            elastic_behavior_bulk=elastic_behavior_bulk,
            subdomain_id=subdomain_id)

        self.wbulk_behavior = dmech.SkeletonPoroBulkElasticMaterial(
            problem = self,
            parameters = {'kappa':self.kappa})
        self.wpor_behavior = dmech.WporPoroElasticMaterial(
            problem = self,
            parameters = {'eta':self.eta, 'n':self.n},
            type = 'inverse')
            # type = 'exp')



    def set_jac_form_hyperelastic(self):

        jac_form = 0

        for subdomain in self.subdomains :
            E_hyper = subdomain.sigma
            jac_form += dolfin.inner(
                dolfin.diff(
                    E_hyper,
                    self.Phi0),
                dolfin.derivative(
                    self.kinematics.Et,
                    self.subsols["U"].subfunc,
                    self.subsols["U"].dsubtest)) * dolfin.inner(
                dolfin.diff(
                    self.get_Phi0(),
                    dolfin.variable(self.kinematics.Jt)) * dolfin.diff(
                    self.kinematics.Jt,
                    dolfin.variable(self.kinematics.Ft)),
                dolfin.derivative(
                    self.kinematics.Ft,
                    self.subsols["U"].subfunc,
                    self.subsols["U"].dsubtria)) * self.dV

        return jac_form



    def set_variational_formulation(self,
            normal_penalties=[],
            directional_penalties=[],
            surface_tensions=[],
            surface0_loadings=[],
            pressure0_loadings=[],
            gradient_pressure0_loadings=[],
            volume0_loadings=[],
            surface_loadings=[],
            pressure_loadings=[],
            volume_loadings=[],
            gradient_pressure_loadings=[],
            dt=None):

        self.res_form = 0.

        if self.inertia is not None:
            self.res_form += self.inertia / dt * dolfin.inner(
                    self.subsols["U"].subfunc,
                    self.subsols["U"].dsubtest) * self.dV

        for subdomain in self.subdomains :
            self.res_form += dolfin.inner(
                subdomain.sigma,
                dolfin.sym(dolfin.grad(self.subsols["U"].dsubtest))) * self.dV(subdomain.id)

        for loading in surface_loadings:
            self.res_form -= dolfin.inner(
                loading.val,
                self.subsols["U"].dsubtest) * loading.measure

        for loading in pressure_loadings:
            self.res_form -= dolfin.inner(
               -loading.val * self.mesh_normals,
                self.subsols["U"].dsubtest) * loading.measure

        for loading in gradient_pressure_loadings:
            self.res_form -= dolfin.inner(
                dolfin.inner(
                    -(dolfin.SpatialCoordinate(self.mesh) - loading.xyz_ini),
                    loading.val) * self.mesh_normals,
                self.subsols["U"].dsubtest) * loading.measure

        for loading in volume_loadings:
            self.res_form -= dolfin.inner(
                loading.val,
                self.subsols["U"].dsubtest) * loading.measure

        if self.w_contact:
            self.res_form += self.wbulk_behavior.get_res_term(self.Phi0pos, self.Phipos, w_U=1)
        else:
            if self.type_porosity == 'mixed':
                self.res_form += self.wbulk_behavior.get_res_term(self.Phi0, self.Phi, w_U=1)
            elif self.type_porosity == 'internal':
                self.res_form += self.wbulk_behavior.get_res_term(self.get_Phi0(), self.Phi, w_U=1)

        if self.type_porosity == 'mixed':
            # p0_loading_val = pressure0_loadings[0].val
            # self.res_form += dolfin.inner(
            #         p0_loading_val,
            #         self.subsols["Phi0"].dsubtest) * self.dV
            self.res_form += self.wbulk_behavior.get_res_term(self.Phi0, self.Phi, w_Phi0=1)
            self.res_form += self.wpor_behavior.get_res_term(w_Phi0=1)

        self.jac_form = dolfin.derivative(
            self.res_form,
            self.sol_func,
            self.dsol_tria)

        if self.type_porosity == 'internal':
            self.jac_form += self.set_jac_form_hyperelastic()
            if self.w_contact:
                self.jac_form += self.wbulk_behavior.get_jac_term(self.Phi0pos, self.Phipos, w_Phi0=1)
            else:
                self.jac_form += self.wbulk_behavior.get_jac_term(self.get_Phi0(), self.Phi, w_Phi0=1)



    def add_Phi0_qois(self):

        basename = "PHI0_"

        self.add_qoi(
            name=basename,
            expr=self.Phi0 / self.mesh_V0 * self.dV)



    def add_Phi_qois(self):

        basename = "PHI_"

        self.add_qoi(
            name=basename,
            expr=self.Phi / self.mesh_V0 * self.dV)



    def add_Js_qois(self):

        basename = "Js_"

        self.add_qoi(
            name=basename,
            expr=self.kinematics.Js / self.mesh_V0 * self.dV)



    def add_dWpordJs_qois(self):

        basename = "dWpordJs_"

        self.add_qoi(
            name=basename,
            expr=self.wpor_behavior.get_dWpordJs() / self.mesh_V0 * self.dV)



    def add_dWbulkdJs_qois(self):

        basename = "dWbulkdJs_"

        self.add_qoi(
            name=basename,
            expr=self.wbulk_behavior.get_dWbulkdJs(self.Phi0, self.Phi) / self.mesh_V0 * self.dV)



    def add_Phi0bin_qois(self):

        basename = "PHI0bin_"

        self.add_qoi(
            name=basename,
            expr=self.Phi0bin / self.mesh_V0 * self.dV)



    def add_Phi0pos_qois(self):

        basename = "PHI0pos_"

        self.add_qoi(
            name=basename,
            expr=self.Phi0pos / self.mesh_V0 * self.dV)



    def add_Phipos_qois(self):

        basename = "PHIpos_"

        self.add_qoi(
            name=basename,
            expr=self.Phipos / self.mesh_V0 * self.dV)



    def add_Phibin_qois(self):

        basename = "PHIbin_"

        self.add_qoi(
            name=basename,
            expr=self.Phibin / self.mesh_V0 * self.dV)



    def add_mnorm_qois(self):

        basename = "M_NORM"
        value = self.kinematics.Je * self.Phi - self.Phi0

        self.add_qoi(
            name=basename,
            expr=value / self.mesh_V0 * self.dV)
