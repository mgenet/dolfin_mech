#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import math
import numpy

import dolfin_mech as dmech
from .Problem import Problem

################################################################################

class SphericalProblem(Problem):



    def __init__(self,
            w_incompressibility=False):

        Problem.__init__(self)

        self.dim = 3

        self.w_incompressibility = w_incompressibility



    def set_mesh(self,
            Ri,
            Re,
            N):

        self.Ri = Ri
        self.Re = Re
        self.N  = N

        self.mesh = dolfin.IntervalMesh(self.N, self.Ri, self.Re)

        self.dR = dolfin.Measure(
            "dx",
            domain=self.mesh)

        # self.R = dolfin.MeshCoordinates(self.mesh)
        self.R_fe = dolfin.FiniteElement(
            family="CG",
            cell=self.mesh.ufl_cell(),
            degree=1)
        self.R = dolfin.Expression(
            "x[0]",
            element=self.R_fe)

        self.Vi0 = 4/3 * math.pi * self.Ri**3
        self.Ve0 = 4/3 * math.pi * self.Re**3
        self.Vm0 = dolfin.assemble(4 * math.pi * self.R**2 * self.dR)

        self.Ri_sd = dolfin.CompiledSubDomain("near(x[0], x0) && on_boundary", x0=self.Ri)
        self.Re_sd = dolfin.CompiledSubDomain("near(x[0], x0) && on_boundary", x0=self.Re)

        self.Ri_id = 1
        self.Re_id = 2

        self.boundaries_mf = dolfin.MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        self.boundaries_mf.set_all(0)
        self.Ri_sd.mark(self.boundaries_mf, self.Ri_id)
        self.Re_sd.mark(self.boundaries_mf, self.Re_id)

        self.dS = dolfin.Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=self.boundaries_mf)



    def add_rho_subsol(self,
            degree):

        self.add_scalar_subsol(
            name="Rho",
            family="CG",
            degree=degree)



    def get_rho_function_space(self):

        if (len(self.subsols) == 1):
            return self.sol_fs
        else:
            return self.get_subsol_function_space(name="Rho")



    def add_pressure_subsol(self,
            degree):

        if (degree == 0):
            self.add_scalar_subsol(
                name="P",
                family="DG",
                degree=0)
        else:
            self.add_scalar_subsol(
                name="P",
                family="CG",
                degree=degree)



    def get_pressure_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="P")



    def set_subsols(self,
            Rho_degree=1):

        self.add_rho_subsol(
            degree=Rho_degree)

        if (self.w_incompressibility):
            self.add_pressure_subsol(
                degree=Rho_degree-1)



    def set_solution(self,
            Rho_degree=1):

        self.set_subsols(
            Rho_degree=Rho_degree)
        self.set_solution_finite_element()
        self.set_solution_function_space()
        self.set_solution_functions()

        quadrature_degree = max(1, 2*(Rho_degree-1))
        self.set_quadrature_degree(
            quadrature_degree=quadrature_degree)

        self.set_foi_finite_elements_DG(
            degree=0)
        self.set_foi_function_spaces()



    def set_kinematics(self):

        self.kinematics = dmech.SphericalKinematics(
            R=self.R,
            Rho=self.subsols["Rho"].subfunc,
            Rho_old=self.subsols["Rho"].func_old)

        # self.add_foi(expr=self.kinematics.Fe, fs=self.mfoi_fs, name="F")
        # self.add_foi(expr=self.kinematics.Je, fs=self.sfoi_fs, name="J")
        # self.add_foi(expr=self.kinematics.Ce, fs=self.mfoi_fs, name="C")
        # self.add_foi(expr=self.kinematics.Ee, fs=self.mfoi_fs, name="E")



    def set_materials(self,
            elastic_behavior=None,
            elastic_behavior_dev=None,
            elastic_behavior_bulk=None,
            subdomain_id=None):

        if (self.w_incompressibility):
            assert (elastic_behavior      is     None)
            assert (elastic_behavior_dev  is not None)
            assert (elastic_behavior_bulk is     None)
        else:
            assert  ((elastic_behavior      is not None)
                or  ((elastic_behavior_dev  is not None)
                and  (elastic_behavior_bulk is not None)))

        subdomain = dmech.SubDomain(
            problem=self,
            elastic_behavior=elastic_behavior,
            elastic_behavior_dev=elastic_behavior_dev,
            elastic_behavior_bulk=elastic_behavior_bulk,
            id=subdomain_id)

        self.subdomains += [subdomain]

        # self.add_foi(expr=subdomain.Sigma, fs=self.mfoi_fs, name="Sigma")
        # self.add_foi(expr=subdomain.PK1  , fs=self.mfoi_fs, name="PK1"  )
        # self.add_foi(expr=subdomain.sigma, fs=self.mfoi_fs, name="sigma")



    def set_variational_formulation(self,
            normal_penalties=[],
            directional_penalties=[],
            surface_tensions=[],
            surface0_loadings=[],
            pressure0_loadings=[],
            volume0_loadings=[],
            surface_loadings=[],
            pressure_loadings=[],
            volume_loadings=[],
            dt=None):

        self.Pi = sum([subdomain.Psi * 4 * math.pi * self.R**2 * self.dR(subdomain.id) for subdomain in self.subdomains])
        # print (self.Pi)

        self.res_form = dolfin.derivative(
            self.Pi,
            self.sol_func,
            self.dsol_test)

        for loading in pressure_loadings:
            self.res_form -= loading.val * 4 * math.pi * self.R**2 * dolfin.inv(self.kinematics.Ft)[0,0] * self.R * self.subsols["Rho"].dsubtest * loading.measure

        self.jac_form = dolfin.derivative(
            self.res_form,
            self.sol_func,
            self.dsol_tria)



    def add_strain_qois(self):

        self.add_qoi(
            name="E_RR",
            expr=self.kinematics.Ee[0,0] * 4 * math.pi * self.R**2 * self.dR)
        self.add_qoi(
            name="E_TT",
            expr=self.kinematics.Ee[1,1] * 4 * math.pi * self.R**2 * self.dR)
        self.add_qoi(
            name="E_PP",
            expr=self.kinematics.Ee[2,2] * 4 * math.pi * self.R**2 * self.dR)
        self.add_qoi(
            name="E_RT",
            expr=self.kinematics.Ee[0,1] * 4 * math.pi * self.R**2 * self.dR)
        self.add_qoi(
            name="E_TP",
            expr=self.kinematics.Ee[1,2] * 4 * math.pi * self.R**2 * self.dR)
        self.add_qoi(
            name="E_PR",
            expr=self.kinematics.Ee[2,0] * 4 * math.pi * self.R**2 * self.dR)



    def add_J_qois(self):

        self.add_qoi(
            name="J",
            expr=self.kinematics.Je * 4 * math.pi * self.R**2 * self.dR)



    def add_stress_qois(self,
            stress_type="cauchy"):

        nb_subdomain = 0
        for subdomain in self.subdomains:
            nb_subdomain += 1

        if nb_subdomain == 0:
            if (stress_type in ("cauchy", "sigma")):
                basename = "s_"
                stress = self.sigma
            elif (stress_type in ("piola", "PK2", "Sigma")):
                basename = "S_"
                stress = self.Sigma
            elif (stress_type in ("PK1", "P")):
                basename = "P_"
                stress = self.PK1

        elif nb_subdomain == 1:
            if (stress_type in ("cauchy", "sigma")):
                basename = "s_"
                stress = self.subdomains[0].sigma
            elif (stress_type in ("piola", "PK2", "Sigma")):
                basename = "S_"
                stress = self.subdomains[0].Sigma
            elif (stress_type in ("PK1", "P")):
                basename = "P_"
                stress = self.subdomains[0].PK1

        self.add_qoi(
            name=basename+"XX",
            expr=stress[0,0] * 4 * math.pi * self.R**2 * self.dR)
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"YY",
                expr=stress[1,1] * 4 * math.pi * self.R**2 * self.dR)
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"ZZ",
                    expr=stress[2,2] * 4 * math.pi * self.R**2 * self.dR)
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"XY",
                expr=stress[0,1] * 4 * math.pi * self.R**2 * self.dR)
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"YZ",
                    expr=stress[1,2] * 4 * math.pi * self.R**2 * self.dR)
                self.add_qoi(
                    name=basename+"ZX",
                    expr=stress[2,0] * 4 * math.pi * self.R**2 * self.dR)



    def add_P_qois(self):

        nb_subdomain = 0
        for subdomain in self.subdomains:
            nb_subdomain += 1
        # print nb_subdomain

        if nb_subdomain == 0:
            basename = "P_"
            P = -1/3 * dolfin.tr(self.sigma)
        elif nb_subdomain == 1:
            basename = "P_"
            P = -1/3 * dolfin.tr(self.subdomains[0].sigma)

        self.add_qoi(
            name=basename,
            expr=P * 4*math.pi * self.R**2 * self.dR)
