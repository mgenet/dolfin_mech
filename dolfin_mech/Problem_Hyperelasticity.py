#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin_mech as dmech
from .Problem import Problem

################################################################################

class HyperelasticityProblem(Problem):



    def __init__(self,
            w_incompressibility=False,
            mesh=None,
            define_facet_normals=False,
            domains_mf=None,
            boundaries_mf=None,
            points_mf=None,
            displacement_degree=None,
            pressure_degree=None,
            quadrature_degree=None,
            foi_degree=0,
            elastic_behavior=None,
            elastic_behaviors=None):

        Problem.__init__(self)

        self.w_incompressibility = w_incompressibility

        if (mesh is not None):
            self.set_mesh(
                mesh=mesh,
                define_facet_normals=define_facet_normals)

            self.set_measures(
                domains=domains_mf,
                boundaries=boundaries_mf,
                points=points_mf)

            self.set_subsols(
                displacement_degree=displacement_degree,
                pressure_degree=pressure_degree)
            self.set_solution_finite_element()
            self.set_solution_function_space()
            self.set_solution_functions()

            self.set_quadrature_degree(
                quadrature_degree=quadrature_degree)

            self.set_foi_finite_elements_DG(
                degree=foi_degree)
            self.set_foi_function_spaces()

            self.set_kinematics()

            assert (elastic_behavior is     None) or (elastic_behaviors is     None)
            assert (elastic_behavior is not None) or (elastic_behaviors is not None)
            if (elastic_behavior is not None):
                elastic_behaviors = [elastic_behavior]

            self.add_elasticity_operators(
                elastic_behaviors=elastic_behaviors)



    def get_displacement_name(self):
        return "U"



    def add_displacement_subsol(self,
            degree):

        self.displacement_degree = degree
        self.add_vector_subsol(
            name=self.get_displacement_name(),
            family="CG",
            degree=self.displacement_degree)



    def get_displacement_subsol(self):

        return self.get_subsol(self.get_displacement_name())



    def get_displacement_function_space(self):

        if (len(self.subsols) == 1):
            return self.sol_fs
        else:
            return self.get_subsol_function_space(name=self.get_displacement_name())



    def get_pressure_name(self):
        return "P"



    def add_pressure_subsol(self,
            degree):

        self.pressure_degree = degree
        if (degree == 0):
            self.add_scalar_subsol(
                name=self.get_pressure_name(),
                family="DG",
                degree=0)
        else:
            self.add_scalar_subsol(
                name=self.get_pressure_name(),
                family="CG",
                degree=self.pressure_degree)



    def get_pressure_subsol(self):

        assert (self.w_incompressibility),\
            "There is no pressure subsol. Aborting."
        return self.get_subsol(self.get_pressure_name())



    def get_pressure_function_space(self):

        assert (self.w_incompressibility),\
            "There is no pressure function space. Aborting."
        return self.get_subsol_function_space(name=self.get_pressure_name())



    def set_subsols(self,
            displacement_degree=1,
            pressure_degree=None):

        self.add_displacement_subsol(
            degree=displacement_degree)

        if (self.w_incompressibility):
            if (pressure_degree is None):
                pressure_degree = displacement_degree-1
            self.add_pressure_subsol(
                degree=pressure_degree)



    def set_quadrature_degree(self,
            quadrature_degree=None):

        if (quadrature_degree is None) or (type(quadrature_degree) == int):
            pass
        elif (quadrature_degree == "full"):
            quadrature_degree = None
        elif (quadrature_degree == "default"):
            if   (self.mesh.ufl_cell().cellname() in ("triangle", "tetrahedron")):
                quadrature_degree = max(2, 4*(self.displacement_degree-1)) # MG20211221: This does not allow to reproduce full integration results exactly, but it is quite close…
            elif (self.mesh.ufl_cell().cellname() in ("quadrilateral", "hexahedron")):
                quadrature_degree = max(2, 4*(self.dim*self.displacement_degree-1))
        else:
            assert (0),\
                "Must provide an int, \"full\", \"default\" or None. Aborting."

        Problem.set_quadrature_degree(self,
            quadrature_degree=quadrature_degree)



    def set_kinematics(self):

        self.kinematics = dmech.Kinematics(
            U=self.get_displacement_subsol().subfunc,
            U_old=self.get_displacement_subsol().func_old,
            Q_expr=self.Q_expr)

        self.add_foi(expr=self.kinematics.F, fs=self.mfoi_fs, name="F")
        self.add_foi(expr=self.kinematics.J, fs=self.sfoi_fs, name="J")
        self.add_foi(expr=self.kinematics.C, fs=self.mfoi_fs, name="C")
        self.add_foi(expr=self.kinematics.E, fs=self.mfoi_fs, name="E")
        if (self.Q_expr is not None):
            self.add_foi(expr=self.kinematics.E_loc, fs=self.mfoi_fs, name="E_loc")



    def get_subdomain_measure(self,
            subdomain_id=None):

        if (subdomain_id is None):
            return self.dV
        else:
            return self.dV(subdomain_id)



    def add_elasticity_operator(self,
            material_model,
            material_parameters,
            subdomain_id=None):

        operator = dmech.HyperElasticityOperator(
            U=self.get_displacement_subsol().subfunc,
            U_test=self.get_displacement_subsol().dsubtest,
            kinematics=self.kinematics,
            material_model=material_model,
            material_parameters=material_parameters,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)



    def add_hydrostatic_pressure_operator(self,
            subdomain_id=None):

        operator = dmech.HyperHydrostaticPressureOperator(
            kinematics=self.kinematics,
            U_test=self.get_displacement_subsol().dsubtest,
            P=self.get_pressure_subsol().subfunc,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)



    def add_incompressibility_operator(self,
            subdomain_id=None):

        operator = dmech.HyperIncompressibilityOperator(
            kinematics=self.kinematics,
            P_test=self.get_pressure_subsol().dsubtest,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)



    def add_elasticity_operators(self,
            elastic_behaviors):

        for elastic_behavior in elastic_behaviors:
            operator = self.add_elasticity_operator(
                material_model=elastic_behavior["model"],
                material_parameters=elastic_behavior["parameters"],
                subdomain_id=elastic_behavior.get("subdomain_id", None))
            suffix = "_"+elastic_behavior["suffix"] if "suffix" in elastic_behavior else ""
            self.add_foi(expr=operator.material.Sigma, fs=self.mfoi_fs, name="Sigma"+suffix)
            self.add_foi(expr=operator.material.sigma, fs=self.mfoi_fs, name="sigma"+suffix)
        if (self.w_incompressibility):
            self.add_hydrostatic_pressure_operator()
            self.add_incompressibility_operator()

        if (self.Q_expr is not None):
            assert (0), "ToDo. Aborting."
            # operator.sigma_loc = dolfin.dot(dolfin.dot(self.Q_expr, operator.sigma), self.Q_expr.T)
            # self.add_foi(expr=operator.sigma_loc, fs=self.mfoi_fs, name="sigma_loc")



    def add_point_displacement_qoi(self,
            name,
            coordinates,
            component):

        self.add_qoi(
            name=name,
            expr=self.get_displacement_subsol().subfunc[component],
            point=coordinates,
            update_type="direct")



    def add_point_position_qoi(self,
            name,
            coordinates,
            component):

        self.add_qoi(
            name=name,
            expr=self.get_displacement_subsol().subfunc[component],
            constant=coordinates[component],
            point=coordinates,
            update_type="direct")



    def add_deformed_volume_qoi(self):

        self.add_qoi(
            name="v",
            expr=self.kinematics.J * self.dV)



    def add_global_strain_qois(self):

        basename = "E_"
        strain = self.kinematics.E

        self.add_qoi(
            name=basename+"XX",
            expr=strain[0,0] * self.dV)
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"YY",
                expr=strain[1,1] * self.dV)
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"ZZ",
                    expr=strain[2,2] * self.dV)
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"XY",
                expr=strain[0,1] * self.dV)
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"YZ",
                    expr=strain[1,2] * self.dV)
                self.add_qoi(
                    name=basename+"ZX",
                    expr=strain[2,0] * self.dV)



    def add_global_stress_qois(self,
            stress_type="cauchy"):

        if (stress_type in ("Cauchy", "cauchy", "sigma")):
            basename = "s_"
            stress = "sigma"
        elif (stress_type in ("Piola", "piola", "PK2", "Sigma")):
            basename = "S_"
            stress = "Sigma"
        elif (stress_type in ("Boussinesq", "boussinesq", "PK1", "P")):
            basename = "P_"
            stress = "P"

        self.add_qoi(
            name=basename+"XX",
            expr=sum([getattr(operator.material, stress)[0,0]*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))]))
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"YY",
                expr=sum([getattr(operator.material, stress)[1,1]*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))]))
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"ZZ",
                    expr=sum([getattr(operator.material, stress)[2,2]*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))]))
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"XY",
                expr=sum([getattr(operator.material, stress)[0,1]*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))]))
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"YZ",
                    expr=sum([getattr(operator.material, stress)[1,2]*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))]))
                self.add_qoi(
                    name=basename+"ZX",
                    expr=sum([getattr(operator.material, stress)[2,0]*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))]))



    def add_global_out_of_plane_stress_qois(self,
            stress_type="PK2"):

        if (stress_type in ("Cauchy", "cauchy", "sigma")):
            assert (0), "To do. Aborting."
        elif (stress_type in ("Piola", "piola", "PK2", "Sigma")):
            basename = "S_ZZ"
            stress = "Sigma_ZZ"
        elif (stress_type in ("Boussinesq", "boussinesq", "PK1", "P")):
            assert (0), "To do. Aborting."

        self.add_qoi(
            name=basename,
            expr=sum([getattr(operator.material, stress)*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))]))



    def add_global_pressure_qoi(self):

        self.add_qoi(
            name="P",
            expr=sum([operator.P*operator.measure for operator in self.operators if hasattr(operator, "P")]))
            # expr=sum([-dolfin.tr(operator.material.sigma)/3*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, "sigma"))])+sum([operator.P*operator.measure for operator in self.operators if hasattr(operator, "P")]))
