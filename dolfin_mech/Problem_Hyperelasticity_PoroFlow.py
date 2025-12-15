#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_mech as dmech
from .Problem                 import Problem
from .Problem_Hyperelasticity import HyperelasticityProblem

################################################################################

class PoroHyperelasticityProblem(HyperelasticityProblem):



    def __init__(self,
            mesh=None,
            define_facet_normals=False,
            domains_mf=None,
            boundaries_mf=None,
            points_mf=None,
            displacement_degree=1,
            porosity_known="Phis0",
            porosity_degree=None,
            porosity_init_val=None,
            porosity_init_fun=None,
            quadrature_degree=None,
            foi_degree=0,
            skel_behavior=None,
            skel_behaviors=[],
            bulk_behavior=None,
            bulk_behaviors=[],
            pore_behavior=None,
            pore_behaviors=[],
            w_pressure_balancing_gravity=0):

        Problem.__init__(self)

        self.set_mesh(
            mesh=mesh,
            define_facet_normals=define_facet_normals)

        self.set_measures(
            domains=domains_mf,
            boundaries=boundaries_mf,
            points=points_mf)

        assert (porosity_known in ("Phis0", "phis"))
        self.set_known_and_unknown_porosity(porosity_known)

        assert (porosity_init_val is None) or (porosity_init_fun is None)
        self.init_known_porosity(
            porosity_init_val=porosity_init_val,
            porosity_init_fun=porosity_init_fun)
        
        self.w_pressure_balancing_gravity = w_pressure_balancing_gravity
        self.set_subsols(
            displacement_degree=displacement_degree,
            porosity_degree=porosity_degree,
            porosity_init_val=porosity_init_val,
            porosity_init_fun=porosity_init_fun)
        
        self.set_solution_finite_element()
        self.set_solution_function_space()
        self.set_solution_functions()

        self.set_quadrature_degree(
            quadrature_degree=quadrature_degree)

        self.set_foi_finite_elements_DG(
            degree=foi_degree)
        self.set_foi_function_spaces()

        self.set_kinematics()

        self.set_porosity_fields()
        self.add_local_porosity_fois()

        assert (skel_behavior is     None) or (len(skel_behaviors)==0),\
            "Cannot provide both skel_behavior & skel_behaviors. Aborting."
        assert (skel_behavior is not None) or (len(skel_behaviors) >0),\
            "Need to provide skel_behavior or skel_behaviors. Aborting."
        if (skel_behavior is not None):
            skel_behaviors = [skel_behavior]
        self.add_Wskel_operators(skel_behaviors)

        assert (bulk_behavior is     None) or (len(bulk_behaviors)==0),\
            "Cannot provide both bulk_behavior & bulk_behaviors. Aborting."
        assert (bulk_behavior is not None) or (len(bulk_behaviors) >0),\
            "Need to provide bulk_behavior or bulk_behaviors. Aborting."
        if (bulk_behavior is not None):
            bulk_behaviors = [bulk_behavior]
        self.add_Wbulk_operators(bulk_behaviors)

        assert (pore_behavior is None) or (len(pore_behaviors)==0),\
            "Cannot provide both pore_behavior & pore_behaviors. Aborting."
        if (pore_behavior is not None):
            pore_behaviors = [pore_behavior]
        self.add_Wpore_operators(pore_behaviors)



    def set_known_and_unknown_porosity(self,
            porosity_known):

        self.porosity_known = porosity_known
        if (self.porosity_known == "Phis0"):
            self.porosity_unknown = "Phis"
        elif (self.porosity_known == "phis"):
            self.porosity_unknown = "Phis0"



    def init_known_porosity(self,
            porosity_init_val,
            porosity_init_fun):

        if   (porosity_init_val is not None):
            setattr(self, self.porosity_known, dolfin.Constant(porosity_init_val))
        elif (porosity_init_fun is not None):
            setattr(self, self.porosity_known, porosity_init_fun)



    def add_porosity_subsol(self,
            degree,
            init_val=None,
            init_fun=None):

        if (degree == 0):
            self.porosity_subsol = self.add_scalar_subsol(
                name=self.porosity_unknown,
                family="DG",
                degree=0,
                init_val=init_val,
                init_fun=init_fun)
        else:
            self.porosity_subsol = self.add_scalar_subsol(
                name=self.porosity_unknown,
                family="CG",
                degree=degree,
                init_val=init_val,
                init_fun=init_fun)



    def add_pressure_balancing_gravity_subsol(self,
            degree=1):

        self.pressure_balancing_gravity_subsol = self.add_scalar_subsol(
            name="pressure_balancing_gravity",
            family="CG",
            degree=degree)
    


    def add_lmbda_subsol(self,
            init_val=None):

        self.lmbda_subsol = self.add_vector_subsol(
            name="lmbda",
            family="R",
            degree=0,
            init_val=init_val)



    def add_mu_subsol(self,
            init_val=None):

        self.mu_subsol = self.add_vector_subsol(
            name="mu",
            family="R",
            degree=0,
            init_val=init_val)
    

    
    def add_gamma_subsol(self):

        self.gamma_subsol = self.add_scalar_subsol(
            name="gamma",
            family="R",
            degree=0)
    


    def get_deformed_center_of_mass(self):
        
        M = dolfin.assemble(getattr(self, self.porosity_known)*self.dV)
        center_of_mass = numpy.empty(self.dim)
        for k_dim in range(self.dim):
            center_of_mass[k_dim] = dolfin.assemble(getattr(self, self.porosity_known)*self.X[k_dim]*self.dV)/M
        return center_of_mass



    def add_deformed_center_of_mass_subsol(self):
        
        self.deformed_center_of_mass_subsol = self.add_vector_subsol(
            name="xg",
            family="R",
            degree=0,
            init_val=self.get_deformed_center_of_mass())



    def set_subsols(self,
            displacement_degree=1,
            porosity_degree=None,
            porosity_init_val=None,
            porosity_init_fun=None):

        self.add_displacement_subsol(
            degree=displacement_degree)

        if (porosity_degree is None):
            porosity_degree = displacement_degree-1
        self.add_porosity_subsol(
            degree=porosity_degree,
            init_val=porosity_init_val,
            init_fun=porosity_init_fun)
        
        if (self.w_pressure_balancing_gravity):
            self.add_pressure_balancing_gravity_subsol()
            self.add_gamma_subsol()
            self.add_lmbda_subsol()
            self.add_mu_subsol()
            self.add_deformed_center_of_mass_subsol()



    def set_porosity_fields(self):

        if (self.porosity_known == "Phis0"):
            self.Phis = self.porosity_subsol.subfunc
            self.phis = self.Phis/self.kinematics.J
        elif (self.porosity_known == "phis"):
            self.Phis0 = self.porosity_subsol.subfunc
            self.Phis = self.phis*self.kinematics.J



    def add_local_porosity_fois(self):

        if (self.porosity_known == "Phis0"): self.add_foi(
            expr=self.Phis0,
            fs=self.porosity_subsol.fs.collapse(),
            name="Phis0")
        self.add_foi(
            expr=1. - self.Phis0,
            fs=self.porosity_subsol.fs.collapse(),
            name="Phif0")

        if (self.porosity_known == "phis"): self.add_foi(
            expr=self.Phis,
            fs=self.porosity_subsol.fs.collapse(),
            name="Phis")
        self.add_foi(
            expr=self.kinematics.J - self.Phis,
            fs=self.porosity_subsol.fs.collapse(),
            name="Phif")

        self.add_foi(
            expr=self.phis,
            fs=self.porosity_subsol.fs.collapse(),
            name="phis")
        self.add_foi(
            expr=1. - self.phis,
            fs=self.porosity_subsol.fs.collapse(),
            name="phif")



    def add_Wskel_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.WskelPoroOperator(
            kinematics=self.kinematics,
            U_test=self.displacement_subsol.dsubtest,
            Phis0=self.Phis0,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)



    def add_Wskel_operators(self,
            skel_behaviors):

        for skel_behavior in skel_behaviors:
            operator = self.add_Wskel_operator(
                material_parameters=skel_behavior["parameters"],
                material_scaling=skel_behavior["scaling"],
                subdomain_id=skel_behavior.get("subdomain_id", None))
            suffix = "_"+skel_behavior["suffix"] if "suffix" in skel_behavior else ""
            self.add_foi(expr=operator.material.Sigma, fs=self.mfoi_fs, name="Sigma_skel"+suffix)
            self.add_foi(expr=operator.material.sigma, fs=self.mfoi_fs, name="sigma_skel"+suffix)



    def add_Wbulk_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.WbulkPoroOperator(
            kinematics=self.kinematics,
            U_test=self.displacement_subsol.dsubtest,
            Phis0=self.Phis0,
            Phis=self.Phis,
            unknown_porosity_test=self.porosity_subsol.dsubtest,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)



    def add_Wbulk_operators(self,
            bulk_behaviors):

        for bulk_behavior in bulk_behaviors:
            operator = self.add_Wbulk_operator(
                material_parameters=bulk_behavior["parameters"],
                material_scaling=bulk_behavior["scaling"],
                subdomain_id=bulk_behavior.get("subdomain_id", None))
            suffix = "_"+bulk_behavior["suffix"] if "suffix" in bulk_behavior else ""
            self.add_foi(expr=operator.material.dWbulkdPhis, fs=self.sfoi_fs, name="dWbulkdPhis"+suffix)
            self.add_foi(expr=operator.material.dWbulkdPhis * self.kinematics.J * self.kinematics.C_inv, fs=self.mfoi_fs, name="Sigma_bulk"+suffix)
            self.add_foi(expr=operator.material.dWbulkdPhis * self.kinematics.I, fs=self.mfoi_fs, name="sigma_bulk"+suffix)



    def add_Wpore_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.WporePoroOperator(
            kinematics=self.kinematics,
            Phis0=self.Phis0,
            Phis=self.Phis,
            unknown_porosity_test=self.porosity_subsol.dsubtest,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)



    def add_Wpore_operators(self,
            pore_behaviors):

        for pore_behavior in pore_behaviors:
            self.add_Wpore_operator(
                material_parameters=pore_behavior["parameters"],
                material_scaling=pore_behavior["scaling"],
                subdomain_id=pore_behavior.get("subdomain_id", None))



    def add_pf_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.PfPoroOperator(
            unknown_porosity_test=self.porosity_subsol.dsubtest,
            **kwargs)
        self.add_operator(
            operator=operator,
            k_step=k_step)
        self.add_foi(expr=operator.pf, fs=self.sfoi_fs, name="pf")



    def add_pressure_balancing_gravity_loading_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.PressureBalancingGravityLoadingOperator(
            X=self.X,
            x0=self.deformed_center_of_mass_subsol.subfunc,
            x0_test=self.deformed_center_of_mass_subsol.dsubtest,
            lmbda=self.lmbda_subsol.subfunc,
            lmbda_test=self.lmbda_subsol.dsubtest,
            mu=self.mu_subsol.subfunc,
            mu_test=self.mu_subsol.dsubtest,
            p = self.pressure_balancing_gravity_subsol.subfunc,
            p_test = self.pressure_balancing_gravity_subsol.dsubtest,
            gamma = self.gamma_subsol.subfunc,
            gamma_test = self.gamma_subsol.dsubtest,
            kinematics=self.kinematics,
            U=self.displacement_subsol.subfunc,
            U_test=self.displacement_subsol.dsubtest,
            Phis=self.Phis,
            Phis0=self.Phis0,
            N=self.mesh_normals,
            **kwargs)
        return self.add_operator(operator=operator, k_step=k_step)



    def add_global_porosity_qois(self):

        self.add_qoi(
            name="Phis0",
            expr=self.Phis0 * self.dV)

        self.add_qoi(
            name="Phif0",
            expr=(1. - self.Phis0) * self.dV)

        self.add_qoi(
            name="Phis",
            expr=self.Phis * self.dV)

        self.add_qoi(
            name="Phif",
            expr=(self.kinematics.J - self.Phis) * self.dV)
            
        self.add_qoi(
            name="phis",
            expr=self.phis * self.dV)
            
        self.add_qoi(
            name="phif",
            expr=(1. - self.phis) * self.dV)



    def add_global_stress_qois(self,
            stress_type="cauchy"):

        if (stress_type in ("Cauchy", "cauchy", "sigma")):
            basename = "s_"
            stress = "sigma"
        elif (stress_type in ("Piola", "piola", "PK2", "Sigma")):
            basename = "S_"
            stress = "Sigma"
        elif (stress_type in ("Boussinesq", "boussinesq", "PK1", "P")):
            assert (0), "ToDo. Aborting."

        compnames = ["XX"]
        comps     = [(0,0)]
        if (self.dim >= 2):
            compnames += ["YY"]
            comps     += [(1,1)]
            if (self.dim >= 3):
                compnames += ["ZZ"]
                comps     += [(2,2)]
            compnames += ["XY"]
            comps     += [(0,1)]
            if (self.dim >= 3):
                compnames += ["YZ"]
                comps     += [(1,2)]
                compnames += ["ZX"]
                comps     += [(2,0)]
        for compname, comp in zip(compnames, comps):
            if (stress == "Sigma"):
                self.add_qoi(
                    name=basename+"skel_"+compname,
                    expr=sum([getattr(operator.material, stress)[comp]*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))]))
                self.add_qoi(
                    name=basename+"bulk_"+compname,
                    expr=sum([getattr(operator.material, "dWbulkdPhis")*self.kinematics.J*self.kinematics.C_inv[comp]*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, "dWbulkdPhis"))]))
                self.add_qoi(
                    name=basename+"tot_"+compname,
                    expr=sum([getattr(operator.material, stress)[comp]*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))]))+sum([getattr(operator.material, "dWbulkdPhis")[comp]*self.kinematics.J*self.kinematics.C_inv*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, "dWbulkdPhis"))])
            elif (stress == "sigma"):
                self.add_qoi(
                    name=basename+"skel_"+compname,
                    expr=sum([getattr(operator.material, stress)[comp]*self.kinematics.J*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))]))
                self.add_qoi(
                    name=basename+"bulk_"+compname,
                    expr=sum([getattr(operator.material, "dWbulkdPhis")*self.kinematics.I[comp]*self.kinematics.J*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, "dWbulkdPhis"))]))
                self.add_qoi(
                    name=basename+"tot_"+compname,
                    expr=sum([getattr(operator.material, stress)[comp]*self.kinematics.J*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, stress))])+sum([getattr(operator.material, "dWbulkdPhis")*self.kinematics.I[comp]*self.kinematics.J*operator.measure for operator in self.operators if (hasattr(operator, "material") and hasattr(operator.material, "dWbulkdPhis"))]))



    def add_global_fluid_pressure_qoi(self):

        # for operator in self.operators:
        #     print(type(operator))
        #     print(hasattr(operator, "pf"))

        # for step in self.steps:
        #     print(step)
        #     for operator in step.operators:
        #         print(type(operator))
        #         print(hasattr(operator, "pf"))

        self.add_qoi(
            name="pf",
            expr=sum([operator.pf*operator.measure for step in self.steps for operator in step.operators if hasattr(operator, "pf")]))
