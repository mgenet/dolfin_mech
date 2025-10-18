#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### Edited by Haotian Xiao, 2024-2027                                        ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_mech as dmech
from .Problem_Hyperelasticity import HyperelasticityProblem
from .Operator_DarcyFlow import DarcyFlowOperator
from .Operator_WbulkPoroFlow import WbulkPoroFlowOperator
from .Operator_Poro_Wskel import WskelPoroOperator
################################################################################

class MicroPoroFlowHyperelasticityProblem(HyperelasticityProblem):

    

    def __init__(self,
            mesh=None,
            define_facet_normals=False,
            domains_mf=None,
            mesh_bbox=None,
            vertices=None,
            boundaries_mf=None,
            points_mf=None,
            displacement_degree=1,
            porosity_degree=None,
            quadrature_degree=None,
            foi_degree=0,
            porosity_init_val=None,
            porosity_init_fun=None,
            skel_behavior=None,
            skel_behaviors=[],
            bulk_behavior=None,
            bulk_behaviors=[],
            pore_behavior=None,
            pore_behaviors=[],
            gradient_operators=None):
        
        HyperelasticityProblem.__init__(self)
        
        if (mesh is not None):
            self.set_mesh(
                mesh=mesh,
                define_facet_normals=define_facet_normals)
            
            self.X_0 = [0.]*self.dim
            for k_dim in range(self.dim):
                self.X_0[k_dim] = dolfin.assemble(self.X[k_dim] * self.dV)/self.mesh_V0
            self.X_0 = dolfin.Constant(self.X_0)
            if (mesh_bbox is not None):
                self.mesh_bbox = mesh_bbox
            d = [0]*self.dim
            for k_dim in range(self.dim):
                d[k_dim] = self.mesh_bbox[2*k_dim+1] - self.mesh_bbox[2*k_dim+0]

            self.V0 = numpy.prod(d) 
            self.Vs0 = self.mesh_V0
            self.Vf0 = self.V0 - self.Vs0

            self.set_measures(
                domains=domains_mf,
                boundaries=boundaries_mf,
                points=points_mf)
            self.set_subsols(
                displacement_degree=displacement_degree,
                porosity_degree=porosity_degree,
                porosity_init_val=porosity_init_val,
                porosity_init_fun=porosity_init_fun)
            
              #NEW: Add pressure field
            self.add_scalar_subsol(
                name="pressure",
                family="CG",
                degree=1)


            assert (porosity_init_val is None) or (porosity_init_fun is None)
            if gradient_operators=="direct":
                self.Phis0 = porosity_init_val if (porosity_init_fun is None) else porosity_init_fun ### AP2023 - defined here to initialise subsol center of gravity
                self.set_subsols_gradient_direct()
            elif gradient_operators=="inverse":
                self.set_subsols_gradient_inverse()
            self.mesh_bbox = mesh_bbox
            self.vertices = vertices
            self.set_solution_finite_element()
            periodic_sd = dmech.PeriodicSubDomain(2, self.mesh_bbox, self.vertices)
            self.set_solution_function_space(constrained_domain=periodic_sd)
            #self.set_solution_function_space()
            self.set_solution_functions()

            self.U_bar      = dolfin.dot(self.get_macroscopic_stretch_subsol().subfunc , self.X-self.X_0)
            self.U_bar_old  = dolfin.dot(self.get_macroscopic_stretch_subsol().func_old, self.X-self.X_0)
            self.U_bar_test = dolfin.dot(self.get_macroscopic_stretch_subsol().dsubtest, self.X-self.X_0)

            self.U_tot      = self.U_bar      + self.get_displacement_perturbation_subsol().subfunc
            self.U_tot_old  = self.U_bar_old  + self.get_displacement_perturbation_subsol().func_old
            self.U_tot_test = self.U_bar_test + self.get_displacement_perturbation_subsol().dsubtest


            self.set_quadrature_degree(
                quadrature_degree=quadrature_degree)

            self.set_foi_finite_elements_DG(
                degree=foi_degree)
            self.set_foi_function_spaces()


            self.add_foi(
                expr=self.U_bar,
                fs=self.get_displacement_perturbation_function_space().collapse(),
                name="U_bar",
                update_type="project")
            self.add_foi(
                expr=self.U_tot,
                fs=self.get_displacement_perturbation_function_space().collapse(),
                name="U_tot",
                update_type="project")


            self.set_kinematics()

            assert (porosity_init_val is None) or (porosity_init_fun is None)
            self.init_known_porosity(
                porosity_init_val=porosity_init_val,
                porosity_init_fun=porosity_init_fun)

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

            # self.add_Wbulk_flow_operator(
            # bulk_behaviors,
            # "no",  
            # subdomain_id=None)


            self.add_Wpore_operators(pore_behaviors)

            self.add_Darcy_operator(
                K_l=dolfin.Constant(1),
                rho_l=dolfin.Constant(1),
                Theta_0=dolfin.Constant(0.0))
            self.add_pf_operator()
            


    def get_porosity_name(self):
        return "Phis"



    def add_porosity_subsol(self,
            degree,
            init_val=None,
            init_fun=None):

        if (degree == 0):
            self.add_scalar_subsol(
                name=self.get_porosity_name(),
                family="DG",
                degree=0,
                init_val=init_val,
                init_fun=init_fun)
        else:
            self.add_scalar_subsol(
                name=self.get_porosity_name(),
                family="CG",
                degree=degree,
                init_val=init_val,
                init_fun=init_fun)



    def get_porosity_subsol(self):

        return self.get_subsol(self.get_porosity_name())

    def get_porosity_function_space(self):

        return self.get_subsol_function_space(name=self.get_porosity_name())
    


    def get_balanced_gravity_boundary_pressure_name(self):
            return "pressure_boundary_condition"

    def add_balanced_gravity_boundary_pressure_subsol(self):

        self.add_scalar_subsol(
            name=self.get_balanced_gravity_boundary_pressure_name(),
            family="CG",
            degree=1)
        
    def get_balanced_gravity_boundary_pressure_subsol(self):

        return self.get_subsol(self.get_balanced_gravity_boundary_pressure_name())

    def get_balanced_gravity_boundary_pressure_function_space(self):

        return self.get_subsol_function_space(name=self.get_balanced_gravity_boundary_pressure_name())
    


    def get_lbda_name(self):

        return "lbda"

    def add_lbda_subsol(self,
            init_val=None):

        self.add_vector_subsol(
            name=self.get_lbda_name(),
            family="R",
            degree=0,
            init_val=init_val)
    
    def get_lbda_subsol(self):

        return self.get_subsol(self.get_lbda_name())



    def get_mu_name(self):

        return "mu"
    
    def add_mu_subsol(self,
            init_val=None):

        self.add_vector_subsol(
            name=self.get_mu_name(),
            family="R",
            degree=0,
            init_val=init_val)
        
    def get_mu_subsol(self):

        return self.get_subsol(self.get_mu_name())
    

    
    def get_gamma_name(self):
        return "gamma"

    def add_gamma_subsol(self):

        self.add_scalar_subsol(
            name=self.get_gamma_name(),
            family="R",
            degree=0)
        
    def get_gamma_subsol(self):

        return self.get_subsol(self.get_gamma_name())

    def get_gamma_function_space(self):

        return self.get_subsol_function_space(name=self.get_gamma_name())
    


    def get_center_gravity_direct_problem_name(self):

        return "xg"
            
    def add_center_gravity_direct_problem_subsol(self,
            init_val=None):
        
        self.add_vector_subsol(
            name=self.get_center_gravity_direct_problem_name(),
            family="R",
            degree=0,
            init_val=self.get_center_gravity())

    def get_center_gravity_direct_problem_subsol(self):
        return self.get_subsol(self.get_center_gravity_direct_problem_name())



    def set_subsols(self,
            displacement_degree=1,
            porosity_degree=None,
            porosity_init_val=None,
            porosity_init_fun=None):
        #print(">>> USING subclass set_subsols <<<")
        #self.add_displacement_subsol(degree=displacement_degree)

        self.add_displacement_perturbation_subsol(
            degree=displacement_degree)

        if (porosity_degree is None):
            porosity_degree = displacement_degree - 1
        self.add_porosity_subsol(
            degree=porosity_degree,
            init_val=porosity_init_val,
            init_fun=porosity_init_fun)
        
        self.add_macroscopic_stretch_subsol(
            symmetry=None) 

      


    def set_subsols_gradient_direct(self):
            
                self.add_balanced_gravity_boundary_pressure_subsol()
                self.add_gamma_subsol()
                self.add_lbda_subsol()
                self.add_mu_subsol()
                self.add_center_gravity_direct_problem_subsol()



    def set_subsols_gradient_inverse(self):
        
            self.add_balanced_gravity_boundary_pressure_subsol()
            self.add_gamma_subsol()
            self.add_lbda_subsol()
            self.add_mu_subsol()



    def init_known_porosity(self,
            porosity_init_val,
            porosity_init_fun):

        if   (porosity_init_val   is not None):
            self.Phis0 = dolfin.Constant(porosity_init_val)
        elif (porosity_init_fun is not None):
            self.Phis0 = porosity_init_fun
        self.add_foi(
            expr=self.Phis0,
            fs=self.get_porosity_function_space().collapse(),
            name="Phis0")
        self.add_foi(
            expr=1 - self.Phis0,
            fs=self.get_porosity_function_space().collapse(),
            name="Phif0")
        self.add_foi(
            expr=self.kinematics.J - self.get_porosity_subsol().subfunc,
            fs=self.get_porosity_function_space().collapse(),
            name="Phif")
        self.add_foi(
            expr=self.get_porosity_subsol().subfunc/self.kinematics.J,
            fs=self.get_porosity_function_space().collapse(),
            name="phis")
        self.add_foi(
            expr=1.-self.get_porosity_subsol().subfunc/self.kinematics.J,
            fs=self.get_porosity_function_space().collapse(),
            name="phif")



    def add_Wskel_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.WskelPoroOperator(
            kinematics=self.kinematics,
            U_test=self.get_displacement_subsol().dsubtest,
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
            U_test=self.get_displacement_subsol().dsubtest,
            Phis0=self.Phis0,
            Phis=self.get_porosity_subsol().subfunc,
            Phis_test=self.get_porosity_subsol().dsubtest,
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

    # def add_Wbulk_flow_operator(self,
    #         material_parameters,
    #         material_scaling,
    #         subdomain_id=None):

       
    #     #print(material_parameters[0])
    #     operator = WbulkPoroFlowOperator(
    #         kinematics=self.kinematics,
    #         U_test=self.get_displacement_subsol().dsubtest,
    #         Phis0=self.Phis0,
    #         Phis=self.get_porosity_subsol().subfunc,
    #         Phis_test=self.get_porosity_subsol().dsubtest,
    #         material_parameters=material_parameters[0]["parameters"],
    #         material_scaling=material_scaling,
    #         measure=self.get_subdomain_measure(subdomain_id),
    #         pressure=self.get_subsol("pressure").subfunc)
        
    #     return self.add_operator(operator)



    def add_Wpore_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = dmech.WporePoroOperator(
            kinematics=self.kinematics,
            Phis0=self.Phis0,
            Phis=self.get_porosity_subsol().subfunc,
            Phis_test=self.get_porosity_subsol().dsubtest,
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
        
        operator = dmech.PfFieldOperator(pressure= self.get_subsol("pressure").subfunc,
            Phis_test=self.get_porosity_subsol().dsubtest, measure= self.get_subdomain_measure(None),
            **kwargs)
        self.add_operator(
            operator=operator,
            k_step=k_step)
        
        self.add_foi(expr=operator.pf, fs=self.sfoi_fs, name="pressure")
        

    # def add_pf_operator(self,
    #         k_step=None,
    #         **kwargs):

    #     operator = dmech.PfPoroOperator(
    #         Phis_test=self.get_porosity_subsol().dsubtest,
    #         **kwargs)
    #     self.add_operator(
    #         operator=operator,
    #         k_step=k_step)
    #     self.add_foi(expr=operator.pf, fs=self.sfoi_fs, name="pf")

    # def add_Darcy_operator(self,
    #         K_l,
    #         rho_l,
    #         Theta_0,
    #         subdomain_id=None,
    #         k_step=None):

    #     p       = self.get_subsol("pressure").subfunc
    #     p_test  = self.get_subsol("pressure").dsubtest
    #     measure = self.get_subdomain_measure(subdomain_id)

    #     operator = DarcyFlowOperator(
    #         p=p,
    #         p_test=p_test,
    #         K_l=K_l,
    #         rho_l=rho_l,
    #         Theta_0=Theta_0,
    #         measure=measure)

    #     return self.add_operator(operator=operator, k_step=k_step)

    def add_Darcy_operator(self,
        K_l,
        rho_l,
        Theta_in,
        Theta_out,
        subdomain_id=None,
        inlet_id=None,
        outlet_id=None,
        k_step=None):
    
        p      = self.get_subsol("pressure").subfunc
        p_test = self.get_subsol("pressure").dsubtest

        dx      = self.get_subdomain_measure(subdomain_id)      # e.g., dx or dx(subdomain_id)
        dx_in   = self.get_subdomain_measure(inlet_id)          # dx(inlet_id) for source
        dx_out  = self.get_subdomain_measure(outlet_id)         # dx(outlet_id) for sink

        operator = DarcyFlowOperator(
            p=p,
            p_test=p_test,
            K_l=K_l,
            rho_l=rho_l,
            Theta_in=Theta_in,
            Theta_out=Theta_out,
            dx=dx,
            dx_in=dx_in,
            dx_out=dx_out
        )

        return self.add_operator(operator=operator)


    def add_pressure_balancing_gravity0_loading_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.PressureBalancingGravity0LoadingOperator(
            x = self.x,
            x0 = self.get_center_gravity(),
            n = self.mesh_normals,
            u_test = self.get_displacement_subsol().dsubtest, 
            lbda = self.get_lbda_subsol().subfunc,
            lbda_test = self.get_lbda_subsol().dsubtest,
            p = self.get_balanced_gravity_boundary_pressure_subsol().subfunc,
            p_test = self.get_balanced_gravity_boundary_pressure_subsol().dsubtest,
            gamma = self.get_gamma_subsol().subfunc,
            gamma_test = self.get_gamma_subsol().dsubtest,
            mu = self.get_mu_subsol().subfunc,
            mu_test= self.get_mu_subsol().dsubtest,
            **kwargs)
        return self.add_operator(operator=operator, k_step=k_step)



    def add_pressure_balancing_gravity_loading_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.PressureBalancingGravityLoadingOperator(
            X=self.X,
            x0=self.get_center_gravity_direct_problem_subsol().subfunc,
            x0_test=self.get_center_gravity_direct_problem_subsol().dsubtest,
            lbda=self.get_lbda_subsol().subfunc,
            lbda_test=self.get_lbda_subsol().dsubtest,
            mu=self.get_mu_subsol().subfunc,
            mu_test=self.get_mu_subsol().dsubtest,
            p = self.get_balanced_gravity_boundary_pressure_subsol().subfunc,
            p_test = self.get_balanced_gravity_boundary_pressure_subsol().dsubtest,
            gamma = self.get_gamma_subsol().subfunc,
            gamma_test = self.get_gamma_subsol().dsubtest,
            kinematics=self.kinematics,
            U=self.get_displacement_subsol().subfunc,
            U_test=self.get_displacement_subsol().dsubtest,
            Phis0=self.Phis0,
            N=self.mesh_normals,
            **kwargs)
        return self.add_operator(operator=operator, k_step=k_step)



    def add_global_porosity_qois(self):

        self.add_qoi(
            name=self.get_porosity_name(),
            expr=self.get_porosity_subsol().subfunc * self.dV)

        self.add_qoi(
            name="Phif",
            expr=(self.kinematics.J - self.get_porosity_subsol().subfunc) * self.dV)
            
        self.add_qoi(
            name="phis",
            expr=(self.get_porosity_subsol().subfunc/self.kinematics.J) * self.dV)
            
        self.add_qoi(
            name="phif",
            expr=(1. - self.get_porosity_subsol().subfunc/self.kinematics.J) * self.dV)
        

    
    def get_center_gravity(self): 
        if "Inverse" in str(self):
            porosity = self.phis 
            position = self.x
        else:
            porosity=self.Phis0
            position=self.X
        center_gravity = numpy.empty(self.dim)
        for k_dim in range(self.dim):
            center_gravity[k_dim] = dolfin.assemble(porosity*position[k_dim]*self.dV)/dolfin.assemble(porosity*self.dV)
        return(center_gravity)



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


    def add_pressure_field(self):
        p = self.get_subsol("pressure").subfunc
        fs = self.get_subsol_function_space("pressure")
        self.add_foi(expr=p, fs=fs, name="pressure")


    def add_macroscopic_stretch_component_penalty_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.MacroscopicStretchComponentPenaltyOperator(
            U_bar=self.get_macroscopic_stretch_subsol().subfunc,
            U_bar_test=self.get_macroscopic_stretch_subsol().dsubtest,
            measure=self.dV,
            **kwargs)
        return self.add_operator(operator, k_step=k_step)



    def add_macroscopic_stress_component_constraint_operator(self,
            k_step=None,
            **kwargs):

        for operator in self.operators: # MG20221110: Warning! Only works if there is a single operator with a material law!!
            if hasattr(operator, "material"):
                material = operator.material
                break

        operator = dmech.MacroscopicStressComponentConstraintOperator(
            U_bar=self.get_macroscopic_stretch_subsol().subfunc,
            U_bar_test=self.get_macroscopic_stretch_subsol().dsubtest,
            kinematics=self.kinematics,
            material=material,
            V0=self.V0,
            Vs0=self.Vs0,
            measure=self.dV,
            N=self.mesh_normals,
            **kwargs)
        return self.add_operator(operator, k_step=k_step)

    def get_macroscopic_stretch_name(self):

        return "U_bar"

    def add_macroscopic_stretch_subsol(self,
            degree=0,
            symmetry=None,
            init_val=None):

        self.add_tensor_subsol(
            name=self.get_macroscopic_stretch_name(),
            family="R",
            degree=degree,
            symmetry=symmetry,
            init_val=init_val)

    def get_macroscopic_stretch_subsol(self):

        return self.get_subsol(self.get_macroscopic_stretch_name())

    def get_macroscopic_stretch_function_space(self):

        return self.get_subsol_function_space(name=self.get_macroscopic_stretch_name())
    
    def get_macroscopic_stretch_name(self):

        return "U_bar"
    
    def get_displacement_perturbation_name(self):

        return "U_tilde"

    def add_displacement_perturbation_subsol(self,
            degree):

        self.displacement_perturbation_degree = degree
        self.add_vector_subsol(
            name=self.get_displacement_perturbation_name(),
            family="CG",
            degree=self.displacement_perturbation_degree)

    def get_displacement_perturbation_subsol(self):

        return self.get_subsol(self.get_displacement_perturbation_name())

    def get_displacement_perturbation_function_space(self):

        return self.get_subsol_function_space(name=self.get_displacement_perturbation_name())
