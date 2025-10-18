#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2020-2024                                    ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_mech as dmech
from .Problem_Hyperelasticity import HyperelasticityProblem
from .Operator_DarcyFlow import DarcyFlowOperator,PfFieldOperator
from .Operator_WbulkPoroFlow import WbulkPoroFlowOperator
from .Operator_Poro_Wskel import WskelPoroOperator
################################################################################

class MicroPoroDarcyProblem(HyperelasticityProblem):



    def __init__(self,
            w_solid_incompressibility=False,
            mesh=None,
            mesh_bbox=None,
            vertices=None,
            domains_mf=None,
            boundaries_mf=None,
            points_mf=None,
            displacement_perturbation_degree=None,
            solid_pressure_degree=None,
            quadrature_degree=None,
            foi_degree=0,
            solid_behavior=None,
            bcs="kubc",
            ####New parameters#########
            ##########################
            ): # "kubc" or "pbc"

        HyperelasticityProblem.__init__(self)

                ##################################  
        ### Poro Materials Parameters            ###
        mat_params = {
                "alpha":0.16,
                "gamma":0.5,
                "c1":0.2,
                "c2":0.4,
                "kappa":1e2,
                "eta":1e-5}
        
        mat_params={
        "skel": {"parameters": mat_params, "scaling": "no"},
        "bulk": {"parameters": mat_params, "scaling": "no"},
        "pore": {"parameters": mat_params, "scaling": "no"}
            }
        
        skel_behavior=mat_params["skel"]
        bulk_behavior=mat_params["bulk"]
        pore_behavior=mat_params["pore"]
        
        
        porosity_params={
            "type": "constant",  # can be "constant", "function_constant", or "random"
            "val": 0.5
        }
        poro_type = porosity_params.get("type", "constant")
        poro_val = porosity_params.get("val", 0.5)
        porosity_fun = None
        poro_fs = dolfin.FunctionSpace(mesh, 'DG', 0)
        porosity_fun = dolfin.Function(poro_fs)
        porosity_fun.vector()[:] = poro_val
        poro_val = None

        define_facet_normals=True
        displacement_degree=1
        porosity_degree=None
        porosity_init_val=poro_val
        porosity_init_fun=porosity_fun
        #skel_behavior=None
        skel_behaviors=[]
        #bulk_behavior=None
        bulk_behaviors=[]
        #pore_behavior=None
        pore_behaviors=[]
        gradient_operators=None
        ##########################################      
        ############### Poro Materials Operators ################

        self.w_solid_incompressibility = w_solid_incompressibility
        self.vertices = vertices
        if (mesh is not None):
            self.set_mesh(
                mesh=mesh,
                define_spatial_coordinates=1,
                define_facet_normals=1,
                compute_bbox=(mesh_bbox is None))
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
                displacement_perturbation_degree=displacement_perturbation_degree,
                solid_pressure_degree=solid_pressure_degree,
                porosity_degree=porosity_degree,
                porosity_init_val=porosity_init_val,
                porosity_init_fun=porosity_init_fun)
            
            self.add_scalar_subsol(
                name="pressure",
                family="CG",
                degree=1)
            self.set_solution_finite_element()
            if (bcs == "pbc"):
                periodic_sd = dmech.PeriodicSubDomain(self.dim, self.mesh_bbox, self.vertices)
                #V_pressure = dolfin.FunctionSpace(mesh, "CG", 1, constrained_domain=periodic_sd)
                self.set_solution_function_space(constrained_domain=periodic_sd)
            else:
                self.set_solution_function_space()
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

            # self.add_elasticity_operator(
            #     solid_behavior_model=solid_behavior["model"],
            #     solid_behavior_parameters=solid_behavior["parameters"])
            # if (self.w_solid_incompressibility):
            #     self.add_hydrostatic_pressure_operator()
            #     self.add_incompressibility_operator()


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
            #self.add_pf_operator( pf_ini=0.2, pf_fin=0.2,
            #   )
            self.add_pf_operator(
               )
            # self.add_Darcy_operator(
            #      K_l=dolfin.Constant(1),
            #      rho_l=dolfin.Constant(1),
            #      Theta_in=dolfin.Constant(0.0),
            #      Theta_out=dolfin.Constant(0.0))


            # self.add_Darcy_operator(kinematics=self.kinematics,
            #     K_l=dolfin.Constant(1.0) * dolfin.Identity(2),
            #     rho_l=dolfin.Constant(1),
            #     Theta_in=dolfin.Constant(1000000000000.0),
            #     Theta_out=dolfin.Constant(1000.0),
            #     subdomain_id=None,    # where grad(p)·grad(p) is integrated
            #     inlet_id=3,
            #     outlet_id=4)





            ##################################

            # self.add_macroscopic_stretch_symmetry_operator()
            self.add_macroscopic_stretch_symmetry_penalty_operator(pen_val=1e6)

            # self.add_deformed_total_volume_operator()
            # self.add_deformed_solid_volume_operator()
            # self.add_deformed_fluid_volume_operator()

            if (bcs == "kubc"):
                self.add_kubc()
            elif (bcs == "pbc"):
                pinpoint_sd = dmech.PinpointSubDomain(coords=mesh.coordinates()[-1], tol=1e-3)
                self.add_constraint(
                    V=self.get_displacement_perturbation_function_space(), 
                    val=[0.]*self.dim,
                    sub_domain=pinpoint_sd,
                    method='pointwise')
                

################################################################################
# porosity functions ################################################################################
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

        operator = WskelPoroOperator(
            kinematics=self.kinematics,
            U=self.get_displacement_perturbation_subsol().subfunc,
            U_test=self.get_displacement_perturbation_subsol().dsubtest,
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

        operator = WbulkPoroFlowOperator(
            kinematics=self.kinematics,
            U=self.get_displacement_perturbation_subsol().subfunc,
            U_test=self.get_displacement_perturbation_subsol().dsubtest,
            Phis0=self.Phis0,
            Phis=self.get_porosity_subsol().subfunc,
            Phis_test=self.get_porosity_subsol().dsubtest,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id),
            pressure=self.get_subsol("pressure").subfunc
            )
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



    def add_pf_operator(self,
            k_step=None,
            **kwargs):
        
        operator = PfFieldOperator(pressure= self.get_subsol("pressure").subfunc,
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
    #         Phis_test=self.get_porosity_subsol().dsubtest,measure= self.get_subdomain_measure(None),
    #         **kwargs)
    #     self.add_operator(
    #         operator=operator,
    #         k_step=k_step)
    #     self.add_foi(expr=operator.pf, fs=self.sfoi_fs, name="pf")



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
    
    def add_Darcy_operator(self,
        kinematics,
        K_l,
        rho_l,
        Theta_in,
        Theta_out,
        subdomain_id,
        inlet_id,
        outlet_id,
        k_step=None):
    
        p      = self.get_subsol("pressure").subfunc
        p_test = self.get_subsol("pressure").dsubtest

        dx      = self.get_subdomain_measure(subdomain_id)      # e.g., dx or dx(subdomain_id)
        dx_in   = self.get_subdomain_measure(inlet_id)          # dx(inlet_id) for source
        dx_out  = self.get_subdomain_measure(outlet_id)         # dx(outlet_id) for sink



    # -----------------------------------------


        operator = DarcyFlowOperator(
            kinematics,
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
        self.add_foi(expr=operator.K_l, fs=self.mfoi_fs, name="K_l_ref", update_type="project")
        self.add_foi(expr=operator.k_l, fs=self.mfoi_fs, name="k_l_curr", update_type="project")

        return self.add_operator(operator=operator)
    
    def add_pressure_field(self):
        p = self.get_subsol("pressure").subfunc
        fs = self.get_subsol_function_space("pressure")
        self.add_foi(expr=p, fs=fs, name="pressure")
    
################################################
    #Material definitions
        
    
####################################################################################        

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




    def get_deformed_total_volume_name(self):

        return "v"

    def add_deformed_total_volume_subsol(self):

        self.add_scalar_subsol(
            name=self.get_deformed_total_volume_name(),
            family="R",
            degree=0,
            init_val=self.V0)

    def get_deformed_total_volume_subsol(self):

        return self.get_subsol(self.get_deformed_total_volume_name())



    def get_deformed_solid_volume_name(self):

        return "v_s"

    def add_deformed_solid_volume_subsol(self):

        self.add_scalar_subsol(
            name=self.get_deformed_solid_volume_name(),
            family="R",
            degree=0,
            init_val=self.mesh_V0)

    def get_deformed_solid_volume_subsol(self):

        return self.get_subsol(self.get_deformed_solid_volume_name())



    def get_deformed_fluid_volume_name(self):

        return "v_f"

    def add_deformed_fluid_volume_subsol(self):

        self.add_scalar_subsol(
            name=self.get_deformed_fluid_volume_name(),
            family="R",
            degree=0,
            init_val=self.Vf0)

    def get_deformed_fluid_volume_subsol(self):

        return self.get_subsol(self.get_deformed_fluid_volume_name())
    

    def get_surface_area_name(self):

        return "S_area"

    def add_surface_area_subsol(self,
            degree=0,
            init_val=None):
            
        self.add_scalar_subsol(
            name=self.get_surface_area_name(),
            family="R",
            degree=degree,
            init_val=init_val)


    def get_surface_area_subsol(self):

        return self.get_subsol(self.get_surface_area_name())

    def get_surface_area_space(self):

        return self.get_subsol_function_space(name=self.get_surface_area_name())




    def set_subsols(self,
            displacement_perturbation_degree=None,
            solid_pressure_degree=None,
            porosity_degree=None,
            porosity_init_val=None,
            porosity_init_fun=None):

        self.add_macroscopic_stretch_subsol(
            symmetry=None) # MG20220425: True does not work, cf. https://fenicsproject.discourse.group/t/writing-symmetric-tensor-function-fails/1136/2 & https://bitbucket.org/fenics-project/dolfin/issues/1065/cannot-store-symmetric-tensor-values

        self.add_displacement_perturbation_subsol(
            degree=displacement_perturbation_degree)
        
        if (self.w_solid_incompressibility):
            if (solid_pressure_degree is None):
                solid_pressure_degree = displacement_perturbation_degree-1
            self.add_pressure_subsol(
                degree=solid_pressure_degree)

        # self.add_macroscopic_stress_lagrange_multiplier_subsol()

        # self.add_deformed_total_volume_subsol()
        # self.add_deformed_solid_volume_subsol()
        # self.add_deformed_fluid_volume_subsol()

        if (porosity_degree is None):
            porosity_degree = displacement_perturbation_degree - 1
        self.add_surface_area_subsol()

        self.add_porosity_subsol(
            degree=porosity_degree,
            init_val=porosity_init_val,
            init_fun=porosity_init_fun)

        ####
        # New porosity subsol


        ####

    def set_kinematics(self):

        self.kinematics = dmech.Kinematics(
            U=self.U_tot,
            U_old=self.U_tot_old)

        self.add_foi(expr=self.kinematics.F, fs=self.mfoi_fs, name="F_tot", update_type="project")
        self.add_foi(expr=self.kinematics.J, fs=self.sfoi_fs, name="J_tot", update_type="project")
        self.add_foi(expr=self.kinematics.C, fs=self.mfoi_fs, name="C_tot", update_type="project")
        self.add_foi(expr=self.kinematics.E, fs=self.mfoi_fs, name="E_tot", update_type="project")



    def add_elasticity_operator(self,
            solid_behavior_model,
            solid_behavior_parameters):

        operator = dmech.HyperElasticityOperator(
            U=self.get_displacement_perturbation_subsol().subfunc,
            U_test=self.get_displacement_perturbation_subsol().dsubtest,
            kinematics=self.kinematics,
            material_model=solid_behavior_model,
            material_parameters=solid_behavior_parameters,
            measure=self.dV,
            formulation="ener")
        self.add_foi(expr=operator.material.Sigma, fs=self.mfoi_fs, name="Sigma", update_type="project")
        self.add_foi(expr=operator.material.sigma, fs=self.mfoi_fs, name="sigma", update_type="project")

        return self.add_operator(operator)



    def add_macroscopic_stretch_symmetry_penalty_operator(self,
            **kwargs):

        operator = dmech.MacroscopicStretchSymmetryPenaltyOperator(
            U_bar=self.get_macroscopic_stretch_subsol().subfunc,
            sol=self.sol_func,
            sol_test=self.dsol_test,
            measure=self.dV,
            **kwargs)
        return self.add_operator(operator)



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



    def add_surface_pressure_loading_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.SurfacePressureLoadingOperator(
            U_test=self.get_displacement_perturbation_subsol().dsubtest,
            kinematics=self.kinematics,
            N=self.mesh_normals,
            **kwargs)
        return self.add_operator(operator=operator, k_step=k_step)
    

    def add_surface_tension_loading_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.SurfaceTensionLoadingOperator(
            kinematics=self.kinematics,
            N=self.mesh_normals,
            U_test=self.U_tot_test,
            **kwargs)
        return self.add_operator(operator=operator, k_step=k_step)

    def add_deformed_total_volume_operator(self,
            k_step=None):

        operator = dmech.DeformedTotalVolumeOperator(
            v=self.get_deformed_total_volume_subsol().subfunc,
            v_test=self.get_deformed_total_volume_subsol().dsubtest,
            U_bar=self.get_macroscopic_stretch_subsol().subfunc,
            V0=self.V0,
            measure=self.dV)
        self.add_operator(operator=operator, k_step=k_step)



    def add_deformed_solid_volume_operator(self,
            k_step=None):

        operator = dmech.DeformedSolidVolumeOperator(
            vs=self.get_deformed_solid_volume_subsol().subfunc,
            vs_test=self.get_deformed_solid_volume_subsol().dsubtest,
            J=self.kinematics.J,
            Vs0=self.mesh_V0,
            measure=self.dV)
        self.add_operator(operator=operator, k_step=k_step)



    def add_deformed_fluid_volume_operator(self,
            k_step=None):

        operator = dmech.DeformedFluidVolumeOperator(
            vf=self.get_deformed_fluid_volume_subsol().subfunc,
            vf_test=self.get_deformed_fluid_volume_subsol().dsubtest,
            kinematics=self.kinematics,
            N=self.mesh_normals,
            dS=self.dS,
            U_tot=self.U_tot,
            X=self.X,
            measure=self.dV)
        self.add_operator(operator=operator, k_step=k_step)



    def add_surface_area_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.DeformedSurfaceAreaOperator(
            S_area = self.get_surface_area_subsol().subfunc,
            S_area_test = self.get_surface_area_subsol().dsubtest,
            kinematics=self.kinematics,
            N=self.mesh_normals,
            **kwargs)
        return self.add_operator(operator=operator, k_step=k_step)


    def add_kubc(self,
            xmin_id=1, xmax_id=2,
            ymin_id=3, ymax_id=4,
            zmin_id=5, zmax_id=6):

        self.add_constraint(
            V=self.get_displacement_perturbation_function_space().sub(0),
            sub_domains=self.boundaries,
            sub_domain_id=xmin_id,
            val=0.)
        self.add_constraint(
            V=self.get_displacement_perturbation_function_space().sub(0),
            sub_domains=self.boundaries,
            sub_domain_id=xmax_id,
            val=0.)
        self.add_constraint(
            V=self.get_displacement_perturbation_function_space().sub(1),
            sub_domains=self.boundaries,
            sub_domain_id=ymin_id,
            val=0.)
        self.add_constraint(
            V=self.get_displacement_perturbation_function_space().sub(1),
            sub_domains=self.boundaries,
            sub_domain_id=ymax_id,
            val=0.)
        if (self.dim==3):
            self.add_constraint(
                V=self.get_displacement_perturbation_function_space().sub(2),
                sub_domains=self.boundaries,
                sub_domain_id=zmin_id,
                val=0.)
            self.add_constraint(
                V=self.get_displacement_perturbation_function_space().sub(2),
                sub_domains=self.boundaries,
                sub_domain_id=zmax_id,
                val=0.)



    def add_deformed_solid_volume_qoi(self):

        self.add_qoi(
            name="vs",
            expr=self.kinematics.J * self.dV)



    def add_deformed_fluid_volume_qoi(self):

        U_bar = self.get_macroscopic_stretch_subsol().subfunc
        I_bar = dolfin.Identity(self.dim)
        F_bar = I_bar + U_bar
        J_bar = dolfin.det(F_bar)
        v = J_bar * self.V0

        self.add_qoi(
            name="vf",
            expr=(v/self.Vs0 - self.kinematics.J) * self.dV)



    def add_deformed_volume_qoi(self):

        U_bar = self.get_macroscopic_stretch_subsol().subfunc
        I_bar = dolfin.Identity(self.dim)
        F_bar = I_bar + U_bar
        J_bar = dolfin.det(F_bar)
        v = J_bar * self.V0

        self.add_qoi(
            name="v",
            expr=(v/self.Vs0) * self.dV)



    def add_macroscopic_tensor_qois(self,
            basename,
            get_subsol,
            symmetric=False):

        self.add_qoi(
            name=basename+"_XX",
            expr=get_subsol().subfunc[0,0],
            point=self.mesh.coordinates()[0],
            update_type="direct")
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"_YY",
                expr=get_subsol().subfunc[1,1],
                point=self.mesh.coordinates()[0],
                update_type="direct")
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"_ZZ",
                    expr=get_subsol().subfunc[2,2],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"_XY",
                expr=get_subsol().subfunc[0,1],
                point=self.mesh.coordinates()[0],
                update_type="direct")
            if not (symmetric): self.add_qoi(
                name=basename+"_YX",
                expr=get_subsol().subfunc[1,0],
                point=self.mesh.coordinates()[0],
                update_type="direct")
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"_YZ",
                    expr=get_subsol().subfunc[1,2],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")
                if not (symmetric): self.add_qoi(
                    name=basename+"_ZY",
                    expr=get_subsol().subfunc[2,1],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")
                self.add_qoi(
                    name=basename+"_ZX",
                    expr=get_subsol().subfunc[2,0],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")
                if not (symmetric): self.add_qoi(
                    name=basename+"_XZ",
                    expr=get_subsol().subfunc[0,2],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")



    def add_macroscopic_stretch_qois(self):

        self.add_macroscopic_tensor_qois(
            basename="U_bar",
            get_subsol=self.get_macroscopic_stretch_subsol)



    def add_macroscopic_solid_stress_qois(self,
            symmetric=False):

        for operator in self.operators: # MG20221110: Warning! Only works if there is a single operator with a material law!!
            if hasattr(operator, "material"):
                material = operator.material
                break

        U_bar = self.get_macroscopic_stretch_subsol().subfunc
        I_bar = dolfin.Identity(self.dim)
        F_bar = I_bar + U_bar
        J_bar = dolfin.det(F_bar)
        v = J_bar * self.V0

        self.add_qoi(
            name="sigma_s_bar_XX",
            expr=(material.sigma[0,0] * self.kinematics.J)/v * self.dV)
        if (self.dim >= 2):
            self.add_qoi(
                name="sigma_s_bar_YY",
                expr=(material.sigma[1,1] * self.kinematics.J)/v * self.dV)
            if (self.dim >= 3):
                self.add_qoi(
                    name="sigma_s_bar_ZZ",
                    expr=(material.sigma[2,2] * self.kinematics.J )/v * self.dV)
        if (self.dim >= 2):
            self.add_qoi(
                name="sigma_s_bar_XY",
                expr=(material.sigma[0,1] * self.kinematics.J)/v * self.dV)
            if not (symmetric): self.add_qoi(
                name="sigma_s_bar_YX",
                expr=(material.sigma[1,0] * self.kinematics.J)/v * self.dV)
            if (self.dim >= 3):
                self.add_qoi(
                    name="sigma_s_bar_YZ",
                    expr=(material.sigma[1,2] * self.kinematics.J)/v * self.dV)
                if not (symmetric): self.add_qoi(
                    name="sigma_s_bar_ZY",
                    expr=(material.sigma[2,1] * self.kinematics.J)/v * self.dV)
                self.add_qoi(
                    name="sigma_s_bar_ZX",
                    expr=(material.sigma[2,0] * self.kinematics.J)/v * self.dV)
                if not (symmetric): self.add_qoi(
                    name="sigma_s_bar_XZ",
                    expr=(material.sigma[0,2] * self.kinematics.J)/v * self.dV)



    def add_macroscopic_solid_hydrostatic_pressure_qoi(self):

        for operator in self.operators: # MG20221110: Warning! Only works if there is a single operator with a material law!!
            if hasattr(operator, "material"):
                material = operator.material
                break

        U_bar = self.get_macroscopic_stretch_subsol().subfunc
        I_bar = dolfin.Identity(self.dim)
        F_bar = I_bar + U_bar
        J_bar = dolfin.det(F_bar)
        v = J_bar * self.V0

        self.add_qoi(
            name="p_hydro",
            expr=(material.p_hydro * self.kinematics.J)/v * self.dV)



    def add_fluid_pressure_qoi(self):
        expr_lst = []
        for i in range(len(self.steps)):

            for operator in self.steps[i].operators: 
                if hasattr(operator, "tv_pf"):
                    tv_pf = operator.tv_pf
                    break
            expr_lst.append((tv_pf.val)/self.Vs0 * self.dV)

        self.add_qoi(
            name="p_f",
            expr_lst=expr_lst)
            # expr=(tv_pf.val)/self.Vs0 * self.dV)



    def add_macroscopic_stress_qois(self,
            symmetric=False):

        for operator in self.operators: # MG20221110: Warning! Only works if there is a single operator with a material law!!
            if hasattr(operator, "material"):
                material = operator.material
                break

        for operator in self.steps[0].operators: # MG20231124: Warning! Only works if there is a single step!!
            if hasattr(operator, "tv_pf"):
                tv_pf = operator.tv_pf
                break

        U_bar = self.get_macroscopic_stretch_subsol().subfunc
        I_bar = dolfin.Identity(self.dim)
        F_bar = I_bar + U_bar
        J_bar = dolfin.det(F_bar)
        v = J_bar * self.V0

        self.add_qoi(
            name="sigma_bar_XX",
            expr=(material.sigma[0,0] * self.kinematics.J - (v/self.Vs0 - self.kinematics.J) * tv_pf.val)/v * self.dV)
        if (self.dim >= 2):
            self.add_qoi(
                name="sigma_bar_YY",
                expr=(material.sigma[1,1] * self.kinematics.J - (v/self.Vs0 - self.kinematics.J) * tv_pf.val)/v * self.dV)
            if (self.dim >= 3):
                self.add_qoi(
                    name="sigma_bar_ZZ",
                    expr=(material.sigma[2,2] * self.kinematics.J - (v/self.Vs0 - self.kinematics.J) * tv_pf.val)/v * self.dV)
        if (self.dim >= 2):
            self.add_qoi(
                name="sigma_bar_XY",
                expr=(material.sigma[0,1] * self.kinematics.J)/v * self.dV)
            if not (symmetric): self.add_qoi(
                name="sigma_bar_YX",
                expr=(material.sigma[1,0] * self.kinematics.J)/v * self.dV)
            if (self.dim >= 3):
                self.add_qoi(
                    name="sigma_bar_YZ",
                    expr=(material.sigma[1,2] * self.kinematics.J)/v * self.dV)
                if not (symmetric): self.add_qoi(
                    name="sigma_bar_ZY",
                    expr=(material.sigma[2,1] * self.kinematics.J)/v * self.dV)
                self.add_qoi(
                    name="sigma_bar_ZX",
                    expr=(material.sigma[2,0] * self.kinematics.J)/v * self.dV)
                if not (symmetric): self.add_qoi(
                    name="sigma_bar_XZ",
                    expr=(material.sigma[0,2] * self.kinematics.J)/v * self.dV)
        


    def add_interfacial_surface_qois(self):
            FmTN = dolfin.dot(dolfin.inv(self.kinematics.F).T, self.mesh_normals)
            T = dolfin.sqrt(dolfin.inner(FmTN, FmTN))
            expr= T * self.kinematics.J
            self.add_qoi(
                name="S_area",
                expr=expr*self.dS(0))




