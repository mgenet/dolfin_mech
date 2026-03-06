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

import operator
import dolfin
import numpy

import dolfin_mech as dmech
from .Problem                 import Problem
from .Problem_Hyperelasticity import HyperelasticityProblem
from .Operator_DarcyFlow import MicroDarcyFlowOperator,WskelPoroFlowOperator
from .Operator_ZeroMeanPressure import ZeroMeanPressureOperator
################################################################################
class _SigmaAggregatorMaterial:
    def __init__(self, problem):
        self._problem = problem

    @property
    def sigma(self):
        return self._problem.get_sigma_total()
    
class MicroPoroFlowHyperelasticityProblem(HyperelasticityProblem):



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
            porosity_known="Phis0",
            bcs="pbc",
            porosity_degree=None,
            porosity_init_val=None,
            porosity_init_fun=None,
            quadrature_degree=None,
            foi_degree=1,
            flow_params={},
            skel_behavior=None,
            skel_behaviors=[],
            bulk_behavior=None,
            bulk_behaviors=[],
            pore_behavior=None,
            pore_behaviors=[],
            w_pressure_balancing_gravity=0):

        Problem.__init__(self)

        self.w_solid_incompressibility = w_solid_incompressibility
        self.vertices = vertices


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

        assert (porosity_known in ("Phis0", "phis"))
        self.set_known_and_unknown_porosity(porosity_known)

        assert (porosity_init_val is None) or (porosity_init_fun is None)
        self.init_known_porosity(
            porosity_init_val=porosity_init_val,
            porosity_init_fun=porosity_init_fun)
        
        self.w_pressure_balancing_gravity = w_pressure_balancing_gravity

        self.set_subsols(
            displacement_perturbation_degree=displacement_perturbation_degree,
            solid_pressure_degree=solid_pressure_degree,
            porosity_degree=porosity_degree,
            porosity_init_val=porosity_init_val,
            porosity_init_fun=porosity_init_fun)
        self.set_solution_finite_element()
        if (bcs == "pbc"):
            periodic_sd = dmech.PeriodicSubDomain(self.dim, self.mesh_bbox, self.vertices)
            self.set_solution_function_space(constrained_domain=periodic_sd)
        else:
            self.set_solution_function_space()
        self.set_solution_functions()



        self.U_bar      = dolfin.dot(self.macroscopic_stretch_subsol.subfunc , self.X-self.X_0)
        self.U_bar_old  = dolfin.dot(self.macroscopic_stretch_subsol.func_old, self.X-self.X_0)
        self.U_bar_test = dolfin.dot(self.macroscopic_stretch_subsol.dsubtest, self.X-self.X_0)

        self.U_tot      = self.U_bar      + self.displacement_perturbation_subsol.subfunc
        self.U_tot_old  = self.U_bar_old  + self.displacement_perturbation_subsol.func_old
        self.U_tot_test = self.U_bar_test + self.displacement_perturbation_subsol.dsubtest

        self.set_quadrature_degree(
            quadrature_degree=quadrature_degree)

        self.set_foi_finite_elements_DG(
            degree=foi_degree)
        self.set_foi_function_spaces()

        self.add_foi(
            expr=self.U_bar,
            fs=self.vfoi_fs,
            name="U_bar",
            update_type="project")
        self.add_foi(
            expr=self.U_tot,
            fs=self.vfoi_fs,
            name="U_tot",
            update_type="project")

        self.set_kinematics()


        #self.set_porosity_fields()
        #self.add_local_porosity_fois()


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
        self.kappa_val = bulk_behaviors[0]["parameters"]["kappa"]   


        assert (pore_behavior is None) or (len(pore_behaviors)==0),\
            "Cannot provide both pore_behavior & pore_behaviors. Aborting."
        if (pore_behavior is not None):
            pore_behaviors = [pore_behavior]

        # self.add_macroscopic_stretch_symmetry_operator()
        self.add_macroscopic_stretch_symmetry_penalty_operator(pen_val=1e6)

        #self.add_Wpore_operators(pore_behaviors)

        self.add_pressure_liquid_perturbation_zero_mean_operator()





        if (bcs == "kubc"):
            self.add_kubc()
        elif (bcs == "pbc"):
            pinpoint_sd = dmech.PinpointSubDomain(coords=mesh.coordinates()[-1], tol=1e-3)
            self.add_constraint(
                V=self.displacement_perturbation_subsol.fs, 
                val=[0.]*self.dim,
                sub_domain=pinpoint_sd,
                method='pointwise')
        



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

  
    

        

    def add_macroscopic_stretch_subsol(self,
            degree=0,
            symmetry=None,
            init_val=None):

        self.macroscopic_stretch_subsol = self.add_tensor_subsol(
            name="U_bar",
            family="R",
            degree=degree,
            symmetry=symmetry,
            init_val=init_val)
        
    def add_macroscopic_unknown_porosity_gradient_subsol(self,degree=0,init_val=None):

        self.grad_unknown_porosity_bar_subsol = self.add_vector_subsol(
            name="grad_unknown_porosity_bar",
            family="R",
            degree=degree,
            init_val=init_val)
        
    def add_macroscopic_unknown_porosity_gradient_subsol(self,
        degree=0,
        init_val=None):


        self.grad_unknown_porosity_bar_subsol = self.add_vector_subsol(
            name="grad_unknown_porosity_bar",
            family="R",
            degree=degree,
            init_val=init_val)





    def add_displacement_perturbation_subsol(self,
            degree):

        self.displacement_perturbation_degree = degree
        self.displacement_perturbation_subsol = self.add_vector_subsol(
            name="U_tilde",
            family="CG",
            degree=self.displacement_perturbation_degree)



    def add_deformed_total_volume_subsol(self):

        self.deformed_total_volume_subsol = self.add_scalar_subsol(
            name="v",
            family="R",
            degree=0,
            init_val=self.V0)



    def add_deformed_solid_volume_subsol(self):

        self.deformed_solid_volume_subsol = self.add_scalar_subsol(
            name="v_s",
            family="R",
            degree=0,
            init_val=self.mesh_V0)



    def add_deformed_fluid_volume_subsol(self):

        self.deformed_fluid_volume_subsol = self.add_scalar_subsol(
            name="v_f",
            family="R",
            degree=0,
            init_val=self.Vf0)



    def add_surface_area_subsol(self,
            degree=0,
            init_val=None):
            
        self.surface_area_subsol = self.add_scalar_subsol(
            name="S_area",
            family="R",
            degree=degree,
            init_val=init_val)



    def set_subsols(self,
            displacement_perturbation_degree=None,
            solid_pressure_degree=None,
            liquid_pressure_perturbation_degree=None,
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
        self.add_surface_area_subsol()

        # if (porosity_degree is None):
        #     porosity_degree = displacement_perturbation_degree - 1
        # print("Adding porosity subsolution with degree =", porosity_degree)
        # self.add_porosity_subsol(
        #     degree=porosity_degree,
        #     init_val=porosity_init_val,
        #     init_fun=porosity_init_fun)
        if (self.w_pressure_balancing_gravity):
            self.add_pressure_balancing_gravity_subsol()
            self.add_gamma_subsol()
            self.add_lmbda_subsol()
            self.add_mu_subsol()
            self.add_deformed_center_of_mass_subsol()
        
        if (liquid_pressure_perturbation_degree is None):
            liquid_pressure_perturbation_degree = displacement_perturbation_degree -1
        self.add_pressure_liquid_perturbation_subsol(degree=liquid_pressure_perturbation_degree)
        self.add_pressure_liquid_perturbation_zero_mean_subsol()


        


            
        

    def get_pressure_liquid_name(self):
        return "p_l_perturbation"

    def add_pressure_liquid_perturbation_subsol(self, degree):
        self.pl_perturbation_subsol = self.add_scalar_subsol(
            name=self.get_pressure_liquid_name(),
            family="CG",
            degree=degree
        )
    def add_pressure_liquid_perturbation_zero_mean_subsol(self):
        self.lambda_pl_perturbation_zero_mean_subsol = self.add_scalar_subsol(
            name="lambda_"+self.get_pressure_liquid_name()+"_zero_mean",
            family="R",
            degree=0
        )




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
            fs=self.sfoi_fs,
            name="Phis0",
            update_type="project")
        self.add_foi(
            expr=1. - self.Phis0,
            fs=self.sfoi_fs,
            name="Phif0",
            update_type="project")

        if (self.porosity_known == "phis"): self.add_foi(
            expr=self.Phis,
            fs=self.sfoi_fs,
            name="Phis",
            update_type="project")
        self.add_foi(
            expr=self.kinematics.J - self.Phis,
            fs=self.sfoi_fs,
            name="Phif",
            update_type="project")

        self.add_foi(
            expr=self.phis,
            fs=self.sfoi_fs,
            name="phis",
            update_type="project")
        self.add_foi(
            expr=1. - self.phis,
            fs=self.sfoi_fs,
            name="phif",
            update_type="project")

    def set_kinematics(self):

        self.kinematics = dmech.Kinematics(
            U=self.U_tot,
            U_old=self.U_tot_old)

        self.add_foi(expr=self.kinematics.F, fs=self.mfoi_fs, name="F_tot", update_type="project")
        self.add_foi(expr=self.kinematics.J, fs=self.sfoi_fs, name="J_tot", update_type="project")
        self.add_foi(expr=self.kinematics.C, fs=self.mfoi_fs, name="C_tot", update_type="project")
        self.add_foi(expr=self.kinematics.E, fs=self.mfoi_fs, name="E_tot", update_type="project")

    def add_macroscopic_stretch_symmetry_penalty_operator(self,
            **kwargs):

        operator = dmech.MacroscopicStretchSymmetryPenaltyOperator(
            U_bar=self.macroscopic_stretch_subsol.subfunc,
            sol=self.sol_func,
            sol_test=self.dsol_test,
            measure=self.dV,
            **kwargs)
        return self.add_operator(operator)



    def add_macroscopic_stretch_component_penalty_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.MacroscopicStretchComponentPenaltyOperator(
            U_bar=self.macroscopic_stretch_subsol.subfunc,
            U_bar_test=self.macroscopic_stretch_subsol.dsubtest,
            measure=self.dV,
            **kwargs)
        return self.add_operator(operator, k_step=k_step)



    def add_macroscopic_stress_component_constraint_operator(self,
            k_step=None,
            **kwargs):

        # for operator in self.operators: # MG20221110: Warning! Only works if there is a single operator with a material law!!
        #     if hasattr(operator, "material"):
        #         material = operator.material
        #         break
        material = _SigmaAggregatorMaterial(self)

        operator = dmech.MacroscopicStressComponentConstraintOperator(
            U_bar=self.macroscopic_stretch_subsol.subfunc,
            U_bar_test=self.macroscopic_stretch_subsol.dsubtest,
            kinematics=self.kinematics,
            material=material,
            V0=self.V0,
            Vs0=self.Vs0,
            measure=self.dV,
            N=self.mesh_normals,
            **kwargs)
        return self.add_operator(operator, k_step=k_step)
    
    def add_pressure_liquid_perturbation_zero_mean_operator(self):
        p      = self.pl_perturbation_subsol.subfunc
        p_test = self.pl_perturbation_subsol.dsubtest


        lam      = self.lambda_pl_perturbation_zero_mean_subsol.subfunc
        lam_test = self.lambda_pl_perturbation_zero_mean_subsol.dsubtest

        operator = ZeroMeanPressureOperator(
            p=p,
            p_test=p_test,
            lam=lam,
            lam_test=lam_test,
            measure=self.dV
        )
        return self.add_operator(operator)




    def add_surface_pressure_loading_operator(self,
            k_step=None,
            **kwargs):

        operator = dmech.SurfacePressureLoadingOperator(
            U_test=self.displacement_perturbation_subsol.dsubtest,
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
            v=self.deformed_total_volume_subsol.subfunc,
            v_test=self.deformed_total_volume_subsol.dsubtest,
            U_bar=self.macroscopic_stretch_subsol.subfunc,
            V0=self.V0,
            measure=self.dV)
        self.add_operator(operator=operator, k_step=k_step)



    def add_deformed_solid_volume_operator(self,
            k_step=None):

        operator = dmech.DeformedSolidVolumeOperator(
            vs=self.deformed_solid_volume_subsol.subfunc,
            vs_test=self.deformed_solid_volume_subsol.dsubtest,
            J=self.kinematics.J,
            Vs0=self.mesh_V0,
            measure=self.dV)
        self.add_operator(operator=operator, k_step=k_step)



    def add_deformed_fluid_volume_operator(self,
            k_step=None):

        operator = dmech.DeformedFluidVolumeOperator(
            vf=self.deformed_fluid_volume_subsol.subfunc,
            vf_test=self.deformed_fluid_volume_subsol.dsubtest,
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
            S_area = self.surface_area_subsol.subfunc,
            S_area_test = self.surface_area_subsol.dsubtest,
            kinematics=self.kinematics,
            N=self.mesh_normals,
            **kwargs)
        return self.add_operator(operator=operator, k_step=k_step)



    def add_kubc(self,
            xmin_id=1, xmax_id=2,
            ymin_id=3, ymax_id=4,
            zmin_id=5, zmax_id=6):

        self.add_constraint(
            V=self.displacement_perturbation_subsol.fs.sub(0),
            sub_domains=self.boundaries,
            sub_domain_id=xmin_id,
            val=0.)
        self.add_constraint(
            V=self.displacement_perturbation_subsol.fs.sub(0),
            sub_domains=self.boundaries,
            sub_domain_id=xmax_id,
            val=0.)
        self.add_constraint(
            V=self.displacement_perturbation_subsol.fs.sub(1),
            sub_domains=self.boundaries,
            sub_domain_id=ymin_id,
            val=0.)
        self.add_constraint(
            V=self.displacement_perturbation_subsol.fs.sub(1),
            sub_domains=self.boundaries,
            sub_domain_id=ymax_id,
            val=0.)
        if (self.dim==3):
            self.add_constraint(
                V=self.displacement_perturbation_subsol.fs.sub(2),
                sub_domains=self.boundaries,
                sub_domain_id=zmin_id,
                val=0.)
            self.add_constraint(
                V=self.displacement_perturbation_subsol.fs.sub(2),
                sub_domains=self.boundaries,
                sub_domain_id=zmax_id,
                val=0.)



    def add_deformed_solid_volume_qoi(self):

        self.add_qoi(
            name="vs",
            expr=self.kinematics.J * self.dV)



    def add_deformed_fluid_volume_qoi(self):

        U_bar = self.macroscopic_stretch_subsol.subfunc
        I_bar = dolfin.Identity(self.dim)
        F_bar = I_bar + U_bar
        J_bar = dolfin.det(F_bar)
        v = J_bar * self.V0

        self.add_qoi(
            name="vf",
            expr=(v/self.Vs0 - self.kinematics.J) * self.dV)



    def add_deformed_volume_qoi(self):

        U_bar = self.macroscopic_stretch_subsol.subfunc
        I_bar = dolfin.Identity(self.dim)
        F_bar = I_bar + U_bar
        J_bar = dolfin.det(F_bar)
        v = J_bar * self.V0

        self.add_qoi(
            name="v",
            expr=(v/self.Vs0) * self.dV)



    def add_macroscopic_tensor_qois(self,
            basename,
            subsol,
            symmetric=False):

        self.add_qoi(
            name=basename+"_XX",
            expr=subsol.subfunc[0,0],
            point=self.mesh.coordinates()[0],
            update_type="direct")
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"_YY",
                expr=subsol.subfunc[1,1],
                point=self.mesh.coordinates()[0],
                update_type="direct")
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"_ZZ",
                    expr=subsol.subfunc[2,2],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"_XY",
                expr=subsol.subfunc[0,1],
                point=self.mesh.coordinates()[0],
                update_type="direct")
            if not (symmetric): self.add_qoi(
                name=basename+"_YX",
                expr=subsol.subfunc[1,0],
                point=self.mesh.coordinates()[0],
                update_type="direct")
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"_YZ",
                    expr=subsol.subfunc[1,2],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")
                if not (symmetric): self.add_qoi(
                    name=basename+"_ZY",
                    expr=subsol.subfunc[2,1],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")
                self.add_qoi(
                    name=basename+"_ZX",
                    expr=subsol.subfunc[2,0],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")
                if not (symmetric): self.add_qoi(
                    name=basename+"_XZ",
                    expr=subsol.subfunc[0,2],
                    point=self.mesh.coordinates()[0],
                    update_type="direct")



    def add_macroscopic_stretch_qois(self):

        self.add_macroscopic_tensor_qois(
            basename="U_bar",
            subsol=self.macroscopic_stretch_subsol)



    def add_macroscopic_solid_stress_qois(self,
            symmetric=False):

        # for operator in self.operators: # MG20221110: Warning! Only works if there is a single operator with a material law!!
        #     if hasattr(operator, "material"):
        #         material = operator.material
        #         break
        material = _SigmaAggregatorMaterial(self)

        U_bar = self.macroscopic_stretch_subsol.subfunc
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

        # for operator in self.operators: # MG20221110: Warning! Only works if there is a single operator with a material law!!
        #     if hasattr(operator, "material"):
        #         material = operator.material
        #         break
        material = _SigmaAggregatorMaterial(self)

        U_bar = self.macroscopic_stretch_subsol.subfunc
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

        # for operator in self.operators: # MG20221110: Warning! Only works if there is a single operator with a material law!!
        #     if hasattr(operator, "material"):
        #         material = operator.material
        #         break
        material = _SigmaAggregatorMaterial(self)

        # for operator in self.steps[0].operators: # MG20231124: Warning! Only works if there is a single step!!
        #     if hasattr(operator, "tv_pf"):
        #         tv_pf = operator.tv_pf
        #         break
        #tv_pf = None
        for step in self.steps:
            for operator in step.operators:
                if hasattr(operator, "tv_pf"):
                    tv_pf = operator.tv_pf
                    break
            if tv_pf is not None:
                break

        U_bar = self.macroscopic_stretch_subsol.subfunc
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
            
    def add_Darcy_operator(self,
                        # --- kinematics / fields ---
                        kinematics,
                        U,
                        U_test,
                        X,
                        X_0,

                        # --- macro loads ---
                        grad_p_bar_ini,
                        grad_p_bar_fin,
                        pl_bar_ini,
                        pl_bar_fin,
                        Theta_in_ini,
                        Theta_in_fin,
                        Theta_out_ini,
                        Theta_out_fin,

                        # --- material ---
                        k_l0,               # 2x2 tensor baseline intrinsic permeability (current config)
                        use_kozeny_carman,  # whether to use Kozeny-Carman relative permeability (True) or a constant factor (False)

                        # --- domain / boundary ids ---
                        subdomain_id,
                        inlet_id,
                        outlet_id,

                        # --- step index ---
                        k_step):

        # Pressure perturbation unknown and its test function
        p_tilde = self.pl_perturbation_subsol.subfunc
        p_test  = self.pl_perturbation_subsol.dsubtest

        # Measures restricted to subdomain and (optional) inlet/outlet regions
        dx     = self.get_subdomain_measure(subdomain_id)
        dx_in  = self.get_subdomain_measure(inlet_id)   if inlet_id  is not None else None
        dx_out = self.get_subdomain_measure(outlet_id)  if outlet_id is not None else None

        # Build Darcy operator (reference weak form; outputs q_l/Q_l/k_l_intr/Phis_expr for post-processing)
        operator = MicroDarcyFlowOperator(
            kinematics=kinematics,
            U=U,
            U_test=U_test,
            X=X,
            X_0=X_0,
            p_tilde=p_tilde,
            p_test=p_test,
            grad_p_bar_ini=grad_p_bar_ini,
            grad_p_bar_fin=grad_p_bar_fin,
            pl_bar_ini=pl_bar_ini,
            pl_bar_fin=pl_bar_fin,
            use_kozeny_carman=use_kozeny_carman,
            k_l0=k_l0,
            Phis0=self.Phis0,
            kappa_val=self.kappa_val,
            dx=dx,
            dx_in=dx_in,
            dx_out=dx_out,
            Theta_in_ini=Theta_in_ini,
            Theta_in_fin=Theta_in_fin,
            Theta_out_ini=Theta_out_ini,
            Theta_out_fin=Theta_out_fin,
        )

        # Add any additional FOIs/QOIs you already manage centrally
        self.add_darcy_fois(operator)

        # Register operator for this load step
        return self.add_operator(operator=operator, k_step=k_step)
        
    def add_darcy_qois(self, symmetric=False):
        """
        Add Darcy-related QOIs for the current step.

        QOIs are reference-domain averages (normalized by Area_ref = ∫_Ω 1 dX):
        - pl_bar_avg
        - Q_l_avg_x, Q_l_avg_y      (Piola/reference flux)
        - grad_p_bar_avg_x, grad_p_bar_avg_y
        """

        if not self.steps:
            raise RuntimeError("No steps available.")

        step = self.steps[-1]

        # ---- find Darcy operator in this step ----
        darcy_op = None
        for op in step.operators:
            # Prefer the new API: Q_l is explicitly stored by MicroDarcyFlowOperator
            if hasattr(op, "Q_l") and hasattr(op, "grad_p_bar") and hasattr(op, "pl_bar"):
                darcy_op = op
                break
        if darcy_op is None:
            raise RuntimeError("Cannot find Darcy operator in current step.")

        dx = getattr(darcy_op, "dx_measure", None)
        if dx is None:
            raise RuntimeError("Darcy operator has no dx_measure.")

        Area_ref = dolfin.assemble(1.0 * dx)

        # Reference averages over the Darcy subdomain
        # Average over RVE box
        self.add_qoi(name="Q_l_avg_x",        expr=darcy_op.Q_l[0] * dx,       norm=self.V0)
        self.add_qoi(name="Q_l_avg_y",        expr=darcy_op.Q_l[1] * dx,       norm=self.V0)

        # Average over solid skeleton 
        self.add_qoi(name="pl_bar_avg",       expr=darcy_op.pl_bar * dx,       norm=Area_ref)
        self.add_qoi(name="grad_p_bar_avg_x", expr=darcy_op.grad_p_bar[0] * dx, norm=Area_ref)
        self.add_qoi(name="grad_p_bar_avg_y", expr=darcy_op.grad_p_bar[1] * dx, norm=Area_ref)

    def add_darcy_fois(self, operator):
        """
        Register Darcy-related fields of interest (FOIs).
        This function is idempotent: it adds FOIs only once.
        """

        if getattr(self, "_darcy_fois_added", False):
            return
        self._darcy_fois_added = True

        # --- scalar fields ---
        # Stored as UFL expressions inside the operator (preferred: no re-definition here).
        self.add_foi(expr=operator.Phis_expr,   fs=self.sfoi_fs, name="Phis",   update_type="project")
        self.add_foi(expr=operator.Phif_expr,   fs=self.sfoi_fs, name="Phif",   update_type="project")

        self.add_foi(expr=operator.pl_affine,   fs=self.sfoi_fs, name="pl_affine", update_type="project")
        self.add_foi(expr=operator.pl_bar,      fs=self.sfoi_fs, name="pl_bar",    update_type="project")
        self.add_foi(expr=operator.pl_tot,      fs=self.sfoi_fs, name="pl_tot",    update_type="project")

        # Optional: relative permeability factor (scalar) if you store it on the operator
        if hasattr(operator, "k_rel_expr"):
            self.add_foi(expr=operator.k_rel_expr, fs=self.sfoi_fs, name="k_rel", update_type="project")

        # --- tensor fields ---
        # K_l: pull-back permeability used in the reference weak form
        self.add_foi(expr=operator.K_l,       fs=self.mfoi_fs, name="K_l_ref",   update_type="project")

        # k_l_intr: intrinsic permeability tensor in current configuration
        self.add_foi(expr=operator.k_l_intr,  fs=self.mfoi_fs, name="k_l_intr",  update_type="project")

        # --- fluxes (vectors) ---
        # q_l: current Darcy flux, Q_l: Piola/reference flux
        self.add_foi(expr=operator.q_l,       fs=self.vfoi_fs, name="q_l",       update_type="project")
        self.add_foi(expr=operator.Q_l,       fs=self.vfoi_fs, name="Q_l",       update_type="project")

 
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
    
    
    def get_porosity_subsol(self):

        return self.get_subsol(self.get_porosity_name())

    def get_porosity_function_space(self):

        return self.get_subsol_function_space(name=self.get_porosity_name())
    
    def add_Wskel_operator(self,
            material_parameters,
            material_scaling,
            subdomain_id=None):

        operator = WskelPoroFlowOperator(
            kinematics=self.kinematics,
            U=self.displacement_perturbation_subsol.subfunc,
            U_test=self.displacement_perturbation_subsol.dsubtest,
            Phis0=self.Phis0,
            material_parameters=material_parameters,
            material_scaling=material_scaling,
            measure=self.get_subdomain_measure(subdomain_id))
        return self.add_operator(operator)
    
    def get_porosity_name(self):
        return "Phis"
    
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


    def get_sigma_total(self):
        sigma_total = None

        for op in self.operators:
            sig = getattr(op, "sigma_contrib", None)
            if sig is None:
                continue
            sigma_total = sig if sigma_total is None else (sigma_total + sig)

        if sigma_total is None:

            dim = self.dim  
            sigma_total = dolfin.Constant(((0.0,) * dim,) * dim)

        return sigma_total

