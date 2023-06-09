#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import scipy.optimize

import dolfin_mech as dmech
import dolfin_warp as dwarp

from .Operator import Operator

################################################################################



class EquilibriumGap(Operator):

    def __init__(self,
            problem,
            kinematics,
            material_model,
            material_parameters,
            initialisation_estimation,
            surface_forces,
            volume_forces,
            boundary_conditions):
        

        print("starting optimization")

        initialisation_values = []
        for key, value in initialisation_estimation.items():
            initialisation_values.append(value)

        print("initialisation_estimation", initialisation_estimation)
        V0 = dolfin.assemble(dolfin.Constant(1)*problem.dV)
        sol = scipy.optimize.minimize(J, initialisation_values, args=(problem, kinematics, material_parameters, material_model, V0, initialisation_estimation, surface_forces, volume_forces, boundary_conditions), method="Nelder-Mead") #  options={'xatol':1e-12, 'fatol':1e-12}

        print("real parameters", 160, 0.3)
        indice_printing_results_opti = 0
        for key, value in initialisation_estimation.items():
            print("optimised parameter", key, "=", sol.x[indice_printing_results_opti])
            indice_printing_results_opti+=1
        print("optimised function=", sol.fun)
        # print("optimized parameters are", sol.x[0])

def J(x, problem, kinematics, material_parameters, material_model, V0, initialisation_estimation, surface_forces, volume_forces, boundary_conditions):
    indice_param=0

    parameters_to_identify = {}

    for key, value in initialisation_estimation.items():
        try:
            material_parameters[key]=x[indice_param]
            parameters_to_identify[key]=x[indice_param]
            indice_param+=1
        except:
            pass
    
    norm_params = 1
    for i in range(len(initialisation_estimation)):
        norm_params *= x[i]
        print(x[i])


    # print("E", x[0])
    # print("material_parameters before", material_parameters)
    # material_parameters['E']=x[0]
    # material_parameters['nu']=x[1]
    # print("material_parameters after", material_parameters)
    # print("material model is", material_model)
    porous = False
    if material_model==None:
        # porous = True
        
        # solid_material_skel = dmech.WskelLungElasticMaterial(
        #         kinematics=kinematics,
        #         parameters=material_parameters)
        # material_skel = dmech.PorousElasticMaterial(
        #         solid_material= solid_material_skel,
        #         scaling="linear",
        #         Phis0=kinematics.J*problem.get_porosity_subsol().subfunc)   
        


        # solid_material_bulk = dmech.WbulkLungElasticMaterial(
        #     Phis=kinematics.J * problem.phis, 
        #     Phis0=kinematics.J * problem.get_porosity_subsol().subfunc, 
        #     parameters={"kappa":1e2}, 
        #     kinematics=kinematics)
        # material_bulk = dmech.PorousElasticMaterial(
        #         solid_material= solid_material_bulk,
        #         scaling="linear",
        #         Phis0= kinematics.J * problem.get_porosity_subsol().subfunc)  
         
        # solid_material_pores = dmech.WporeLungElasticMaterial(
        #         Phif = kinematics.J - problem.get_porosity_subsol().subfunc/kinematics.J,
        #         Phif0= 1 - problem.get_porosity_subsol().subfunc,
        #         kinematics=kinematics,
        #         parameters={"eta":1e-5})
        # material_pores = dmech.PorousElasticMaterial(
        #         solid_material= solid_material_pores,
        #         scaling="linear",
        #         Phis0=problem.get_porosity_subsol().subfunc)   
        # material = dmech.WskelLungElasticMaterial(kinematics=kinematics, parameters=material_parameters)
        material   = dmech.material_factory(kinematics, "CGNHMR_poro", material_parameters, problem)
    else:
        material   = dmech.material_factory(kinematics, material_model, material_parameters)
    

    # print("material computed")
    # print("material=", material)

    # print(kinematics.epsilon)
    
    # print(material_parameters)
    if porous:
        # operator_skel = problem.add_Wskel_operator(
        #         material_parameters=material_parameters,
        #         material_scaling="linear",
        #         subdomain_id=None)
        
        # operator_bulk = problem.add_Wbulk_operator(
        #         material_parameters={"kappa":1e2},
        #         material_scaling="linear",
        #         subdomain_id= None)
        
        # sigma = operator_skel.material.sigma + operator_bulk.material.sigma
        # sigma = material_skel.sigma + material_bulk.sigma #  + material_pores.sigma
        sigma = material.sigma 
    else:
        sigma = material.sigma 

    # print("norm tr(epsilon)", dolfin.assemble(dolfin.inner(dolfin.tr(kinematics.epsilon),dolfin.tr(kinematics.epsilon))*problem.dV))
    # print("norm epsilon", dolfin.assemble(dolfin.inner(kinematics.epsilon, kinematics.epsilon)*problem.dV))
    # print("norm div(tr(epsilon))", dolfin.assemble(dolfin.inner(dolfin.div(dolfin.tr(kinematics.epsilon)*kinematics.I), dolfin.div(dolfin.tr(kinematics.epsilon)*kinematics.I))*problem.dV))
    # print("norm div(epsilon)", dolfin.assemble(dolfin.inner(dolfin.div(kinematics.epsilon), dolfin.div(kinematics.epsilon))*problem.dV))
    div_sigma_value = 0
    # print("volume forces", volume_forces[0][0])
    if volume_forces != []:
        # print("volume_force", volume_forces[0][0])
        # div_sigma = dwarp.VolumeRegularizationDiscreteEnergy(problem=problem, b=volume_forces[0][0], model=material_model, young=x[0], poisson=x[1])
        div_sigma = dwarp.VolumeRegularizationDiscreteEnergy(problem=problem, b=[volume_forces[0][0][0]*1e-6*problem.phis,volume_forces[0][0][1]*1e-6*problem.phis,volume_forces[0][0][2]*1e-6*problem.phis], model=material_model, parameters_to_identify=parameters_to_identify)
        div_sigma_value = div_sigma.assemble_ener()  # / abs(norm_params) 
        print("div_sigma", div_sigma_value)
        # div_sigma = dolfin.div(sigma)+dolfin.Constant(volume_forces[0][0])
    # div_sigma_value=0

    #     norm_div_sigma += (1/2*dolfin.assemble(dolfin.inner(div_sigma,div_sigma)*volume_forces[0][1])/dolfin.assemble(dolfin.Constant(1)*problem.dV))**(1/2)
    # print("norm_div_sigma", norm_div_sigma)

    norm_sigma_t = 0

    
    # print("surface_forces", surface_forces)
    if surface_forces != []:
        redo_loop=True
        S_old = []
        Sref = surface_forces[0][1]
        # print("Sref=", Sref)
        while redo_loop:
            redo_loop=False
            sigma_t = dolfin.dot(sigma, problem.mesh_normals)
            # print("sigma_t", sigma_t)
            for force in surface_forces:
                if force[1]==Sref:
                    if type(force) == float:
                        sigma_t += dolfin.Constant(force[0])*problem.mesh_normals
                    else:
                        sigma_t += force[0]*problem.mesh_normals
                if force[1]!=Sref and force[1] not in S_old:
                    Sref_temp = force[1]
                    redo_loop=True
                S_old.append(Sref)
            norm_sigma_t += (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*Sref)/dolfin.assemble(dolfin.Constant(1)*Sref) )**(1/2) #  / abs(norm_params)
            if redo_loop:
                Sref=Sref_temp
    print("out of loop")

    # norm_sigma_t = 0
    print("norm_sigma_t", norm_sigma_t)
            

    # sigma_t = dolfin.dot(sigma, problem.mesh_normals) - 1*problem.mesh_normals    
    # norm_sigma_t = (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*problem.dS)/dolfin.assemble(dolfin.Constant(1)*problem.dS) )**(1/2)

        
        
    # print("norm tsigma - sigma", (1/2*dolfin.assemble(dolfin.inner(sigma.T-sigma, sigma.T-sigma)*Sref)))
           
    # print("norm_sigma_t", norm_sigma_t)    

    
    # sigma_t = dolfin.dot(sigma, problem.mesh_normals) - 1*problem.mesh_normals

    # norm_sigma_t = (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t) * problem.dS)/dolfin.assemble(dolfin.Constant(1)*problem.dS))**(1/2)  
    # print("norm_sigma_t_working", norm_sigma_t)

    
    print("function", div_sigma_value+norm_sigma_t)
    # print("norm sigma.n - t", norm_sigma_t)
    return(div_sigma_value+norm_sigma_t)
    



    