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
            inverse,
            kinematics,
            material_model,
            material_parameters,
            initialisation_estimation,
            surface_forces,
            volume_forces,
            boundary_conditions):
        
        print("material_parameters", material_parameters)
        

        print("starting optimization")

        initialisation_values = []
        for key, value in initialisation_estimation.items():
            initialisation_values.append(value)

        print("initialisation_estimation", initialisation_estimation)
        V0 = dolfin.assemble(dolfin.Constant(1)*problem.dV)
        sol = scipy.optimize.minimize(J,  initialisation_values, args=(inverse, problem, kinematics, material_parameters, material_model, V0, initialisation_estimation, surface_forces, volume_forces, boundary_conditions), method="Nelder-Mead") #  options={'xatol':1e-12, 'fatol':1e-12}

        # print("real parameters", 160, 0.3)
        indice_printing_results_opti = 0
        for key, value in initialisation_estimation.items():
            print("optimised parameter", key, "=", sol.x[indice_printing_results_opti])
            indice_printing_results_opti+=1
        print("optimised function=", sol.fun)
        # print("optimized parameters are", sol.x[0])

def J(x, inverse, problem, kinematics, material_parameters, material_model, V0, initialisation_estimation, surface_forces, volume_forces, boundary_conditions):
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
    
    # print("material model is", material_model)
    if material_model==None:
        material   = dmech.material_factory(kinematics, "CGNHMR_poro", material_parameters, problem)
        assert(inverse is not None),\
            "should say if inverse or direct problem... Aborting!"
        if inverse:
            sigma = material.sigma 
            n = problem.mesh_normals
        else:
            sigma = material.sigma
            nf = dolfin.dot(problem.mesh_normals, dolfin.inv(kinematics.F))
            nf_norm = dolfin.sqrt(dolfin.inner(nf,nf))
            n = nf/nf_norm
    else:
        material   = dmech.material_factory(kinematics, material_model, material_parameters)
        sigma = material.sigma 
        n = problem.mesh_normals

    div_sigma_value = 0

    volume_forces = []
    # print("volume forces", volume_forces[0][0])
    if volume_forces != []:
        # print("volume_force", volume_forces[0][0])
        # print("parameters to identify", parameters_to_identify)
        div_sigma = dwarp.VolumeRegularizationDiscreteEnergy(problem=problem,  b=volume_forces[0][0], model=material_model, parameters_to_identify=parameters_to_identify, inverse=inverse)
        div_sigma_value = div_sigma.assemble_ener()  # / abs(norm_params) 
        print("div_sigma", div_sigma_value)
    # div_sigma_value=0

   

    norm_sigma_t = 0

    
    # print("surface_forces", len(surface_forces))
    if surface_forces != []:
        redo_loop=True
        S_old = []
        Sref = surface_forces[0][1]
        # print("Sref=", Sref)
        while redo_loop:
            redo_loop=False
            sigma_t = dolfin.dot(sigma, n)
            # print("sigma_t", sigma_t)
            for force in surface_forces:
                if force[1]==Sref:
                    if type(force) == float:
                        sigma_t += dolfin.Constant(force[0])*n
                    else:
                        sigma_t += force[0]*n
                if force[1]!=Sref and force[1] not in S_old:
                    Sref_temp = force[1]
                    redo_loop=True
                S_old.append(Sref)
            if material_model==None:
                if not inverse: 
                    norm_sigma_t += (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*kinematics.J*Sref)/dolfin.assemble(dolfin.Constant(1)*kinematics.J*Sref) )**(1/2) #  / abs(norm_params)
                else:
                    norm_sigma_t += (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*Sref)/dolfin.assemble(dolfin.Constant(1)*Sref) )**(1/2) #  / abs(norm_params)
            if redo_loop:
                Sref=Sref_temp
    # print("out of loop")


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
    



    