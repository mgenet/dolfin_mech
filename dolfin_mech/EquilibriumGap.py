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
import numpy

import dolfin_mech as dmech
import dolfin_warp as dwarp

from .Operator import Operator

import math
import random
import pandas as pd
import copy 
import numpy

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
            boundary_conditions,
            U):
        
        # path_solution =  "/Users/peyrault/Seafile/PhD_Alice/Equilibrium_gap_method/Results_cube/results.dat"
        initialisation_estimation={"E":numpy.log(1)}
        init = [numpy.log(1.)]
        V0 = dolfin.assemble(dolfin.Constant(1)*problem.dV)
        sol = scipy.optimize.minimize(J_recalage, init, args=(inverse, problem, kinematics, material_parameters, material_model, V0, initialisation_estimation, surface_forces, volume_forces, boundary_conditions), method="Nelder-Mead")
        print("iteration completed with value for initialisation", numpy.exp(sol.x[0]), init, sol.fun)
            
        # # print("number degrees of freedom", len(U.vector()[:]))
        # # print("volume is", problem.mesh_V0)

        # computing_cost_func = False
        # if computing_cost_func:
        #     func_lst = []
        #     V0 = dolfin.assemble(dolfin.Constant(1)*problem.dV)
        #     U_norm =  (dolfin.assemble(dolfin.inner(U, U)*problem.dV)/2/V0)**(1/2)
        #     scale = 1/1.25*U_norm
        #     U_noise = U.copy(deepcopy=True)
        #     noise = U.copy(deepcopy=True)
        #     noise.vector()[:] = numpy.random.normal(loc=0.0, scale=scale, size=U.vector().get_local().shape)
        #     U_noise.vector().set_local(U.vector().get_local()[:]+noise.vector().get_local()[:])
        #     kinematics = dmech.LinearizedKinematics(u=U_noise)
        #     E_lst = numpy.linspace(-360, 360, 450)
        #     for E_ in E_lst:
        #         material_parameters["E"] = E_
        #         material   = dmech.material_factory(kinematics, "Hooke", material_parameters, problem)
        #         sigma = material.sigma 
        #         n = problem.mesh_normals
        #         sigma_t = dolfin.dot(sigma, n) + surface_forces[0][0] * n
        #         norm_sigma_t = (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*surface_forces[0][1])/dolfin.assemble(dolfin.Constant(1)*surface_forces[0][1]) )
        #         func_lst.append(norm_sigma_t)
        #     results = {}
        #     results["E"] = E_lst
        #     results["func"] = func_lst
        #     df = pd.DataFrame(results)
        #     myfile= open("/Users/peyrault/Seafile/PhD_Alice/Equilibrium_gap_method/Results_cube/func_noise.dat", 'w')
        #     myfile.write(df.to_string(index=False))
        #     myfile.close()

        # checking_norm_epsilon = False
        # if checking_norm_epsilon:
        #     results_epsilon = {}
        #     results_epsilon["noise"] = []
        #     results_epsilon["epsilon_u_meas_norm"] = []
        #     results_epsilon["epsilon_noise_norm"] = []
        #     results_epsilon["epsilon_u_noise_norm"] = []
        #     results_epsilon["ratio"] = []
        #     noises = [1/20., 1/10., 1/5., 1/2.5, 1/1.25]
        #     V0 = dolfin.assemble(dolfin.Constant(1)*problem.dV)
        #     U_norm =  (dolfin.assemble(dolfin.inner(U, U)*problem.dV)/2/V0)**(1/2)
        #     for noise_level in noises:
        #         results_epsilon["noise"].append(noise_level)
        #         epsilon_u_noise_norm_lst, epsilon_noise_norm_lst, epsilon_u_meas_norm_lst = [], [], []
        #         for i in range(100):
        #             scale=noise_level*U_norm
        #             U_noise = U.copy(deepcopy=True)
        #             noise = U.copy(deepcopy=True)
        #             noise.vector()[:] = numpy.random.normal(loc=0.0, scale=scale, size=U.vector().get_local().shape)
        #             U_noise.vector().set_local(U.vector().get_local()[:]+noise.vector().get_local()[:])
        #             kinematics_u_noise = dmech.LinearizedKinematics(u=U_noise)
        #             epsilon_u_noise=kinematics_u_noise.epsilon
        #             kinematics_noise = dmech.LinearizedKinematics(u=noise)
        #             epsilon_noise=kinematics_noise.epsilon
        #             kinematics_u_meas = dmech.LinearizedKinematics(u=U)
        #             epsilon_u_meas=kinematics_u_meas.epsilon
        #             epsilon_u_noise_norm= (dolfin.assemble(dolfin.inner(epsilon_u_noise, epsilon_u_noise)*problem.dV)/2/V0)
        #             epsilon_noise_norm= (dolfin.assemble(dolfin.inner(epsilon_noise, epsilon_noise)*problem.dV)/2/V0)
        #             epsilon_u_meas_norm= (dolfin.assemble(dolfin.inner(epsilon_u_meas, epsilon_u_meas)*problem.dV)/2/V0)
        #             epsilon_u_noise_norm_lst.append(epsilon_u_noise_norm)
        #             epsilon_noise_norm_lst.append(epsilon_noise_norm)
        #             epsilon_u_meas_norm_lst.append(epsilon_u_meas_norm)
        #         epsilon_u_meas_norm = numpy.average(epsilon_u_meas_norm_lst)
        #         epsilon_noise_norm = numpy.average(epsilon_noise_norm_lst)
        #         epsilon_u_noise_norm = numpy.average(epsilon_u_noise_norm_lst)
        #         results_epsilon["epsilon_u_meas_norm"].append(epsilon_u_meas_norm)
        #         results_epsilon["epsilon_noise_norm"].append(epsilon_noise_norm)
        #         results_epsilon["epsilon_u_noise_norm"].append(epsilon_u_noise_norm)
        #         results_epsilon["ratio"].append(epsilon_u_meas_norm/epsilon_noise_norm)
        #         # print(noise_level, epsilon_u_noise_norm, epsilon_noise_norm, epsilon_u_meas_norm)
        #     print(results_epsilon)
        
        # print("starting optimization")
        # results = {}
        # # storing_results = {}
        # initialisation_values = []
        # for key, value in initialisation_estimation.items():
        #     initialisation_values.append(value)
        #     results[str(key)] = []
        # print("results ini", results)

        # projection = False

        # images_recalage = True

        # if not images_recalage:
        #     print("should not be there !")
        
        #     path_solution =  "/Users/peyrault/Seafile/PhD_Alice/Equilibrium_gap_method/Results_cube/results.dat"
        #     results={}
        #     params_names = []
        #     initialisation=[]
        #     noises = [1/10., 1/20., 1/10., 1/5.]
        #     results["noises"] = noises
        #     initialisation_values, ref_values = {}, []
        #     for key, value in initialisation_estimation.items():
        #         ref_values.append(float(value))
        #         results[key]=[]
        #         params_names.append(key)
        #         initialisation.append(value)
        #     nb_parameters = len(initialisation_estimation)
        #     for noise_level in noises:
        #         compteur_value = 0
        #         initialisation_values[noise_level] = []
        #         for value in ref_values :
        #             initialisation_values[noise_level].append([])
        #             for compteur in range(0,1000):
        #                 initial_value = float(numpy.random.lognormal(mean=value, sigma=abs(0.5*value), size=1))
        #                 # initialisation_values[noise_level][compteur_value].append(initial_value)
        #                 initialisation_values[noise_level][compteur_value].append(numpy.log(2*value))
        #             compteur_value +=1
        #     V0 = dolfin.assemble(dolfin.Constant(1)*problem.dV)
        #     U_norm =  (dolfin.assemble(dolfin.inner(U, U)*problem.dV)/2/V0)**(1/2)
        #     noise_number = 0
        #     for noise_level in noises:
        #         print("noise=", noise_level)
        #         params_opt = {}
        #         for i in range(0, nb_parameters):
        #             params_opt[params_names[i]]=[]
        #             results[params_names[i]].append([])
        #         compteur_iter = 0
        #         if noise_level !=0:
        #             scale = noise_level*U_norm
        #         else:
        #             scale = 0.
        #         print("scale", scale)
        #         converged=False
        #         while not converged:
        #             init = []
        #             for key, value in initialisation_estimation.items():
        #                 for param in range(0, nb_parameters):
        #                     initialisation_estimation[key]=initialisation_values[noise_level][param][compteur_iter]
        #                     init.append(initialisation_values[noise_level][param][compteur_iter])
        #             U_noise = U.copy(deepcopy=True)
        #             random_gaussian_fields = True
        #             if random_gaussian_fields:
        #                 U_tmp = U.copy(deepcopy=True)
        #                 beta_min  = 2*math.pi/50
        #                 beta_max  = 2*math.pi/50
        #                 theta_min = 0.
        #                 theta_max = 2*math.pi
        #                 gamma_min = 0.
        #                 gamma_max = 2*math.pi
        #                 N = 1
        #                 for _ in range(N):
        #                     betax  = random.uniform( beta_min,  beta_max)
        #                     betay  = random.uniform( beta_min,  beta_max)
        #                     betaz  = random.uniform( beta_min,  beta_max)
        #                     thetax = random.uniform(theta_min, theta_max)
        #                     thetay = random.uniform(theta_min, theta_max)
        #                     thetaz = random.uniform(theta_min, theta_max)
        #                     gammax = random.uniform(gamma_min, gamma_max)
        #                     gammay = random.uniform(gamma_min, gamma_max)
        #                     gammaz = random.uniform(gamma_min, gamma_max)
        #                     U_expr = dolfin.Expression(
        #                         ("cos(betax * (x[0]*nxx + x[1]*nxy + x[2]*nxz) - gammax)",
        #                             "cos(betay * (x[0]*nyx + x[1]*nyy + x[2]*nyz) - gammay)",
        #                             "cos(betaz * (x[0]*nzx + x[1]*nzy + x[2]*nzz) - gammaz)"
        #                             ),
        #                         betax=betax, betay=betay, betaz=betaz,
        #                         nxx=math.cos(thetax)*math.sin(thetaz), nyx=math.sin(thetaz)*math.cos(thetay), nzx=math.cos(thetax)*math.sin(thetaz),
        #                         nxy=math.sin(thetaz)*math.sin(thetax), nyy=math.sin(thetay)*math.sin(thetaz), nzy=math.sin(thetay)*math.sin(thetaz),
        #                         nxz=math.cos(thetaz), nyz=math.cos(thetaz), nzz=math.cos(thetaz),
        #                         gammax=gammax, gammay=gammay, gammaz=gammaz,
        #                         element=problem.sol_fe)
        #                     U_tmp.interpolate(U_expr)
        #                     # print (problem.U.vector().get_local())
        #                     # print (U_tmp.vector().get_local())
        #                     U_noise.vector().axpy(noise_level/N, U_tmp.vector())
        #                     # print (problem.U.vector().get_local())
        #             else:
        #                 print("should not go there ?!")
        #                 noise = U.copy(deepcopy=True)
        #                 noise.vector()[:] = numpy.random.normal(loc=0.0, scale=scale, size=U.vector().get_local().shape)
        #             mean_value=0
        #             # for i in range(len(noise.vector()[:])):
        #             #     mean_value += noise.vector()[i]
        #             # mean_value = mean_value/i
        #             # print("mean value =", mean_value)
        #             # noise_norm = (dolfin.assemble(dolfin.dot(noise, noise)*problem.dV)/2/V0)**(1/2)
        #             # print("noise_norm", noise_norm)
        #             if projection:
        #                 print("should not go there either...")
        #                 mesh_coarser = dolfin.BoxMesh(
        #                     dolfin.Point(0, 0, 0), dolfin.Point(100, 100, 100),
        #                     10, 10, 10)
        #                 mesh_finer = dolfin.BoxMesh(
        #                     dolfin.Point(0, 0, 0), dolfin.Point(100, 100, 100),
        #                     20, 20, 20)
        #                 V = dolfin.VectorFunctionSpace(mesh_finer,'CG',1)
        #                 W = dolfin.VectorFunctionSpace(mesh_coarser,'CG',1)
                    
        #                 noise = dolfin.project(noise, W)
        #                 noise = dolfin.project(noise,V)
                        
        #             # U_noise.vector().set_local(U.vector().get_local()[:]+noise.vector().get_local()[:])

        #             U_noise_mean, U_mean =  0, 0
        #             for i in range(len(U_noise.vector()[:])):
        #                 U_noise_mean += U_noise.vector()[i]
        #                 U_mean += U.vector()[i]

        #             # print("U vs U_noise mean", U_noise_mean, U_mean )

                    

                    

        #             kinematics = dmech.LinearizedKinematics(u=U_noise)
                    
        #             # if inverse:
        #             #     kinematics = dmech.InverseKinematics(u=U_noise)
        #             #     # kinematics = dmech.InverseKinematics(u=U_noise_smooth)
        #             # else:
        #             #     kinematics=dmech.Kinematics(U=U_noise)
        #                 # kinematics=dmech.Kinematics(U=U_noise_smooth)
        #             # print("kinematics retrieved!")

        #             sol = scipy.optimize.minimize(J, init, args=(inverse, problem, kinematics, material_parameters, material_model, V0, initialisation_estimation, surface_forces, volume_forces, boundary_conditions), method="Nelder-Mead")
        #             if sol.success == True:
        #                 print("iteration completed with value for initialisation", compteur_iter, numpy.exp(sol.x[0]), init, sol.fun)
        #                 for i in range(0, nb_parameters):
        #                     # results[params_names[i]][noise_number].append((float(sol.x[i])-initialisation[i])/initialisation[i]*100)
        #                     # params_opt[params_names[i]].append(float(sol.x[i]))
        #                     results[params_names[i]][noise_number].append((numpy.exp(float(sol.x[i]))-initialisation[i])/initialisation[i]*100)
        #                     params_opt[params_names[i]].append(numpy.exp(float(sol.x[i])))
        #                     if compteur_iter > 4:
        #                         converged, crit = criteria_convergence(params_opt, 0.005)
        #                         print("converged, crit", converged, crit)
        #                     if converged:
        #                         converged=True    
        #                 compteur_iter += 1
        #             else:
        #                 print("did not converge...")
        #         noise_number +=1


        #     print("results after calculation", results)
        #     for key, value in initialisation_estimation.items():
        #         if key != noises:
        #             lst_global =  results[key]
        #             results.pop(key)
        #             results[key+'mean'] = []
        #             results[key+'plus'] = []
        #             results[key+'minus'] = []
        #             for i in range(len(noises)):
        #                 list = lst_global[i]
        #                 key_mean = numpy.average(list)
        #                 key_minus = numpy.average(list) - abs(numpy.std(list))
        #                 key_plus = numpy.average(list) + abs(numpy.std(list))
        #                 results[key+'mean'].append(key_mean)
        #                 results[key+'plus'].append(key_plus)
        #                 results[key+'minus'].append(key_minus)
                            
        #     print("results final", results)
        #     df = pd.DataFrame(results)
        #     myfile= open(path_solution, 'w')
        #     myfile.write(df.to_string(index=False))
        #     myfile.close()

        # else:
            
        


def J_recalage(x, inverse, problem, kinematics, material_parameters, material_model, V0, initialisation_estimation, surface_forces, volume_forces, boundary_conditions):
    
    
    indice_param=0

    print(x[0])

    parameters_to_identify = {}

    for key, value in initialisation_estimation.items():
        try:
            # print(x[indice_param])
            material_parameters[key]=numpy.exp(x[indice_param])
            parameters_to_identify[key]=numpy.exp(x[indice_param])
            indice_param+=1
        except:
            pass
    print("material parameters", material_parameters)
    print("parameters_to_identify", parameters_to_identify)
    # print("material model", material_model)
    if material_model==None:
        material   = dmech.material_factory(kinematics, "CGNHMR_poro", material_parameters, problem)
        assert(inverse is not None),\
            "should say if inverse or direct problem... Aborting!"
        if inverse:
            sigma = material.sigma 
            n = problem.mesh_normals
        else:
            Sigma = kinematics.F*material.Sigma
            N = problem.mesh_normals
            nf = dolfin.dot(N, dolfin.inv(kinematics.F))
            nf_norm = dolfin.sqrt(dolfin.inner(nf,nf))
    elif material_model=="CGNH":
        if not inverse:
            # print("in the right material model")
            material   = dmech.material_factory(kinematics, material_model, material_parameters)
            Sigma= material.Sigma 
            N = problem.mesh_normals
            nf = dolfin.dot(N, dolfin.inv(kinematics.F))
            nf_norm = dolfin.sqrt(dolfin.inner(nf,nf))
        else:
            material   = dmech.material_factory(kinematics, material_model, material_parameters)
            sigma=material.sigma 
            n = problem.mesh_normals
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
        # print("div_sigma", div_sigma_value)
    # div_sigma_value=0

    norm_sigma_t = 0

    # surface_forces = []
    # print("surface_forces", surface_forces, len(surface_forces))
    if surface_forces != []:
        redo_loop=True
        S_old = []
        Sref = surface_forces[0][1]
        # print("Sref=", Sref)
        while redo_loop:
            redo_loop=False
            if inverse:
                sigma_t = dolfin.dot(sigma, n)
            else:
                sigma_t = dolfin.dot(Sigma, N)
                # print("norm_sigma_t")
            # print("sigma_t", sigma_t)
            for force in surface_forces:
                if force[1]==Sref:
                    if inverse:
                        if type(force) == float:
                            sigma_t += dolfin.Constant(force[0])*n
                        else:
                            sigma_t += force[0]*n
                    else:
                        if type(force) == float:
                            sigma_t += dolfin.Constant(force[0])*kinematics.J*nf
                        else:
                            sigma_t += force[0]*kinematics.J*nf
                if force[1]!=Sref and force[1] not in S_old:
                    Sref_temp = force[1]
                    redo_loop=True
                S_old.append(Sref)
            # norm_sigma_t_test=0
            # norm_sigma_t_test =  (dolfin.assemble(dolfin.dot(sigma_t, dolfin.Constant(force[0])*n) *Sref)/dolfin.assemble(dolfin.Constant(1)*Sref) )**2
            # for i in range(3):
            #     norm_sigma_t_test +=  (dolfin.assemble(sigma_t[i] *Sref)/dolfin.assemble(dolfin.Constant(1)*Sref) )**2
            # print("norm sigma_t_test", norm_sigma_t_test)
            if not inverse: 
                # print("J=", kinematics.J )
                # print("denominateur",dolfin.assemble(dolfin.Constant(1)*kinematics.J*nf_norm*Sref) )
                # print("numerateur", dolfin.assemble(dolfin.inner(dolfin.dot(Sigma, n), dolfin.dot(Sigma, n))*Sref))
                norm_sigma_t += (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*kinematics.J*nf_norm*Sref)/dolfin.assemble(dolfin.Constant(1)*kinematics.J*nf_norm*Sref) ) #  / abs(norm_params)
                print("norm_sigma_t", abs(norm_sigma_t))
            else:
                norm_sigma_t += ((1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*Sref)/dolfin.assemble(dolfin.Constant(1)*Sref) )) #  / abs(norm_params)
                print("norm_sigma_t", norm_sigma_t)
            # print("norm_sigma_t", norm_sigma_t)
            if redo_loop:
                Sref=Sref_temp

    return(abs(norm_sigma_t))



def J(x, inverse, problem, kinematics, material_parameters, material_model, V0, initialisation_estimation, surface_forces, volume_forces, boundary_conditions):
    indice_param=0

    parameters_to_identify = {}

    for key, value in initialisation_estimation.items():
        try:
            # print(x[indice_param])
            material_parameters[key]=numpy.exp(x[indice_param])
            parameters_to_identify[key]=numpy.exp(x[indice_param])
            indice_param+=1
        except:
            pass
    

    if material_model==None:
        material   = dmech.material_factory(kinematics, "CGNHMR_poro", material_parameters, problem)
        assert(inverse is not None),\
            "should say if inverse or direct problem... Aborting!"
        if inverse:
            sigma = material.sigma 
            n = problem.mesh_normals
        else:
            Sigma = kinematics.F*material.Sigma
            N = problem.mesh_normals
            nf = dolfin.dot(N, dolfin.inv(kinematics.F))
            nf_norm = dolfin.sqrt(dolfin.inner(nf,nf))
    else:
        material   = dmech.material_factory(kinematics, material_model, material_parameters)
        sigma = material.sigma 
        n = problem.mesh_normals

    div_sigma_value = 0

    # print("computing div sigma")
    # div_sigma = dwarp.VolumeRegularizationDiscreteEnergy(problem=problem,  b=[0.,0.,0.], model=material_model, parameters_to_identify=parameters_to_identify, inverse=inverse)
    # div_sigma_value = div_sigma.assemble_ener()
    # print("computed div sigma")
    # print("computed div sigma", div_sigma_value)

    volume_forces = []
    # print("volume forces", volume_forces[0][0])
    if volume_forces != []:
        # print("volume_force", volume_forces[0][0])
        # print("parameters to identify", parameters_to_identify)
        div_sigma = dwarp.VolumeRegularizationDiscreteEnergy(problem=problem,  b=volume_forces[0][0], model=material_model, parameters_to_identify=parameters_to_identify, inverse=inverse)
        div_sigma_value = div_sigma.assemble_ener()  # / abs(norm_params) 
        # print("div_sigma", div_sigma_value)
    # div_sigma_value=0

    norm_sigma_t = 0

    # surface_forces = []
    # print("surface_forces", surface_forces, len(surface_forces))
    if surface_forces != []:
        redo_loop=True
        S_old = []
        Sref = surface_forces[0][1]
        # print("Sref=", Sref)
        while redo_loop:
            redo_loop=False
            if inverse:
                sigma_t = dolfin.dot(sigma, n)
            else:
                sigma_t = dolfin.dot(Sigma, N)
                # print("norm_sigma_t")
            # print("sigma_t", sigma_t)
            for force in surface_forces:
                if force[1]==Sref:
                    if inverse:
                        if type(force) == float:
                            sigma_t += dolfin.Constant(force[0])*n
                        else:
                            sigma_t += force[0]*n
                    else:
                        if type(force) == float:
                            sigma_t += dolfin.Constant(force[0])*kinematics.J*nf
                        else:
                            sigma_t += force[0]*kinematics.J*nf
                if force[1]!=Sref and force[1] not in S_old:
                    Sref_temp = force[1]
                    redo_loop=True
                S_old.append(Sref)
            norm_sigma_t_test=0
            norm_sigma_t_test =  (dolfin.assemble(dolfin.dot(sigma_t, dolfin.Constant(force[0])*n) *Sref)/dolfin.assemble(dolfin.Constant(1)*Sref) )**2
            # for i in range(3):
            #     norm_sigma_t_test +=  (dolfin.assemble(sigma_t[i] *Sref)/dolfin.assemble(dolfin.Constant(1)*Sref) )**2
            # print("norm sigma_t_test", norm_sigma_t_test)
            if not inverse: 
                # print("J=", kinematics.J )
                # print("denominateur",dolfin.assemble(dolfin.Constant(1)*kinematics.J*nf_norm*Sref) )
                # print("numerateur", dolfin.assemble(dolfin.inner(dolfin.dot(Sigma, n), dolfin.dot(Sigma, n))*Sref))
                norm_sigma_t += (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*kinematics.J*nf_norm*Sref)/dolfin.assemble(dolfin.Constant(1)*kinematics.J*nf_norm*Sref) ) #  / abs(norm_params)
            else:
                norm_sigma_t += ((1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*Sref)/dolfin.assemble(dolfin.Constant(1)*Sref) )) #  / abs(norm_params)
            # print("norm_sigma_t", norm_sigma_t)
            if redo_loop:
                Sref=Sref_temp
    # print("out of loop")


    # print("norm_sigma_t", norm_sigma_t)
            

    # sigma_t = dolfin.dot(sigma, problem.mesh_normals) - 1*problem.mesh_normals    
    # norm_sigma_t = (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*problem.dS)/dolfin.assemble(dolfin.Constant(1)*problem.dS) )**(1/2)

        
        
    # print("norm tsigma - sigma", (1/2*dolfin.assemble(dolfin.inner(sigma.T-sigma, sigma.T-sigma)*Sref)))
           
    # print("norm_sigma_t", norm_sigma_t)    

    
    # sigma_t = dolfin.dot(sigma, problem.mesh_normals) - 1*problem.mesh_normals

    # norm_sigma_t = (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t) * problem.dS)/dolfin.assemble(dolfin.Constant(1)*problem.dS))**(1/2)  
    # print("norm_sigma_t_working", norm_sigma_t)

    
    # print("function", div_sigma_value+norm_sigma_t)
    # print("norm sigma.n - t", norm_sigma_t)
    # print(norm_sigma_t_test)
    return(norm_sigma_t_test)
    # return(norm_sigma_t) #+1e-1*abs(x[0]-160))



def criteria_convergence(params_opt={}, tol=1e-3):
    converged =False
    crit_max = []
    for key, list in params_opt.items():
        l = list[:]
        if numpy.percentile(l[:-1], 50) == 0:
            median = abs((numpy.percentile(l[:],50)-numpy.percentile(l[:-1],50)))
        else:
            median = abs((numpy.percentile(l[:],50)-numpy.percentile(l[:-1],50))/numpy.percentile(l[:-1],50))
        if numpy.percentile(l[:-1],25) == 0:
            q1 = abs((numpy.percentile(l[:],25)-numpy.percentile(l[:-1],25)))
        else:
            q1 = abs((numpy.percentile(l[:],25)-numpy.percentile(l[:-1],25))/numpy.percentile(l[:-1],25))
        if numpy.percentile(l[:-1],25) == 0:
            q3 = abs((numpy.percentile(l[:],75)-numpy.percentile(l[:-1],75)))
        else:
            q3 = abs((numpy.percentile(l[:],75)-numpy.percentile(l[:-1],75))/numpy.percentile(l[:-1], 75))
        ##### to modify, but tempor fix with problem with e-16 values as it is ==> std 1 while it is between 2 numerical 0
        if numpy.std(l[:-1])==0:
            std_plus = abs(numpy.percentile(l[:],50)+numpy.std(l[:])-numpy.percentile(l[:-1],50)-numpy.std(l[:-1]))
            std_minus = abs(numpy.percentile(l[:],50)-numpy.std(l[:])-numpy.percentile(l[:-1],50)-numpy.std(l[:-1]))
        else:
            std_plus = abs((numpy.percentile(l[:],50) + numpy.std(l[:]) - numpy.std(l[:-1])-numpy.percentile(l[:-1],50))/(numpy.std(l[:-1])+numpy.percentile(l[:-1],50)))
            std_minus = abs((numpy.percentile(l[:],50) - numpy.std(l[:]) + numpy.std(l[:-1])-numpy.percentile(l[:-1],50))/(-numpy.std(l[:-1])+numpy.percentile(l[:-1],50)))
        std = max(std_minus, std_plus)
        # print("average for list", l, numpy.average(l[:]), numpy.average(l[:-1]))
        if numpy.average(l[:-1]) !=0 :
            average = abs((numpy.average(l[:])-numpy.average(l[:-1]))/numpy.average(l[:-1]))
        else:
            average = abs((numpy.average(l[:])-numpy.average(l[:-1])))
        crit_max.append(median)
        crit_max.append(q1)
        crit_max.append(q3)
        crit_max.append(std)
        crit_max.append(average)
    # for list in params_init_distrib:
    #     l = list[1:]
    #     mean_distrib = numpy.mean(l)
    #     std_distrib = abs(numpy.std(l)-30)/30
        # print("average for list", l, numpy.average(l[:]), numpy.average(l[:-1]))
    criteria = max(crit_max)
    # if criteria < float(tol) and abs(mean_distrib) < 2 and std_distrib < 0.02:
    if criteria < float(tol):
        converged = True
    # print("criteria, mean, std", criteria)
    return(converged, criteria)