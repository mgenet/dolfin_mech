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
            

        computing_cost_func = False
        if computing_cost_func:
            func_lst = []
            V0 = dolfin.assemble(dolfin.Constant(1)*problem.dV)
            U_norm =  (dolfin.assemble(dolfin.inner(U, U)*problem.dV)/2/V0)**(1/2)
            scale = 1/1.25*U_norm
            U_noise = U.copy(deepcopy=True)
            noise = U.copy(deepcopy=True)
            noise.vector()[:] = numpy.random.normal(loc=0.0, scale=scale, size=U.vector().get_local().shape)
            U_noise.vector().set_local(U.vector().get_local()[:]+noise.vector().get_local()[:])
            kinematics = dmech.LinearizedKinematics(u=U_noise)
            E_lst = numpy.linspace(-360, 360, 450)
            for E_ in E_lst:
                material_parameters["E"] = E_
                material   = dmech.material_factory(kinematics, "Hooke", material_parameters, problem)
                sigma = material.sigma 
                n = problem.mesh_normals
                sigma_t = dolfin.dot(sigma, n) + surface_forces[0][0] * n
                norm_sigma_t = (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*surface_forces[0][1])/dolfin.assemble(dolfin.Constant(1)*surface_forces[0][1]) )
                func_lst.append(norm_sigma_t)
            results = {}
            results["E"] = E_lst
            results["func"] = func_lst
            df = pd.DataFrame(results)
            myfile= open("/Users/peyrault/Seafile/PhD_Alice/Equilibrium_gap_method/Results_cube/func_noise.dat", 'w')
            myfile.write(df.to_string(index=False))
            myfile.close()




        

        print("starting optimization")
        results = {}
        # storing_results = {}
        initialisation_values = []
        for key, value in initialisation_estimation.items():
            initialisation_values.append(value)
            results[str(key)] = []
        print("results ini", results)

        projection = False
        
        path_solution =  "/Users/peyrault/Seafile/PhD_Alice/Equilibrium_gap_method/Results_cube/results.dat"
        results={}
        params_names = []
        initialisation=[]
        noises = [0., 1/20., 1/10., 1./5, 1/2.5]
        results["noises"] = noises
        initialisation_values, ref_values = {}, []
        for key, value in initialisation_estimation.items():
            ref_values.append(float(value))
            results[key]=[]
            params_names.append(key)
            initialisation.append(value)
        nb_parameters = len(initialisation_estimation)
        for noise_level in noises:
            compteur_value = 0
            initialisation_values[noise_level] = []
            for value in ref_values :
                initialisation_values[noise_level].append([])
                for compteur in range(0,1000):
                    initial_value = float(numpy.random.normal(loc=value, scale=abs(0.3*value), size=1))
                    initialisation_values[noise_level][compteur_value].append(initial_value)
                compteur_value +=1
        V0 = dolfin.assemble(dolfin.Constant(1)*problem.dV)
        U_norm =  (dolfin.assemble(dolfin.inner(U, U)*problem.dV)/2/V0)**(1/2)
        noise_number = 0
        for noise_level in noises:
            print("noise=", noise_level)
            params_opt = {}
            for i in range(0, nb_parameters):
                params_opt[params_names[i]]=[]
                results[params_names[i]].append([])
            compteur_iter = 0
            if noise_level !=0:
                scale = noise_level*U_norm
            else:
                scale = 0.
            converged=False
            while not converged:
                init = []
                for key, value in initialisation_estimation.items():
                    for param in range(0, nb_parameters):
                        initialisation_estimation[key]=initialisation_values[noise_level][param][compteur_iter]
                        init.append(initialisation_values[noise_level][param][compteur_iter])
                U_noise = U.copy(deepcopy=True)
                noise = U.copy(deepcopy=True)
                noise.vector()[:] = numpy.random.normal(loc=0.0, scale=scale, size=U.vector().get_local().shape)
                U_noise.vector().set_local(U.vector().get_local()[:]+noise.vector().get_local()[:])

                if projection:
                    mesh_coarser = dolfin.BoxMesh(
                        dolfin.Point(0, 0, 0), dolfin.Point(100, 100, 100),
                        2, 2, 22)
                    mesh_finer = dolfin.BoxMesh(
                        dolfin.Point(0, 0, 0), dolfin.Point(100, 100, 100),
                        10, 10, 10)
                    V = dolfin.VectorFunctionSpace(mesh_finer,'CG',1)
                    W = dolfin.VectorFunctionSpace(mesh_coarser,'CG',1)
                
                    U_noise_smooth = dolfin.project(U_noise, W)
                    U_noise_smooth = dolfin.project(U_noise_smooth,V)


                    U_noise.vector().set_local(U_noise_smooth.vector().get_local()[:])

                

                kinematics = dmech.LinearizedKinematics(u=U_noise)
                
                # if inverse:
                #     kinematics = dmech.InverseKinematics(u=U_noise)
                #     # kinematics = dmech.InverseKinematics(u=U_noise_smooth)
                # else:
                #     kinematics=dmech.Kinematics(U=U_noise)
                    # kinematics=dmech.Kinematics(U=U_noise_smooth)
                # print("kinematics retrieved!")

                sol = scipy.optimize.minimize(J, init, args=(inverse, problem, kinematics, material_parameters, material_model, V0, initialisation_estimation, surface_forces, volume_forces, boundary_conditions), method="Nelder-Mead")
                if sol.success == True:
                    print("iteration completed with value for initialisation", compteur_iter, sol.x[0], init, sol.fun)
                    for i in range(0, nb_parameters):
                        results[params_names[i]][noise_number].append((float(sol.x[i])-initialisation[i])/initialisation[i]*100)
                        params_opt[params_names[i]].append(float(sol.x[i]))
                        if compteur_iter > 4:
                            converged, crit = criteria_convergence(params_opt, 0.005)
                            print("converged, crit", converged, crit)
                        if converged:
                            converged=True    
                    compteur_iter += 1
                else:
                    print("did not converge...")
            noise_number +=1


        print("results after calculation", results)
        for key, value in initialisation_estimation.items():
            if key != noises:
                lst_global =  results[key]
                results.pop(key)
                results[key+'mean'] = []
                results[key+'plus'] = []
                results[key+'minus'] = []
                for i in range(len(noises)):
                    list = lst_global[i]
                    key_mean = numpy.average(list)
                    key_minus = numpy.average(list) - abs(numpy.std(list))
                    key_plus = numpy.average(list) + abs(numpy.std(list))
                    results[key+'mean'].append(key_mean)
                    results[key+'plus'].append(key_plus)
                    results[key+'minus'].append(key_minus)
                        
        print("results final", results)
        df = pd.DataFrame(results)
        myfile= open(path_solution, 'w')
        myfile.write(df.to_string(index=False))
        myfile.close()


        


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
            if not inverse: 
                # print("J=", kinematics.J )
                # print("denominateur",dolfin.assemble(dolfin.Constant(1)*kinematics.J*nf_norm*Sref) )
                # print("numerateur", dolfin.assemble(dolfin.inner(dolfin.dot(Sigma, n), dolfin.dot(Sigma, n))*Sref))
                norm_sigma_t += (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*kinematics.J*nf_norm*Sref)/dolfin.assemble(dolfin.Constant(1)*kinematics.J*nf_norm*Sref) ) #  / abs(norm_params)
            else:
                norm_sigma_t += (1/2*dolfin.assemble(dolfin.inner(sigma_t, sigma_t)*Sref)/dolfin.assemble(dolfin.Constant(1)*Sref) ) #  / abs(norm_params)
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
    return(div_sigma_value+norm_sigma_t)



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



def newton(
        get_res,
        get_jac,
        sol,
        res_rel_tol=None,
        res_cvg_tol=None,
        res_abs_tol=None,
        sol_cvg_tol=None,
        n_max_iter=100,
        tab="    ",
        verbose=False):
    """
    A simple Newton solver for nonlinear problems.
    It operates on numpy arrays.
    Multiple exit criteria are considered, including relative residual error, residual convergence error, absolute residual error, and solution convergence error.
    If the maximum number of iterations is reached, the loop is exited with a failure flag.

    Args:
        get_res  (function): A function returning the residual vector (actually, a dim x 1 2D numpy array) for a given state of the solution vector.
        get_jac  (function): A function returning the jacobian matrix (a dim x dim 2D numpy array) for a given state of the solution vector.
        sol (numpy.ndarray): Initial value of the solution.
        res_rel_tol (float): The tolerance for relative residual error, i.e., |res|/|res_0|.
        res_cvg_tol (float): The tolerance for residual convergence error, i.e., |res-res_old|-|/|res_old|.
        res_abs_tol (float): The tolerance for absolute residual error, i.e., |res|.
        sol_cvg_tol (float): The tolerance for solution convergence error, i.e., |dsol|/|sol|.
        n_max_iter    (int): The maximum number of iterations.
        tab           (str): A string to shift output lines for better visualisation or iterations informations.
        verbose      (bool): A flag to output, or not, iterations informations.

    Returns:
        success (bool): If the solver converged:                     True; Otherwise: False.
        sol    (float): If the solver converged:      The converged value; Otherwise: None.
        k_iter   (int): If the solver converged: The number of iterations; Otherwise: None.
    """

    assert (type(sol) == numpy.ndarray), "sol must be a numpy array (currently type(sol) = "+str(type(sol))+"). Aborting."
    assert (sol.ndim == 1), "sol must be a 1D numpy array (currently sol.ndim = "+str(sol.ndim)+"). Aborting."
    if (verbose): print ("sol = "+str(sol))
    
    res = get_res(sol)
    if (verbose): print ("res = "+str(res))
    assert (type(res) == numpy.ndarray), "get_res must return a numpy array (currently type(res) = "+str(type(res))+"). Aborting."
    assert (res.ndim == 2), "get_res must return a 2D numpy array (currently res.ndim = "+str(res.ndim)+"). Aborting."
    assert (res.shape[1] == 1), "get_res must return a dim x 1 numpy array (currently res.shape[1] = "+str(res.shape[1])+"). Aborting."
    dim = res.shape[0]
    res_old = numpy.copy(res)
    res_0 = numpy.copy(res)
    
    jac = get_jac(sol)
    if (verbose): print ("jac = "+str(jac))
    assert (type(jac) == numpy.ndarray), "get_jac must return a numpy array (currently type(jac) = "+str(type(jac))+"). Aborting."
    assert (jac.shape == (dim,dim)), "get_jac must return a dim x dim numpy array, where dim is the dimension of res (currently jac.shape = "+str(jac.shape)+" and dim = "+str(dim)+"). Aborting."
    
    k_iter = 0
    while (True):
        k_iter += 1
        if (verbose): print (tab+"k_iter = "+str(k_iter))

        # safety test to prevent infinite loop
        if (k_iter == n_max_iter):
            success = False
            break

        # for residual convergence error
        if (res_cvg_tol is not None):
            if (k_iter > 1):
                res_old[:] = res[:]

        # residual
        res[:] = get_res(sol) ### YOUR CODE HERE ###
        if (verbose): print (tab+"res = "+str(res))

        # residual relative error
        if (res_rel_tol is not None):
            if (k_iter == 1):
                res_0 = numpy.linalg.norm(res)
            if (k_iter > 1):
                if (res_0 != 0):
                    res_rel_err = numpy.linalg.norm(res)/res_0
                    if (verbose): print (tab+"res_rel_err = "+str(res_rel_err))
                    if (res_rel_err <= res_rel_tol):
                        success = True
                        break

        # residual convergence error
        if (res_cvg_tol is not None):
            if (k_iter > 1):
                if (numpy.linalg.norm(res_old) != 0):
                    res_cvg_err = numpy.linalg.norm(res - res_old)/numpy.linalg.norm(res_old)
                    if (verbose): print (tab+"res_cvg_err = "+str(res_cvg_err))
                    if (res_cvg_err <= res_cvg_tol):
                        success = True
                        break

        # residual absolute error
        if (res_abs_tol is not None):
            res_abs_err = numpy.linalg.norm(res)
            if (verbose): print (tab+"res_abs_err = "+str(res_abs_err))
            if (res_abs_err <= res_abs_tol):
                success = True
                break

        # jacobian
        jac[:] = get_jac(sol) 
        if (verbose): print (tab+"jac = "+str(jac))

        # solution increment
        dsol = - numpy.linalg.solve(jac, res) 
        if (verbose): print (tab+"dsol = "+str(dsol))

        # solution convergence error
        if (sol_cvg_tol is not None):
            if (numpy.linalg.norm(sol) != 0):
                sol_cvg_err = numpy.linalg.norm(dsol)/numpy.linalg.norm(sol)
                if (verbose): print (tab+"sol_cvg_err = "+str(sol_cvg_err))
                if (sol_cvg_err <= sol_cvg_tol):
                    success = True
                    break

        # solution update
        sol[:] += dsol[:,0] 
        if (verbose): print (tab+"sol = "+str(sol))

    if (success):
        if (verbose): print (tab+"SUCCESS!")
        return success, sol, k_iter
    else:
        if (verbose): print (tab+"FAILURE!")
        return success, None, None


    
    



    