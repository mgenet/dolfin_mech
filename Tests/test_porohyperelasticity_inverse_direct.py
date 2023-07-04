#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

#################################################################### imports ###

import sys
import dolfin

import myPythonLibrary as mypy
import dolfin_mech     as dmech

################################################################# parameters ###

mat_params = {
    "alpha":0.16,
    "gamma":0.5,
    "c1":0.6,
    "c2":0.,
    "kappa":1e2,
    "eta":1e-5}


####################################################################### test ###

res_folder = sys.argv[0][:-3]
test = mypy.Test(
    res_folder=res_folder,
    perform_tests=1,
    stop_at_failure=1,
    clean_after_tests=1)


dim = 3

inverse_lst  = []
inverse_lst += [1]
inverse_lst += [0]

load_lst  = []
load_lst += [["p_boundary_condition", "p_boundary_condition0"]]
load_lst += [["external", "external0"]]

print("load_lst", load_lst)

for loads in load_lst:
    print("loads=", loads)

    try:
        del phis
    except:
        pass
    try:
        del U_tot
    except:
        pass

    for inverse in inverse_lst:

        porosity_lst  = []
    
        scaling = "linear"

        load = loads[-inverse]

        print("inverse =",inverse)
        print("load =",load)

        try:
            phis
        except NameError:
            porosity = {"type": "mesh_function_xml", "val":0.5}
            move=False
            U_move=None
        else:
            porosity = {"type": "function_xml_from_array", "val":phis}
            move=True
            U_move=U
    
        if load=="p_boundary_condition" or load=="p_boundary_condition0":
            cube_params={"X1":1, "Y1":1, "Z1":1, "l": 0.1} #### AP2023 - necessary for convergence
        else:
            cube_params={}

        res_basename  = sys.argv[0][:-3]
        res_basename += "-inverse="+str(inverse)

        U, phis, V = dmech.RivlinCube_PoroHyperelasticity(
                    dim=dim,
                    inverse=inverse,
                    cube_params=cube_params,
                    porosity_params=porosity,
                    mat_params={"scaling":scaling, "parameters":mat_params},
                    step_params={"dt_min":1e-4},
                    load_params={"type":load},
                    move = {"move":move, "U":U_move},
                    res_basename=res_folder+"/"+res_basename,
                    plot_curves=0,
                    get_results=1,
                    verbose=0)

        try:
            U_tot
        except NameError:
            U_tot = U.copy(deepcopy=True)
        else:
            U_tot.vector()[:] += U.vector()[:]
        
    U_tot_norm = (dolfin.assemble(dolfin.inner(U_tot, U_tot)*V)/2/dolfin.assemble(dolfin.Constant(1)*V))**(1/2)  

    print("displacement norm", U_tot_norm)

    assert (U_tot_norm/cube_params.get("X1", 1.) < 1e-2),\
        "Warning, did not find the initial geometry. Aborting."
    
    test.test(res_basename)
