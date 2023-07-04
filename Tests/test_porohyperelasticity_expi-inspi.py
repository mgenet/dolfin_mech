#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
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
    clean_after_tests=0)


dim = 3

cube_params = {"X1":100, "Y1":100, "Z1":100, "l": 10}

inverse_lst  = []
inverse_lst += [1]
inverse_lst += [0]

for inverse in inverse_lst:

    porosity_lst  = []
    
    scaling = "linear"

    load_lst  = []
    load_lst += ["pgra"]
    load_lst += ["pgra0"]

    load = load_lst[-inverse]
    

    print("inverse =",inverse)
    print("load =",load)

    try:
        phis
    except NameError:
        porosity = {"type": "mesh_function_xml", "val":0.5}
        move=False
        U_move=None
    else:
        porosity = {"type": "function_xml_custom", "val":phis}
        move=True
        U_move=U
    

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
                get_results_fields = 1,
                verbose=1)

    try:
        U_new
    except NameError:
        U_new = U.copy(deepcopy=True)
    else:
        U_new.vector()[:] += U.vector()[:]
        
U_final_norm = (dolfin.assemble(dolfin.inner(U_new,U_new)*V)/2/dolfin.assemble(1*V))**(1/2)  
print("displacement norm", U_final_norm)

if U_final_norm/cube_params.get("X1", 1.) > 1e-2:
    print("Warning, did not find the initial geometry")  

test.test(res_basename)
