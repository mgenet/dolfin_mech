#coding=utf8

################################################################################
###                                                                          ###
### Created by Alice Peyraut, 2023-2024                                      ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2023                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

#################################################################### imports ###

import sys

import dolfin

import myPythonLibrary as mypy
import dolfin_mech     as dmech

####################################################################### test ###

res_folder = sys.argv[0][:-3]
test = mypy.Test(
    res_folder=res_folder,
    perform_tests=1,
    stop_at_failure=1,
    clean_after_tests=1)

move, U_move = False, None
inverse_lst = [0,1]
dim_lst  = []
dim_lst += [2]
dim_lst += [3]
for dim in dim_lst:

    cube_params = {"X1":1., "Y1":1., "l": 0.1}
    if (dim==3):
        cube_params["Z1"]:1.
    cube_params["mesh_filebasename"] = res_folder+"/"+"mesh"

    load_lst  = []
    load_lst += [["volu", "volu0"]]
    load_lst += [["surf", "surf0"]]
    load_lst += [["pres", "pres0"]]
    load_lst += [["pgra", "pgra0"]]
    for loads in load_lst:
        try:
            del U_tot
        except:
            pass

        for inverse in inverse_lst:
            load=loads[inverse]

            print("dim =",dim)
            print("load =",load)
            print("inverse=", inverse)

            res_basename  = sys.argv[0][:-3]
            res_basename += "-dim="+str(dim)
            res_basename += "-load="+str(load)

            U, V=dmech.run_RivlinCube_Hyperelasticity(
                dim=dim,
                cube_params=cube_params,
                mat_params={"model":"CGNHMR", "parameters":{"E":1., "nu":0.3, "dim":dim}},
                step_params={"dt_min":0.1},
                load_params={"type":load},
                move_params={"move":move, "U":U_move},
                res_basename=res_folder+"/"+res_basename,
                inverse=inverse,
                get_results=1,
                verbose=0)
            
            try:
                U_tot
            except NameError:
                U_tot = U.copy(deepcopy=True)
                move=True
                U_move=U
                V_ini = V
            else:
                U_tot.vector()[:] += U.vector()[:]
                move=False
                U_move=None
            
        U_tot_norm = (dolfin.assemble(dolfin.inner(U_tot, U_tot)*V_ini)/2/dolfin.assemble(dolfin.Constant(1)*V))**(1/2)  

        print("displacement norm", U_tot_norm)

        assert (U_tot_norm/cube_params.get("X1", 1.) < 1e-2),\
            "Warning, did not find the initial geometry. Aborting."
        
        test.test(res_basename)
