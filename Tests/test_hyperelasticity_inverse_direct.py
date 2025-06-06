#coding=utf8

################################################################################
###                                                                          ###
### Created by Alice Peyraut, 2023-2024                                      ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2025                                              ###
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

dim_lst  = [ ]
dim_lst += [2]
# dim_lst += [3]
for dim in dim_lst:

    load_lst  = [                 ]
    load_lst += [["volu", "volu0"]]
    load_lst += [["surf", "surf0"]]
    load_lst += [["pres", "pres0"]]
    load_lst += [["pgra", "pgra0"]]
    for loads in load_lst:
        try:
            del U_tot
        except:
            pass

        inverse_lst = [0,1]
        for inverse in inverse_lst:
            load=loads[inverse]

            print("dim =",dim)
            print("load =",load)
            print("inverse =",inverse)

            res_basename  = sys.argv[0][:-3]
            res_basename += "-dim="+str(dim)
            res_basename += "-load="+str(load)

            U, dV=dmech.run_RivlinCube_Hyperelasticity(
                dim=dim,
                cube_params={"l":0.1, "mesh_filebasename":res_folder+"/"+"mesh"},
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
                move = True
                U_move = U
                dV_ini = dV
            else:
                U_tot.vector()[:] += U.vector()[:]
                move = False
                U_move = None
            
        U_tot_norm = (dolfin.assemble(dolfin.inner(U_tot, U_tot)*dV_ini)/2/dolfin.assemble(dolfin.Constant(1)*dV))**(1/2)  

        print("displacement norm", U_tot_norm)

        assert (U_tot_norm < 1e-2),\
            "Warning, did not find the initial geometry. Aborting."
        
        test.test(res_basename)
