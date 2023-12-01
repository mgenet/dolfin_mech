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

import myPythonLibrary as mypy
import dolfin_mech     as dmech

####################################################################### test ###

res_folder = sys.argv[0][:-3]
test = mypy.Test(
    res_folder=res_folder,
    perform_tests=1,
    stop_at_failure=1,
    clean_after_tests=1)

n_steps_lst  = []
n_steps_lst += [1]
n_steps_lst += [2]
n_steps_lst += [3]
for n_steps in n_steps_lst:

    provide_u_or_u_lst_lst  = []
    provide_u_or_u_lst_lst += ["u"]
    provide_u_or_u_lst_lst += ["u_lst"]
    for provide_u_or_u_lst in provide_u_or_u_lst_lst:

        print("n_steps ="           ,n_steps           )
        print("provide_u_or_u_lst =",provide_u_or_u_lst)

        load_params = {}
        load_params["type"] = "disp"
        if   (provide_u_or_u_lst == "u"):
            load_params["u"] = 0.5
        elif (provide_u_or_u_lst == "u_lst"):
            load_params["u_lst"] = [(k_step+1)*0.5/n_steps for k_step in range(n_steps)]

        res_basename  = sys.argv[0][:-3]
        res_basename += "-n_steps="+str(n_steps)
        res_basename += "-provide_u_or_u_lst="+str(provide_u_or_u_lst)

        dmech.run_RivlinCube_Hyperelasticity(
            dim          = 2,
            cube_params  = {"mesh_filebasename":res_folder+"/"+"mesh"},
            mat_params   = {"model":"CGNH", "parameters":{"E":1., "nu":0.3}},
            step_params  = {"n_steps":n_steps, "Deltat": 1., "dt_ini":1., "dt_min":0.1},
            load_params  = load_params,
            res_basename = res_folder+"/"+res_basename,
            verbose      = 0)

        test.test(res_basename)
