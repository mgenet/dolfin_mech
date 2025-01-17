#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
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

incomp_lst  = [ ]
incomp_lst += [0]
incomp_lst += [1]
for incomp in incomp_lst:

    if (incomp):
        mat_model = "NHMR"
    else:
        mat_model = "CGNHMR"
    mat_params = {
        "E":1.,
        "nu":0.5*(incomp)+0.3*(1-incomp)}

    load_lst  = [      ]
    load_lst += ["disp"]
    load_lst += ["pres"]
    for load in load_lst:

        print("incomp =",incomp)
        print("load =",load)

        res_basename  = sys.argv[0][:-3]
        res_basename += "-incomp="+str(incomp)
        res_basename += "-load="+str(load)

        dmech.run_HeartSlice_Hyperelasticity(
            incomp=incomp,
            mesh_params={"X0":0.5, "Y0":0.5, "Ri":0.2, "Re":0.4, "l":0.05, "mesh_filebasename":res_folder+"/"+"mesh"},
            mat_params={"model":mat_model, "parameters":mat_params},
            step_params={"dt_ini":1/10, "dt_min":1/100},
            load_params={"type":load},
            res_basename=res_folder+"/"+res_basename,
            verbose=0)

        test.test(res_basename)
