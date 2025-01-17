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

dim_lst  = []
dim_lst += [2]
dim_lst += [3]
for dim in dim_lst:

    PS_lst  = []
    if (dim == 2):
        PS_lst += [0]
        PS_lst += [1]
    elif (dim == 3):
        PS_lst += [0]
    for PS in PS_lst:

        incomp_lst  = []
        if (PS == 0):
            incomp_lst += [0]
            incomp_lst += [1]
        elif (PS == 1):
            incomp_lst += [0]
        for incomp in incomp_lst:

            if (incomp):
                mat_model = "H_dev"
            else:
                mat_model = "H"

            mat_params = {
                "E":1.,
                "nu":0.5*(incomp)+0.3*(1-incomp),
                "PS":PS}

            load_lst  = []
            load_lst += ["disp"]
            load_lst += ["volu"]
            load_lst += ["surf"]
            load_lst += ["pres"]
            load_lst += ["pgra"]
            load_lst += ["tens"]
            for load in load_lst:

                print("dim =",dim)
                if (dim == 2): print("PS =",PS)
                print("incomp =",incomp)
                print("load =",load)

                res_basename  = sys.argv[0][:-3]
                res_basename += "-dim="+str(dim)
                if (dim == 2): res_basename += "-PS"*PS + "-PE"*(1-PS)
                res_basename += "-incomp="+str(incomp)
                res_basename += "-load="+str(load)

                dmech.run_RivlinCube_Elasticity(
                    dim=dim,
                    incomp=incomp,
                    cube_params={"mesh_filebasename":res_folder+"/"+"mesh"},
                    mat_params={"model":mat_model, "parameters":mat_params},
                    load_params={"type":load},
                    res_basename=res_folder+"/"+res_basename,
                    verbose=0)

                test.test(res_basename)
