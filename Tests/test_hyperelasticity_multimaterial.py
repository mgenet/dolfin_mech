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

dim_lst  = []
dim_lst += [2]
dim_lst += [3]
for dim in dim_lst:

    incomp_lst  = []
    incomp_lst += [0]
    incomp_lst += [1]
    for incomp in incomp_lst:

        print("dim =",dim)
        print("incomp =",incomp)

        res_basename  = sys.argv[0][:-3]
        res_basename += "-dim="+str(dim)
        res_basename += "-incomp="+str(incomp)

        dmech.RivlinCube_Hyperelasticity(
            dim           = dim                                                 ,
            incomp        = incomp                                              ,
            multimaterial = 1                                                   ,
            cube_params   = {"l":0.1, "mesh_filebasename":res_folder+"/"+"mesh"},
            step_params   = {"dt_min":0.1}                                      ,
            load_params   = {"type":"disp"}                                     ,
            res_basename  = res_folder+"/"+res_basename                         ,
            verbose       = 0                                                   )

        test.test(res_basename)
