#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
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

    load_lst  = []
    load_lst += ["pres0"]
    load_lst += ["pres0_multi"]
    load_lst += ["pres0_inertia"]
    for load in load_lst:

        if (load in ("pres0", "pres0_multi")):
            const_params = {"type":"sym"}
        elif (load in ("pres0_inertia")):
            const_params = {"type":"None"}

        print("dim =",dim)
        print("load =",load)

        res_basename  = sys.argv[0][:-3]
        res_basename += "-dim="+str(dim)
        res_basename += "-load="+str(load)

        dmech.run_RivlinCube_Hyperelasticity(
            dim          = dim                                                ,
            inverse      = 1                                                  ,
            cube_params  = {"mesh_filebasename":res_folder+"/"+"mesh"}        ,
            mat_params   = {"model":"CGNHMR", "parameters":{"E":1., "nu":0.3}},
            step_params  = {"dt_min":0.1}                                     ,
            const_params = const_params                                  ,
            load_params  = {"type":load}                                      ,
            res_basename = res_folder+"/"+res_basename                        ,
            verbose      = 0                                                  )

        test.test(res_basename)
