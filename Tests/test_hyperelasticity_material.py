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

    incomp_lst  = []
    incomp_lst += [0]
    incomp_lst += [1]
    for incomp in incomp_lst:

        mat_params = {
            "E":1.,
            "nu":0.5*(incomp)+0.3*(1-incomp)}

        mat_lst  = []
        if (incomp):
            mat_lst += ["NH"]
            mat_lst += ["NH_bar"]
            mat_lst += ["NHMR"]
            mat_lst += ["NHMR_bar"]
            mat_lst += ["SVK_dev"]
        else:
            mat_lst += ["CGNH"]
            mat_lst += ["CGNH_bar"]
            mat_lst += ["CGNHMR"]
            mat_lst += ["CGNHMR_bar"]
            mat_lst += ["SVK"]
        for mat in mat_lst:

            print("dim =",dim)
            print("incomp =",incomp)
            print("mat =",mat)

            res_basename  = sys.argv[0][:-3]
            res_basename += "-dim="+str(dim)
            res_basename += "-incomp="+str(incomp)
            res_basename += "-mat="+str(mat)

            dmech.run_RivlinCube_Hyperelasticity(
                dim          = dim                                        ,
                incomp       = incomp                                     ,
                cube_params  = {"mesh_filebasename":res_folder+"/"+"mesh"},
                mat_params   = {"model":mat, "parameters":mat_params}     ,
                load_params  = {"type":"disp"}                            ,
                res_basename = res_folder+"/"+res_basename                ,
                verbose      = 0                                          )

            test.test(res_basename)
