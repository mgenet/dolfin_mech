#coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2023                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2023                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
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
    clean_after_tests=1,
    tester_numpy_tolerance=2e-2)

dim_lst  = []
dim_lst += [2]
dim_lst += [3]

for dim in dim_lst:

    print("dim =",dim)

    centering_lst  = []
    centering_lst += [0]
    centering_lst += [1]

    for centering in centering_lst:

        res_basename  = sys.argv[0][:-3]
        res_basename += "-dim="+str(dim)
        res_basename += "-centering="+str(centering)

        dmech.HollowBox_Homogenization(
            mat_params={"E":1.0, "nu":0.3},
            mesh_params={"dim":dim, "centering":centering, "width":1, "r0":1/5},
            res_basename=res_folder+"/"+res_basename,
            verbose=1)

        test.test(res_basename)
