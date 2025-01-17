#coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2024                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2025                                              ###
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
    tester_numpy_tolerance=1e-2)

dim_lst  = [ ]
dim_lst += [2]
# dim_lst += [3]

for dim in dim_lst:

    print("dim =",dim)

    centering_lst  = []
    centering_lst += [0]
    centering_lst += [1]

    for centering in centering_lst:

        if (centering):
            xshift = -1/2
            yshift = -1/2
            zshift = -1/2
        else:
            xshift = 0.
            yshift = 0.
            zshift = 0.

        res_basename  = sys.argv[0][:-3]
        res_basename += "-dim="+str(dim)
        res_basename += "-centering="+str(centering)

        dmech.run_HollowBox_Homogenization(
            dim=dim,
            mesh_params={"dim":dim, "xmin":0., "ymin":0., "zmin":0., "xmax":1., "ymax":1., "zmax":1., "xshift":xshift, "yshift":yshift, "zshift":zshift, "r0":1/5, "l":1/20, "mesh_filebasename":res_folder+"/mesh"},
            mat_params={"E":1.0, "nu":0.3},
            res_basename=res_folder+"/"+res_basename,
            verbose=0)

        test.test(res_basename)
