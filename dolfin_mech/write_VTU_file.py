#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import os
import shutil

import myVTKPythonLibrary as myvtk

import dolfin_mech as dmech

################################################################################

def write_VTU_file(
        filebasename,
        function,
        time=None,
        zfill=3,
        preserve_connectivity=False):

    if (preserve_connectivity):
        ugrid = dmech.mesh2ugrid(function.function_space().mesh())
        dmech.add_function_to_ugrid(
            function=function,
            ugrid=ugrid)
        myvtk.writeUGrid(
            ugrid=ugrid,
            filename=filebasename+("_"+str(time).zfill(zfill) if (time is not None) else "")+".vtu")

    else:
        file_pvd = dolfin.File(filebasename+"__.pvd")
        file_pvd << (function, float(time) if (time is not None) else 0.)
        os.remove(
            filebasename+"__.pvd")
        shutil.move(
            filebasename+"__"+"".zfill(6)+".vtu",
            filebasename+("_"+str(time).zfill(zfill) if (time is not None) else "")+".vtu")
