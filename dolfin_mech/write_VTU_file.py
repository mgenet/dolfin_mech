#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
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
        preserve_connectivity=False,
        refine_and_interpolate_before_write=0):

    if (preserve_connectivity):
        ugrid = dmech.mesh2ugrid(function.function_space().mesh())
        dmech.add_function_to_ugrid(
            function=function,
            ugrid=ugrid)
        myvtk.writeUGrid(
            ugrid=ugrid,
            filename=filebasename+("_"+str(time).zfill(zfill) if (time is not None) else "")+".vtu")

    else:
        if (refine_and_interpolate_before_write):
            mesh = function.function_space().mesh()
            for _ in range(refine_and_interpolate_before_write):
                mesh = dolfin.refine(mesh)
            V = dolfin.FunctionSpace(mesh, "CG", 1)
            function = dolfin.interpolate(function, V)

        file_pvd = dolfin.File(filebasename+"__.pvd")
        file_pvd << (function, float(time) if (time is not None) else 0.)
        os.remove(
            filebasename+"__.pvd")
        shutil.move(
            filebasename+"__"+"".zfill(6)+".vtu",
            filebasename+("_"+str(time).zfill(zfill) if (time is not None) else "")+".vtu")
