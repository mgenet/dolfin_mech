#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2020-2024                                    ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

# import dolfin
# import numpy
# import sys

# import myPythonLibrary as mypy
# from pathlib import Path
# local_path = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(local_path))
# import dolfin_mech     as dmech

# sys.path.remove(str(local_path))
import dolfin
import math
import numpy
import myPythonLibrary as mypy
import sys
import os
from pathlib import Path
local_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(local_path))

import dolfin_mech as dmech
from dolfin_mech.Problem_Darcy_MicroPoro import MicroPoroDarcyProblem
from dolfin_mech.run_HollowBox_MicroPoroFlowHyperelasticity import run_HollowBox_MicroPoroFlow

####################################################################### test ###

res_folder = sys.argv[0][:-3]
test = mypy.Test(
    res_folder=res_folder,
    perform_tests=0,
    stop_at_failure=1,
    clean_after_tests=0,
    tester_numpy_tolerance=1e-2)

dim_lst  = [ ]
dim_lst += [2]
# dim_lst += [3]
for dim in dim_lst:

    bcs_lst  = [      ]
    #bcs_lst += ["kubc"]
    bcs_lst += ["pbc" ]
    for bcs in bcs_lst:

        load_lst  = [                     ]
        #load_lst += ["internal_pressure"  ]
        #load_lst += ["macroscopic_stretch"]
        #load_lst += ["macroscopic_stress" ]
        load_lst += ["macro_grad_p"]

        for load in load_lst:

            print("dim =",dim)
            print("bcs =",bcs)
            print("load =",load)

            #res_basename  = sys.argv[0][:-3]
            res_basename = "-dim="+str(dim)
            res_basename += "-bcs="+str(bcs)
            res_basename += "-load="+str(load)

            load_params = {}
            if (load == "internal_pressure"):
                load_params["pf"] = +0.2
                for i in range(dim):
                 for j in range (dim):
                    load_params["sigma_bar_"+str(i)+str(j)] = 0.
            elif (load == "macroscopic_stretch"):
                load_params["pf"] = 0.
                load_params["U_bar_00"] = 0.5
                for i in range(dim):
                 for j in range (dim):
                  if ((i != 0) or (j != 0)):
                    load_params["sigma_bar_"+str(i)+str(j)] = 0.
            elif (load == "macroscopic_stress"):
                load_params["pf"] = 0.
                # load_params["grad_p_bar_0"] = 1
                # load_params["grad_p_bar_1"] = 0.5 
                for i in range(dim):
                 for j in range (dim):
                    load_params["sigma_bar_"+str(i)+str(j)] = 0.
                load_params["sigma_bar_00"] = 0.5

            elif (load == "macro_grad_p"):
                load_params["pf"] = 0.
                #load_params["p_bar"] = 0.0   # 如果你要做响应曲线，这里固定成 0
                # 定义宏观梯度方向（例如沿 y 方向）
                load_params["grad_p_bar_0"] = 1
                load_params["grad_p_bar_1"] = 0.5  # 你之后做循环扫 alpha 时会覆盖这个


                # 固体相关载荷全部置零，确保只有压力作用
                # for i in range(dim):
                #     for j in range(dim):
                #         load_params["U_bar_00"] = 0.5
                #         load_params["U_bar_"+str(i)+str(j)] = 0.0
                #         #load_params["sigma_bar_"+str(i)+str(j)] = 0.0

                load_params["U_bar_00"] = 0.0   
                for i in range(dim):
                    for j in range (dim):
                        if ((i != 0) or (j != 0)):
                            load_params["sigma_bar_"+str(i)+str(j)] = 0.




            mat_params = {
                "alpha":0.16,
                "gamma":0.5,
                "c1":0.2,
                "c2":0.4,
                "kappa":1e2,
                "eta":1e-5}


            run_HollowBox_MicroPoroFlow(
                dim=dim,
                mesh_params={"dim":dim, "xmin":0., "ymin":0., "zmin":0., "xmax":1., "ymax":1., "zmax":1., "xshift":-0.3, "yshift":-0.3, "zshift":-0.3, "r0":0.2, "l":0.04, "mesh_filebasename":res_folder+"/"+"mesh"},
                mat_params=mat_params,
                bcs=bcs,
                #step_params={"n_steps":5,"dt_ini":1e-1, "dt_min":1e-3},
                step_params={"dt_ini":0.05, "dt_min":1e-3},
                load_params=load_params,
                res_basename=res_folder+"/"+res_basename,
                #res_basename=res_basename,
                verbose=1)

            test.test(res_basename)

###############################################################################
## TEST CODE FINISHED
###############################################################################