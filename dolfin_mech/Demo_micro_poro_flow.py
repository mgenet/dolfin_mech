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

                load_params["U_bar_00"] = 1.0   
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
                #step_params={"dt_ini":0.05, "dt_min":1e-3},
                step_params={"dt_ini":0.1, "dt_min":0.01,"dt_max":0.1},
                load_params=load_params,
                res_basename=res_folder+"/"+res_basename,
                #res_basename=res_basename,
                verbose=1)

            test.test(res_basename)

###############################################################################
## TEST CODE FINISHED
###############################################################################
#%%

#%%
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 读取 QOI 文件
# -------------------------------------------------
def load_qois(qois_filename):
    qois_vals = np.loadtxt(qois_filename)
    with open(qois_filename, "r") as f:
        qois_names = f.readline().split()[1:]
    return qois_vals, qois_names


# -------------------------------------------------
# 获取某列 QOI
# -------------------------------------------------
def get(qois_vals, qois_names, key):
    return qois_vals[:, qois_names.index(key)]


# -------------------------------------------------
# 自动绘图函数（修正版本）
# -------------------------------------------------
def plot_curve(x, y, xlabel, ylabel, title, filename, marker="o-", color=None):

    # 创建 plots 文件夹
    os.makedirs("plots", exist_ok=True)

    # 绘图
    plt.figure(figsize=(6, 5))
    plt.plot(x, y, marker, color=color, linewidth=2)

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=17)

    plt.grid(ls="--", alpha=0.4)
    plt.tight_layout()

    # 正确的文件保存路径（修复）
    filepath = os.path.join("plots", filename)
    plt.savefig(filepath, bbox_inches="tight")

    print(f"Saved: {filepath}")

    plt.close()   # 不显示图像，也不占内存


# -------------------------------------------------
# 主函数
# -------------------------------------------------
def plot_selected(res_folder, res_basename):

    qois_filename = res_folder+"/"+res_basename+"-qois.dat"

    q, names = load_qois(qois_filename)

    needed = ["q_avg_x", "q_avg_y", "grad_p_bar_x", "grad_p_bar_y","U_bar_XX"]
    for key in needed:
        if key not in names:
            raise ValueError(f"Missing QOI: {key}")

    qx = get(q, names, "q_avg_x")
    qy = get(q, names, "q_avg_y")
    gx = get(q, names, "grad_p_bar_x")
    gy = get(q, names, "grad_p_bar_y")
    Uxx = get(q, names, "U_bar_XX")

    eps = 1e-12
    Kxx = -qx / (gx + eps)
    Kyy = -qy / (gy + eps)
    Kxy = -qx / (gy + eps)
    Kyx = -qy / (gx + eps)

    # 自动选择
    choices = ["1","5","11"]

    for c in choices:
        c = c.strip()

        # q_x vs grad p_x
        if c == "1":
            plot_curve(gx, qx,
                r"$\partial_x \bar p$", r"$\langle q_x\rangle$",
                "qx vs grad px", "qx_vs_gradp_x.png")

        # q_y vs grad p_y
        elif c == "2":
            plot_curve(gy, qy,
                r"$\partial_y \bar p$", r"$\langle q_y\rangle$",
                "qy vs grad py", "qy_vs_gradp_y.png")

        # q_x vs grad p_y
        elif c == "3":
            plot_curve(gy, qx,
                r"$\partial_y \bar p$", r"$\langle q_x\rangle$",
                "qx vs grad py", "qx_vs_gradp_y.png")

        # q_y vs grad p_x
        elif c == "4":
            plot_curve(gx, qy,
                r"$\partial_x \bar p$", r"$\langle q_y\rangle$",
                "qy vs grad px", "qy_vs_gradp_x.png")

        # ---- K curves ----
        elif c == "5":
            plot_curve(gx, Kxx,
                r"$\partial_x \bar p$", r"$K_{xx}$",
                "Kxx vs grad px", "Kxx_vs_gradp_x.png")

        elif c == "6":
            plot_curve(gy, Kyy,
                r"$\partial_y \bar p$", r"$K_{yy}$",
                "Kyy vs grad py", "Kyy_vs_gradp_y.png")

        elif c == "7":
            plot_curve(gy, Kxy,
                r"$\partial_y \bar p$", r"$K_{xy}$",
                "Kxy vs grad py", "Kxy_vs_gradp_y.png")

        elif c == "8":
            plot_curve(gx, Kyx,
                r"$\partial_x \bar p$", r"$K_{yx}$",
                "Kyx vs grad px", "Kyx_vs_gradp_x.png")
            
        elif c == "11":
            plot_curve(Uxx, Kxx,
                r"$U_{\bar XX}$", r"$K_{xx}$",
                "Kxx vs U_bar_XX",
                "Kxx_vs_UbarXX.png")


        else:
            print(f"Unknown option: {c}")


# -------------------------------------------------
# Script entry
# -------------------------------------------------
if __name__ == "__main__":
    plot_selected(res_folder, res_basename)

