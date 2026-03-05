
# -------------------------------------------------
# For plotting
# -------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import sys  
# pf_values = [0.0, 0.03,0.06]
pf_values = [0]
#res_folder = sys.argv[0][:-3]
res_folder = "/Users/xiao/PhD/dolfin_mech_HX2/dolfin_mech/run_MicroPoroflow"

def load_qois(qois_filename):
    qois_vals = np.loadtxt(qois_filename)
    with open(qois_filename, "r") as f:
        qois_names = f.readline().split()[1:]
    return qois_vals, qois_names

def get(qois_vals, qois_names, key):
    return qois_vals[:, qois_names.index(key)]

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_K_vs_pg_multi_Ex(
    res_folder,
    res_basename_prefix,
    Ex_list,
    slice_start=0,
    eps=1e-12,
    pg_key="p_f",       
    gx_key="grad_p_bar_x",
    gy_key="grad_p_bar_y",
    qx_key="q_avg_x",
    qy_key="q_avg_y",
    pg_in_kPa=True,
    savepath="plots/K_vs_pf_multi_Ex.png",
):
    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    colors = [
        ("#1f77b4", "#aec7e8"),
        ("#d62728", "#ff9896"),
        ("#2ca02c", "#98df8a"),
        ("#9467bd", "#c5b0d5"),
        ("#8c564b", "#c49c94"),
    ]

    for idx, Ex in enumerate(Ex_list):
        filename = f"{res_folder}/{res_basename_prefix}-Ex={Ex}-qois.dat"
        if not os.path.exists(filename):
            print(f"[WARNING] File missing: {filename}")
            continue

        qois_vals, names = load_qois(filename)

        pf = get(qois_vals, names, pg_key)[slice_start:].astype(float)
        if not pg_in_kPa:
            pf = pf / 1000.0

        gx = get(qois_vals, names, gx_key)[slice_start:].astype(float)
        gy = get(qois_vals, names, gy_key)[slice_start:].astype(float)
        qx = get(qois_vals, names, qx_key)[slice_start:].astype(float)
        qy = get(qois_vals, names, qy_key)[slice_start:].astype(float)

        Kxx = -qx / (gx + eps)
        Kyy = -qy / (gy + eps)

        order = np.argsort(pf)
        pf, Kxx, Kyy = pf[order], Kxx[order], Kyy[order]

        c_dark, c_light = colors[idx % len(colors)]
        ax.plot(pf, Kxx, color=c_dark,  lw=2.6, label=rf"$\tilde{{K}}_{{xx}}$, $E_x={Ex}$")
        ax.plot(pf, Kyy, color=c_light, lw=2.6, label=rf"$\tilde{{K}}_{{yy}}$, $E_x={Ex}$")

        print(f"Read: {os.path.basename(filename)}  points={len(pf)}")

    ax.set_xlabel(r"$p_f\,(kPa)$", fontsize=16)  # 你这里写 p_f 更一致
    ax.set_ylabel(r"$\tilde{K}_{xx},\,\tilde{K}_{yy}\,(m^2/(Pa\cdot s))$", fontsize=16)
    #ax.grid(ls="--", alpha=0.4)
    ax.legend(fontsize=11, framealpha=0.9, loc="upper left")

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight",dpi=300)
    plt.close()
    print(f"Saved: {savepath}")


def plot_q_vs_gradp_multi_pf(res_folder, pf_list, res_basename_prefix, k_hom=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs("plots", exist_ok=True)

    colors = [
        ("#1f77b4", "#aec7e8"),
        ("#d62728", "#ff9896"),
        ("#2ca02c", "#98df8a"),
        ("#9467bd", "#c5b0d5"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axx, axy = axes

    gx_all, qx_all, gy_all, qy_all = [], [], [], []

    for idx, pf in enumerate(pf_list):
        filename = f"{res_folder}/{res_basename_prefix}-pf={pf}-qois.dat"
        if not os.path.exists(filename):
            print(f"[WARNING] File missing: {filename}")
            continue

        qois_vals, names = load_qois(filename)

        qx = get(qois_vals, names, "q_avg_x")[2:]
        qy = get(qois_vals, names, "q_avg_y")[2:]
        gx = get(qois_vals, names, "grad_p_bar_x")[2:]
        gy = get(qois_vals, names, "grad_p_bar_y")[2:]

        # store for reference line range
        gx_all.append(gx); qx_all.append(qx)
        gy_all.append(gy); qy_all.append(qy)

        c_dark, c_light = colors[idx % len(colors)]

        axx.plot(gx, qx, color=c_dark, linewidth=2.5, label=rf"$p_f={pf}$")
        axy.plot(gy, qy, color=c_light, linewidth=2.5, label=rf"$p_f={pf}$")

    # ---- add theoretical reference lines ----
    if k_hom is not None:
        k_hom = np.asarray(k_hom, dtype=float)
        kxx = k_hom[0, 0]
        kyy = k_hom[1, 1]

        # pick a reasonable x-range from all datasets
        if len(gx_all) > 0:
            gx_min = min([np.min(g) for g in gx_all])
            gx_max = max([np.max(g) for g in gx_all])
            gx_ref = np.linspace(gx_min, gx_max, 200)
            axx.plot(gx_ref, -kxx * gx_ref, "k--", linewidth=2.0,
                     label=rf"linear model: $q_x=-k_{{xx}}\nabla\bar p_x$ ($k_{{xx}}={kxx:.3g}$)")

        if len(gy_all) > 0:
            gy_min = min([np.min(g) for g in gy_all])
            gy_max = max([np.max(g) for g in gy_all])
            gy_ref = np.linspace(gy_min, gy_max, 200)
            axy.plot(gy_ref, -kyy * gy_ref, "k--", linewidth=2.0,
                     label=rf"linear model: $q_y=-k_{{yy}}\nabla\bar p_y$ ($k_{{yy}}={kyy:.3g}$)")

    axx.set_xlabel(r"$\nabla \bar{p}_x$", fontsize=16)
    axx.set_ylabel(r"$q_x$", fontsize=16)
    axx.grid(ls="--", alpha=0.4)
    axx.legend(fontsize=11, framealpha=0.9)

    axy.set_xlabel(r"$\nabla \bar{p}_y$", fontsize=16)
    axy.set_ylabel(r"$q_y$", fontsize=16)
    axy.grid(ls="--", alpha=0.4)
    axy.legend(fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("plots/q_vs_gradp_multi_pf.png", bbox_inches="tight")
    plt.close()
    print("Saved: plots/q_vs_gradp_multi_pf.png")


def plot_Kxx_Kyy_vs_Uxx_multi_pf(
    res_folder,
    pf_list,
    res_basename_prefix,
    K0_ref=None,          # 2x2 reference permeability tensor (reference config)
    slice_start=6,
    eps=1e-12,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs("plots", exist_ok=True)

    colors = [
        ("#1f77b4", "#aec7e8"),
        ("#d62728", "#ff9896"),
        ("#2ca02c", "#98df8a"),
        ("#9467bd", "#c5b0d5"),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    # sanitize K0_ref
    if K0_ref is not None:
        K0_global = np.asarray(K0_ref, dtype=float)
        if K0_global.shape != (2, 2):
            raise ValueError(f"K0_ref must be shape (2,2) in 2D, got {K0_global.shape}")
    else:
        K0_global = None

    for idx, pf in enumerate(pf_list):
        filename = f"{res_folder}/{res_basename_prefix}-pf={pf}-qois.dat"
        if not os.path.exists(filename):
            print(f"[WARNING] File missing: {filename}")
            continue

        qois_vals, names = load_qois(filename)

        # --- macro stretch components ---
        Uxx = get(qois_vals, names, "U_bar_XX")[slice_start:]
        Uyy = get(qois_vals, names, "U_bar_YY")[slice_start:]
        Uxy = get(qois_vals, names, "U_bar_XY")[slice_start:]  # assume symmetry: Uyx=Uxy

        # --- Darcy outputs (NEW names) ---
        # Q_l_avg_* are Piola/reference flux components
        Qx = get(qois_vals, names, "Q_l_avg_x")[slice_start:]
        Qy = get(qois_vals, names, "Q_l_avg_y")[slice_start:]
        gx = get(qois_vals, names, "grad_p_bar_avg_x")[slice_start:]
        gy = get(qois_vals, names, "grad_p_bar_avg_y")[slice_start:]

        Uxx = np.asarray(Uxx, dtype=float)
        Uyy = np.asarray(Uyy, dtype=float)
        Uxy = np.asarray(Uxy, dtype=float)
        Qx  = np.asarray(Qx,  dtype=float)
        Qy  = np.asarray(Qy,  dtype=float)
        gx  = np.asarray(gx,  dtype=float)
        gy  = np.asarray(gy,  dtype=float)

        # --- "measured" reference permeability from Piola flux ---
        # Q = - K_ref * grad_X(p)  =>  K_ref,xx ~ -Qx/gx  (component-wise)
        Kxx_ref = -Qx / (gx + eps)
        Kyy_ref = -Qy / (gy + eps)

        # choose colors
        c_dark, c_light = colors[idx % len(colors)]

        # plot measured
        ax.plot(
            Uxx, Kxx_ref,
            color=c_dark, linewidth=2.5,
            label=rf"$K_{{xx}}^{{ref}}$, $p_f={pf}\,\mathrm{{kPa}}$"
        )
        ax.plot(
            Uxx, Kyy_ref,
            color=c_light, linewidth=2.5,
            label=rf"$K_{{yy}}^{{ref}}$, $p_f={pf}\,\mathrm{{kPa}}$"
        )

        ax.set_xlabel(r"$U_{\bar{XX}}$", fontsize=14)
        ax.set_ylabel(
            r"$K_{xx}^{ref},\,K_{yy}^{ref}\;(\mathrm{m}^2\,\mathrm{Pa}^{-1}\,\mathrm{s}^{-1})$",
            fontsize=14
        )

        # --- reference K0 for prediction (optional) ---
        if K0_global is not None:
            K0 = K0_global
        else:
            # fallback: diagonal K0 from first point (keeps line meaningful)
            K0 = np.array([[float(Kxx_ref[0]), 0.0],
                           [0.0,             float(Kyy_ref[0])]], dtype=float)

        # If you later want to compare with a predicted pull-back:
        # K_pred = J * F^{-1} * k_intr * F^{-T}  (or any other model)
        # You can add dashed lines here.

        print(f"pf={pf}: K0 used for pred =\n{K0}")

    ax.legend(fontsize=9.5, framealpha=0.9, ncol=1)
    plt.tight_layout()
    plt.savefig("plots/Kxx_Kyy_vs_Uxx_multi_pf.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved: plots/Kxx_Kyy_vs_Uxx_multi_pf.png")


if __name__ == "__main__":

    pf_list = pf_values  
    res_basename_prefix = "-dim=2-bcs=pbc-load=K_vs_U"
    res_basename_prefix_pf = "-dim=2-bcs=pbc-load=K_vs_pf"
    # for imported rve mesh
    # k_hom = [
    #     [6.88814361e-01, 7.48353084e-07],
    #     [7.48353084e-07, 6.88804851e-01],
    # ]
    # k_hom = [
    #     [ 9.99997937e-01, 5.36264851e-11],
    #     [5.36264851e-11,  9.99997937e-01]
    # ]
    k_hom = [[6.17843158e-16, 0.00000000e+00],
              [0.00000000e+00, 6.03204544e-16]]

    plot_Kxx_Kyy_vs_Uxx_multi_pf(res_folder, pf_list, res_basename_prefix)
    #plot_q_vs_gradp_multi_pf(res_folder, pf_list, res_basename_prefix, k_hom=None)
    #plot_K_vs_pg_multi_Ex(res_folder,  res_basename_prefix_pf, Ex_list=[0.0, 0.1, 0.2], slice_start=3, pg_in_kPa=True)


