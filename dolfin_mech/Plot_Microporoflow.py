
# -------------------------------------------------
# For plotting
# -------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import sys  
pf_values = [0.0, 0.03,0.06]

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
    k0_hom=None,          # 2x2 reference permeability tensor in reference config
    slice_start=3,        # keep consistent with your [2:] slicing
    eps=1e-12,
    use_F_equals_U=True,  # if True: F = U_bar ; otherwise you can modify later
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

    # sanitize k0_hom
    if k0_hom is not None:
        K0_global = np.asarray(k0_hom, dtype=float)
        if K0_global.shape != (2, 2):
            raise ValueError(f"k0_hom must be shape (2,2) in 2D, got {K0_global.shape}")
    else:
        K0_global = None

    for idx, pf in enumerate(pf_list):
        filename = f"{res_folder}/{res_basename_prefix}-pf={pf}-qois.dat"
        if not os.path.exists(filename):
            print(f"[WARNING] File missing: {filename}")
            continue

        qois_vals, names = load_qois(filename)

        # --- macro stretch components (you said you have YY and XY) ---
        Uxx = get(qois_vals, names, "U_bar_XX")[slice_start:]
        Uyy = get(qois_vals, names, "U_bar_YY")[slice_start:]
        Uxy = get(qois_vals, names, "U_bar_XY")[slice_start:]  # assume symmetric stretch, so Uyx=Uxy

        # --- flow outputs ---
        qx  = get(qois_vals, names, "q_avg_x")[slice_start:]
        qy  = get(qois_vals, names, "q_avg_y")[slice_start:]
        gx  = get(qois_vals, names, "grad_p_bar_x")[slice_start:]
        gy  = get(qois_vals, names, "grad_p_bar_y")[slice_start:]

        Uxx = np.asarray(Uxx, dtype=float)
        Uyy = np.asarray(Uyy, dtype=float)
        Uxy = np.asarray(Uxy, dtype=float)
        qx  = np.asarray(qx,  dtype=float)
        qy  = np.asarray(qy,  dtype=float)
        gx  = np.asarray(gx,  dtype=float)
        gy  = np.asarray(gy,  dtype=float)

        # --- "measured" K from q/g (component-wise) ---
        Kxx = -qx / (gx + eps)
        Kyy = -qy / (gy + eps)

        # choose colors
        c_dark, c_light = colors[idx % len(colors)]

        # plot measured
        ax.plot(
            Uxx, Kxx,
            color=c_dark, linewidth=2.5,
            label=rf"$\tilde{{K}}_{{xx}}$, $p_g={pf}\,\mathrm{{kPa}}$"
        )

        ax.plot(
            Uxx, Kyy,
            color=c_light, linewidth=2.5,
            label=rf"$\tilde{{K}}_{{yy}}$, $p_g={pf}\,\mathrm{{kPa}}$"
        )


        ax.set_xlabel(r"$E_x\;()$", fontsize=14)
        ax.set_ylabel(
            r"$\tilde{K}_{xx},\,\tilde{K}_{yy}\;(\mathrm{m}^2\,\mathrm{Pa}^{-1}\,\mathrm{s}^{-1})$",
            fontsize=14
        )



        # --- build K0 for prediction ---
        if K0_global is not None:
            K0 = K0_global
        else:
            # fallback: use first point to define a diagonal reference K0
            # (keeps the reference line meaningful even if you didn't pass k0_hom)
            k0_x = float(Kxx[0])
            k0_y = float(Kyy[0])
            K0 = np.array([[k0_x, 0.0],
                           [0.0,  k0_y]], dtype=float)

        # --- predicted k via push-forward: k_pred = (1/J) F K0 F^T ---
        Kxx_pred = np.zeros_like(Uxx)
        Kyy_pred = np.zeros_like(Uxx)

        for n in range(len(Uxx)):
            # F = I + U_bar  (because u_bar = U_bar * (X - X0))
            F = np.array([[1.0 + Uxx[n],       Uxy[n]],
                        [      Uxy[n], 1.0 + Uyy[n]]], dtype=float)

            J = float(np.linalg.det(F))
            Finv = np.linalg.inv(F)
            if abs(J) < 1e-15:
                Kxx_pred[n] = np.nan
                Kyy_pred[n] = np.nan
                continue

            K_back = J * (Finv @ K0 @ Finv.T)

            k_pred = (1.0 / J) * (F @ K0 @ F.T)

            Kxx_pred[n] = K_back[0, 0]
            Kyy_pred[n] = K_back[1, 1]

        # plot predicted (dashed)
        # ax.plot(Uxx, Kxx_pred, color=c_dark,  ls="--", linewidth=2.0,
        #         label=rf"$\frac{{1}}{{J}}(F K_0 F^T)_{{xx}}$, $p_f={pf}$")
        # ax.plot(Uxx, Kyy_pred, color=c_light, ls="--", linewidth=2.0,
        #         label=rf"$\frac{{1}}{{J}}(F K_0 F^T)_{{yy}}$, $p_f={pf}$")

        # optional quick print
        print(f"pf={pf}: K0 used for pred =\n{K0}")

    # ax.set_xlabel(r"$U_{\bar{XX}}$", fontsize=16)
    # ax.set_ylabel(r"$K_{xx}, K_{yy}$", fontsize=16)
    # ax.grid(ls="--", alpha=0.4)

    # legend can get big; make it compact
    ax.legend(fontsize=9.5, framealpha=0.9, ncol=1)

    plt.tight_layout()
    plt.savefig("plots/Kxx_Kyy_vs_Uxx_multi_pf.png", bbox_inches="tight",dpi=300)
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
    k_hom = [
        [ 9.99997937e-01, 5.36264851e-11],
        [5.36264851e-11,  9.99997937e-01]
    ]

    #plot_Kxx_Kyy_vs_Uxx_multi_pf(res_folder, pf_list, res_basename_prefix,k0_hom=None)
    #plot_q_vs_gradp_multi_pf(res_folder, pf_list, res_basename_prefix, k_hom=None)
    plot_K_vs_pg_multi_Ex(res_folder,  res_basename_prefix_pf, Ex_list=[0.0, 0.1, 0.2], slice_start=3, pg_in_kPa=True)


