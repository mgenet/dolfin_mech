import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ paths ------------------ #
base_dir = Path("/Users/xiao/PhD/dolfin_mech_HX2/dolfin_mech")
dir_circle = base_dir / "test_MicroPoroflow_circle"
dir_hex    = base_dir / "test_MicroPoroflow_hex"

# ------------------ options ------------------ #
USE_FIXED_GRAD = False
gx_fixed = 1e-3
gy_fixed = 1e-3

# ---- theory ----
Km = 1.0  # matrix permeability used in dilute/differential curves

# If your phi_f is actually the GAS/void fraction used in theory, keep True
PHI_IS_GAS_FRACTION = True

# ------------------ helpers ------------------ #
def parse_phi_f_from_filename(name: str):
    # matches: ...-phi=0.011900-...
    m = re.search(r"-phi=([0-9]*\.?[0-9]+)-", name)
    return float(m.group(1)) if m else None

def load_one_folder(qois_dir: Path, label: str) -> pd.DataFrame:
    qois_files = sorted(qois_dir.glob("*-qois.dat"))
    if not qois_files:
        raise RuntimeError(f"No *-qois.dat found in {qois_dir}")

    cols = [
        "t", "q_avg_x", "q_avg_y", "grad_p_bar_x", "grad_p_bar_y",
        "p_tilde_avg_current", "vs", "vf", "v",
        "U_bar_XX", "U_bar_YY", "U_bar_XY", "U_bar_YX",
        "sigma_s_bar_XX", "sigma_s_bar_YY", "sigma_s_bar_XY", "sigma_s_bar_YX",
        "sigma_bar_XX", "sigma_bar_YY", "sigma_bar_XY", "sigma_bar_YX",
        "p_f", "S_area",
    ]

    rows = []
    skipped = []

    for f in qois_files:
        phi_f = parse_phi_f_from_filename(f.name)
        if phi_f is None:
            skipped.append((f.name, "phi not found in filename"))
            continue

        df = pd.read_csv(f, delim_whitespace=True, comment="#", header=None)

        if df.shape[1] != len(cols):
            skipped.append((f.name, f"column mismatch: got {df.shape[1]}, expected {len(cols)}"))
            continue

        df.columns = cols
        last = df.loc[df["t"].idxmax()]

        qx = float(last["q_avg_x"])
        qy = float(last["q_avg_y"])

        if USE_FIXED_GRAD:
            gx = gx_fixed
            gy = gy_fixed
        else:
            gx = float(last["grad_p_bar_x"])
            gy = float(last["grad_p_bar_y"])

        if abs(gx) < 1e-30 or abs(gy) < 1e-30:
            skipped.append((f.name, f"grad too small: gx={gx}, gy={gy}"))
            continue

        # Darcy: q = -K grad(p)
        Kxx = -qx / gx
        Kyy = -qy / gy
        Keq = 0.5 * (Kxx + Kyy)

        rows.append({
            "shape": label,
            "file": f.name,
            "phi_f": phi_f,
            "t": float(last["t"]),
            "q_avg_x": qx,
            "q_avg_y": qy,
            "grad_p_bar_x": gx,
            "grad_p_bar_y": gy,
            "Kxx": Kxx,
            "Kyy": Kyy,
            "Keq": Keq,
        })

    if not rows:
        print(f"[{label}] Skipped examples (up to 10):")
        for s in skipped[:10]:
            print("  ", s)
        raise RuntimeError(f"[{label}] No valid rows parsed. Check qois.dat format / filename phi pattern.")

    out = pd.DataFrame(rows).sort_values("phi_f").reset_index(drop=True)

    if skipped:
        print(f"\n[WARN] [{label}] Skipped files (up to 10):")
        for s in skipped[:10]:
            print("  ", s)

    return out

def make_theory(phi_vals: np.ndarray):
    """
    Build theory curves as a function of 'f' used in formulas.
    dilute:       K = Km*(1 - 2f)  (stop at 0)
    differential: K = Km*(1 - f)^2
    """
    if PHI_IS_GAS_FRACTION:
        f = phi_vals.copy()
    else:
        # if phi_vals is FLUID fraction, and gas fraction is (1-phi)
        f = 1.0 - phi_vals

    K_dilute = Km * (1.0 - 2.0 * f)
    K_diff   = Km * (1.0 - f) ** 2

    # stop dilute at zero: only keep the part where K_dilute > 0
    mask = K_dilute > 0.0
    return f, K_dilute, K_diff, mask

# ------------------ load data ------------------ #
circle = load_one_folder(dir_circle, "circle")
hexagon = load_one_folder(dir_hex, "hex")

all_df = pd.concat([circle, hexagon], ignore_index=True).sort_values(["shape", "phi_f"]).reset_index(drop=True)

# Save combined CSV (optional)
out_csv = base_dir / "K_vs_Phi_f_circle_hex.csv"
all_df.to_csv(out_csv, index=False)
print("Saved:", out_csv)

# ------------------ plotting ------------------ #
# theory x-grid: use union of both datasets, or a dense linspace for smooth curves
phi_min = float(all_df["phi_f"].min())
phi_max = float(all_df["phi_f"].max())
phi_grid = np.linspace(phi_min, phi_max, 300)

f_grid, K_dilute_grid, K_diff_grid, mask_dilute = make_theory(phi_grid)

# ---- Plot Keq + theory (single figure) ----
plt.figure()

# circle
plt.plot(circle["phi_f"], circle["Keq"], marker="o", linestyle="-", label="circle: Keq")
# hex
plt.plot(hexagon["phi_f"], hexagon["Keq"], marker="s", linestyle="-", label="hex: Keq")

# theory
plt.plot(phi_grid[mask_dilute], K_dilute_grid[mask_dilute], "--", label="dilute (Km=1): 1-2f (stop at 0)")
plt.plot(phi_grid, K_diff_grid, "--", label="differential (Km=1): (1-f)^2")

plt.xlabel("Phi_f")
plt.ylabel("K")
plt.grid(True)
plt.legend()
plt.tight_layout()

fig_path = base_dir / "Keq_circle_hex_vs_Phi_f_with_theory.png"
plt.savefig(fig_path, dpi=200)
print("Saved figure:", fig_path)

# ---- Optional: Plot Kxx and Kyy (two datasets + theory) ----
plt.figure()

plt.plot(circle["phi_f"], circle["Kxx"]*(1 - circle["phi_f"]), linestyle="-", label="circle: Kxx,Kyy")
#plt.plot(circle["phi_f"], circle["Kyy"]*(1 - circle["phi_f"]), marker="o", linestyle="--", label="circle: Kyy")

plt.plot(hexagon["phi_f"], hexagon["Kxx"]*(1 - hexagon["phi_f"]), linestyle="-", label="hex: Kxx,Kyy")
#plt.plot(hexagon["phi_f"], hexagon["Kyy"]*(1 - hexagon["phi_f"]), marker="s", linestyle="--", label="hex: Kyy")

plt.plot(phi_grid[mask_dilute], K_dilute_grid[mask_dilute], ":", label="dilute (stop at 0)")
plt.plot(phi_grid, K_diff_grid, ":", label="differential")

plt.xlabel("Phi_f")
plt.ylabel("K")
#plt.grid(True)
plt.legend()
plt.tight_layout()

fig_path2 = base_dir / "Kxx_Kyy_circle_hex_vs_Phi_f_with_theory.png"
plt.savefig(fig_path2, dpi=200)
print("Saved figure:", fig_path2)