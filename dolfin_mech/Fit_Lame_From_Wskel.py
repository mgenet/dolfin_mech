import dolfin
import numpy as np
import sys
from pathlib import Path
local_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(local_path))
from fenics import *
import dolfin_mech as dmech
# ---- minimal kinematics: reference configuration ----

def lame_to_Enu_3D(lam, mu):
    lam = float(lam); mu = float(mu)
    nu = lam / (2.0 * (lam + mu))
    E  = mu * (3.0*lam + 2.0*mu) / (lam + mu)
    return {"E": E, "nu": nu}

def fit_lame_from_wskel(dim, material_parameters, eps0=1e-7):

    if dim == 2:
        mesh = dolfin.UnitSquareMesh(1, 1)
    elif dim == 3:
        mesh = dolfin.UnitCubeMesh(1, 1, 1)
    else:
        raise ValueError("dim must be 2 or 3")

    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    X = dolfin.SpatialCoordinate(mesh)
    dx = dolfin.Measure("dx", domain=mesh)
    vol = dolfin.assemble(dolfin.Constant(1.0) * dx)

    def set_affine_u(A):
        A = np.asarray(A, dtype=float)
        u_ufl = dolfin.as_vector([
            sum(dolfin.Constant(A[i, j]) * X[j] for j in range(dim))
            for i in range(dim)
        ])
        return dolfin.project(u_ufl, V)

    def avg_S(u):
        kin = dmech.Kinematics(U=u) 
        solid = dmech.WskelLungElasticMaterial(kinematics=kin, parameters=dict(material_parameters))
        S = solid.Sigma
        S_avg = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            for j in range(dim):
                S_avg[i, j] = dolfin.assemble(S[i, j] * dx) / vol
        return S_avg

    e = float(eps0)

    # symmetric shear -> mu
    g = e
    A_sh = np.zeros((dim, dim))
    A_sh[0, 1] = 0.5 * g
    A_sh[1, 0] = 0.5 * g
    S_sh = avg_S(set_affine_u(A_sh))
    mu = S_sh[0, 1] / g

    # uniaxial strain -> lambda
    A_u = np.zeros((dim, dim))
    A_u[0, 0] = e
    S_u = avg_S(set_affine_u(A_u))
    lam = S_u[1, 1] / e

    # checks
    Sxx_pred = (lam + 2.0 * mu) * e
    Sxx_err = float(S_u[0, 0] - Sxx_pred)
    Sxy_err = float(S_sh[0, 1] - mu * g)

    return float(lam), float(mu), {
        "S_uniax_strain": S_u,
        "S_sym_shear": S_sh,
        "Sxx_err": Sxx_err,
        "Sxy_err": Sxy_err,
        "eps0": e,
    }




mat_params = {
    "alpha": 0.16,
    "gamma": 0.5,
    "c1": 0.2,
    "c2": 0.4,
    "kappa": 1e2,
    "eta": 1e-5,
}

dim = 2
for eps0 in [1e-8, 3e-8, 1e-7, 3e-7, 1e-6]:
    lam, mu, dbg = fit_lame_from_wskel(dim, mat_params, eps0=eps0)
    print(f"eps0={eps0:.1e}  lambda={lam:.6e}  mu={mu:.6e}  "
          f"Sxx_err={dbg['Sxx_err']:.2e}  Sxy_err={dbg['Sxy_err']:.2e}")

# one detailed print at your preferred eps0
eps0 = 1e-7
lam, mu, dbg = fit_lame_from_wskel(dim, mat_params, eps0=eps0)
print("\n=== Detailed output ===")
print(f"dim   = {dim}")
print(f"eps0  = {dbg['eps0']:.3e}")
print(f"lambda= {lam:.6e}")
print(f"mu    = {mu:.6e}")
print("\nS (uniaxial strain):\n", dbg["S_uniax_strain"])
print("\nS (symmetric shear):\n", dbg["S_sym_shear"])
print(f"\nCheck: Sxx_err = {dbg['Sxx_err']:.6e}")
print(f"Check: Sxy_err = {dbg['Sxy_err']:.6e}\n")

import numpy as np
import matplotlib.pyplot as plt

dim = 2


eps_list = np.logspace(-10, -1.7, 30)

lam_list = []
mu_list  = []
Sxxerr_list = []
Sxyerr_list = []

for eps0 in eps_list:
    lam, mu, dbg = fit_lame_from_wskel(dim, mat_params, eps0=float(eps0))
    lam_list.append(lam)
    mu_list.append(mu)
    Sxxerr_list.append(dbg["Sxx_err"])
    Sxyerr_list.append(dbg["Sxy_err"])

lam_list = np.array(lam_list)
mu_list  = np.array(mu_list)

lam0 = lam_list[0]
mu0  = mu_list[0]
rel_lam = (lam_list - lam0) / abs(lam0)
rel_mu  = (mu_list  - mu0 ) / abs(mu0 )

# # ---- plot 1: lambda, mu vs eps ----
# plt.figure()
# plt.semilogx(eps_list, lam_list, marker='o', label='lambda')
# plt.semilogx(eps_list, mu_list,  marker='s', label='mu')
# plt.xlabel('eps0')
# plt.ylabel('fitted coefficient')
# plt.grid(True, which='both')
# plt.legend()
# plt.tight_layout()

# # ---- plot 2: relative deviation from linear baseline ----
# plt.figure()
# plt.semilogx(eps_list, rel_lam, marker='o', label='(lambda-lambda0)/|lambda0|')
# plt.semilogx(eps_list, rel_mu,  marker='s', label='(mu-mu0)/|mu0|')
# plt.xlabel('eps0')
# plt.ylabel('relative deviation')
# plt.grid(True, which='both')
# plt.legend()
# plt.tight_layout()

# plt.show()
Enu = lame_to_Enu_3D(lam, mu)
print("Equivalent (3D) Enu =", Enu)

import numpy as np
import matplotlib.pyplot as plt

# -------- parameters (edit these) --------
kappa = 1.0      # your kappa
phis0 = 0.2      # Phis0 (reference solid fraction)
delta = 0.02     # Delta Phis (try 0.005, 0.01, 0.02, 0.04, ...)
# ----------------------------------------

def dWbulkdPhis(phis, kappa, phis0):
    # dW/dPhis = kappa * (1/Phis0 - 1/Phis)
    return kappa * (1.0/phis0 - 1.0/phis)

# Phis range (avoid 0 to prevent singularity)
phis_min = 0.05
phis_max = 0.6
phis = np.linspace(phis_min, phis_max, 1000)

# Ensure perturbed Phis stays positive
phis_pert = phis + delta
mask = phis_pert > 1e-12
phis_plot = phis[mask]
phis_pert_plot = phis_pert[mask]

q = dWbulkdPhis(phis_plot, kappa, phis0)
q_pert = dWbulkdPhis(phis_pert_plot, kappa, phis0)

dq = q_pert - q
rel = dq / np.maximum(np.abs(q), 1e-14)  # relative change, avoid divide by 0

# ---- plot 1: dW/dPhis curves ----
plt.figure()
plt.plot(phis_plot, q, label="dWbulkdPhis(Phis)")
plt.plot(phis_plot, q_pert, label=f"dWbulkdPhis(Phis + {delta:g})", linestyle="--")
plt.axvline(phis0, linestyle=":", label="Phis0")
plt.xlabel("Phis")
plt.ylabel("dWbulkdPhis")
plt.legend()
plt.title("dWbulkdPhis vs Phis (with perturbation)")
plt.grid(True, alpha=0.3)

# ---- plot 2: absolute change ----
# plt.figure()
# plt.plot(phis_plot, dq, label="Delta(dWbulkdPhis)")
# plt.xlabel("Phis")
# plt.ylabel("dWbulkdPhis(Phis+delta) - dWbulkdPhis(Phis)")
# plt.legend()
# plt.title("Absolute impact of delta on dWbulkdPhis")
# plt.grid(True, alpha=0.3)

# ---- plot 3: relative change ----
# plt.figure()
# plt.plot(phis_plot, rel, label="Relative change")
# plt.xlabel("Phis")
# plt.ylabel("Delta / |base|")
# plt.legend()
# plt.title("Relative impact of delta on dWbulkdPhis")
# plt.grid(True, alpha=0.3)

plt.show()

# ---- quick sensitivity number at a point (optional) ----
# pick a representative point near your average, e.g. Phis=0.2
phis_star = 0.2
q_star = dWbulkdPhis(phis_star, kappa, phis0)
q_star_pert = dWbulkdPhis(phis_star + delta, kappa, phis0)
print("At Phis =", phis_star)
print("dWbulkdPhis =", q_star)
print("dWbulkdPhis(Phis+delta) =", q_star_pert)
print("Absolute change =", q_star_pert - q_star)
print("Relative change =", (q_star_pert - q_star) / (abs(q_star) + 1e-14))
