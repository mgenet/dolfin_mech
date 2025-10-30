import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
C = 1.0          # Kozeny–Carman constant
Phi_s0 = 0.7     # reference solid volume fraction
nu = 0.3         # Poisson ratio

# Stretch ratios in x-direction (uniaxial)
lambda_x = np.linspace(0.6, 1.5, 300)   # compression → tension

# Green–Lagrange strain (x-direction)
E_x = 0.5 * (lambda_x**2 - 1.0)

# ----------------------------------------------------------------------
# Finite strain quantities with Poisson effect
# ----------------------------------------------------------------------
lambda_y = lambda_x**(-nu / (1 - nu))
lambda_z = lambda_y.copy()
J = lambda_x * lambda_y * lambda_z     # det(F)

# ----------------------------------------------------------------------
# Deformation-dependent permeability
# ----------------------------------------------------------------------
def k_l(J, Phi_s):
    """Kozeny–Carman-type permeability depending on J and Phi_s."""
    return C * ((J - Phi_s)**3) / (J * Phi_s**2)

k_E = k_l(J, Phi_s0)                   # permeability vs deformation

# ----------------------------------------------------------------------
# Porosity evolution
# ----------------------------------------------------------------------
phi_l = 1.0 - Phi_s0 / J               # current liquid volume fraction (Eulerian)
Phi_l = J * phi_l                      # pulled-back (Lagrangian) porosity

# ----------------------------------------------------------------------
# Also check k vs Phi_s for fixed deformation (optional)
# ----------------------------------------------------------------------
Phi_s_range = np.linspace(0.4, 0.9, 300)
J_fixed = 1.2
k_Phi = k_l(J_fixed, Phi_s_range)

# ----------------------------------------------------------------------
# Plot results
# ----------------------------------------------------------------------
plt.figure(figsize=(14,5))

# ---- (1) k vs E_x ----
plt.subplot(1,3,1)
plt.plot(E_x, k_E, lw=2)
plt.xlabel(r'Green--Lagrange strain $E_x$')
plt.ylabel(r'$k_\ell$')
plt.title(r'$k_\ell$ vs. strain $E_x$' + f'\n($\Phi_s$={Phi_s0}, $\nu$={nu})')
plt.grid(True)

# ---- (2) phi_l vs E_x ----
plt.subplot(1,3,2)
plt.plot(E_x, phi_l, 'g', lw=2)
plt.xlabel(r'Green--Lagrange strain $E_x$')
plt.ylabel(r'$\phi_\ell$ (current liquid fraction)')
plt.title(r'Porosity $\phi_\ell$ vs. strain $E_x$')
plt.grid(True)

# ---- (3) k vs phi_l ----
plt.subplot(1,3,3)
plt.plot(phi_l, k_E, 'r', lw=2)
plt.xlabel(r'$\phi_\ell$')
plt.ylabel(r'$k_\ell$')
plt.title(r'$k_\ell$ vs. $\phi_\ell$')
plt.grid(True)

plt.tight_layout()
plt.show()
