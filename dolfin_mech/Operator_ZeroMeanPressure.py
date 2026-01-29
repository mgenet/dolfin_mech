import dolfin

import dolfin_mech as dmech
from .Operator import Operator

################################################################################

class ZeroMeanPressureOperator(Operator):

    def __init__(self,
            p, p_test,
            lam, lam_test,
            measure,
            J=None):

        self.measure = measure
        self.J = J  # optional weight (current configuration)
        V0 = dolfin.assemble(1.0 * measure)

        weight = 1.0 if (J is None) else J

        # enforce: ∫ p * weight dV = 0
        self.res_form  = lam_test * (p      / V0) * self.measure
        self.res_form += lam      * (p_test / V0) * self.measure
