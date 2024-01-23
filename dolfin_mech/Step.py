#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech

################################################################################

class Step():



    def __init__(self,
            t_ini=0.,
            t_fin=1.,
            dt_ini=None,
            dt_min=None,
            dt_max=None,
            operators=None,    # MG20180508: Do not use list as default value because it is static
            constraints=None): # MG20180508: Do not use list as default value because it is static

        self.t_ini = t_ini
        self.t_fin = t_fin

        self.dt_ini = dt_ini if (dt_ini is not None) else self.t_fin - self.t_ini
        self.dt_min = dt_min if (dt_min is not None) else self.dt_ini
        self.dt_max = dt_max if (dt_max is not None) else self.dt_ini

        self.operators   = operators   if (operators   is not None) else []
        self.constraints = constraints if (constraints is not None) else []
