#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech

################################################################################

class Operator():

    def set_value_at_t_step(self, *args, **kwargs): pass
    def set_dt(self, *args, **kwargs): pass
