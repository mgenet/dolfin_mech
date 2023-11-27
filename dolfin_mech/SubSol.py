#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_mech as dmech

################################################################################

class SubSol():



    def __init__(self,
            name,
            fe,
            init_val=None,
            init_fun=None):

        self.name = name
        self.fe = fe

        if   (init_val is     None) and (init_fun is     None):
            self.init_val = numpy.zeros(fe.value_shape())
            self.init = self.init_with_val
        elif (init_val is not None) and (init_fun is     None):
            assert (numpy.shape(init_val) == self.fe.value_shape())
            self.init_val = numpy.asarray(init_val)
            self.init = self.init_with_val
        elif (init_val is     None) and (init_fun is not None):
            self.init_fun = init_fun
            self.init = self.init_with_field
        else:
            assert (0), "Can only provide init_val or init_fun. Aborting."



    def init_with_val(self):
        init_val_str = self.init_val.astype(str).tolist()
        # print(self.func.vector().get_local())
        self.func.interpolate(dolfin.Expression(
            init_val_str,
            element=self.fe))
        # print(self.func.vector().get_local())
        self.func_old.interpolate(dolfin.Expression(
            init_val_str,
            element=self.fe))



    def init_with_field(self):
        self.func.vector()[:]     = self.init_fun.vector().get_local()
        self.func_old.vector()[:] = self.init_fun.vector().get_local()
