#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Cécile Patte, 2019-2021                                              ###
###                                                                          ###
### INRIA, Palaiseau, France                                                 ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech
from .Material_Elastic import ElasticMaterial

################################################################################

class WporPoroElasticMaterial(ElasticMaterial):



    def __init__(self,
            problem,
            parameters,
            type):

        self.problem = problem

        assert ('eta' in parameters)
        self.eta = parameters['eta']
        self.n   = parameters['n']

        assert ((type in ['inverse','exp']) or isinstance(type, int))
        self.type = type



    def get_dWpordJs(self):
        # Jf = self.kinematics.Je - self.kinematics.Js
        Jf = self.problem.kinematics.Je * self.problem.Phi
        if (self.type == 'inverse'): # inverse problem
            dWpor1dJs =   self.eta * (self.n+1) * ((self.problem.Phi0 - Jf) / Jf)**self.n * self.problem.Phi0 / (Jf)**2
            dWpor2dJs = - self.eta * (self.n+1) * ((Jf / self.problem.Phi0) - 1.)**self.n / self.problem.Phi0
            dWpordJs  = dolfin.conditional(dolfin.lt(self.problem.Phi, self.problem.Phi0), dWpor1dJs, dWpor2dJs)
        elif (self.type == 'exp'):
            dWpordJs = self.eta * (self.problem.Phi0 / (Jf))**2 * dolfin.exp(-Jf / (self.problem.Phi0 - Jf)) / (self.problem.Phi0 - Jf)
            dWpordJs = dolfin.conditional(dolfin.lt(self.problem.Phi, self.problem.Phi0), dWpordJs, 0)
        elif (isinstance(self.type, int)):
            if (self.type == 2):
                dWpordJs = self.eta * self.n * ((self.problem.Phi0 - Jf) / Jf)**(self.n-1) * self.problem.Phi0 / (Jf)**2
                dWpordJs = dolfin.conditional(dolfin.lt(self.problem.Phi, self.problem.Phi0), dWpordJs, 0)
        dWpordJs = (1. - self.problem.Phi0) * dWpordJs

        return dWpordJs



###################################################### for mixed formulation ###

    def get_res_term(self,
			w_Phi0=None,
			w_Phi=None):

        assert ((w_Phi0 is     None) or (w_Phi is     None))
        assert ((w_Phi0 is not None) or (w_Phi is not None))

        dWpordJs = self.get_dWpordJs()

        if (w_Phi0 is not None):
            res_form = dolfin.inner(
				dWpordJs,
				self.problem.subsols["Phi0"].dsubtest) * self.problem.dV
        elif (w_Phi is not None):
            res_form = dolfin.inner(
				dWpordJs,
				self.problem.subsols["Phi"].dsubtest) * self.problem.dV

        return res_form



########################################## for internal variable formulation ###

    def get_jac_term(self, w_Phi0 = None, w_Phi = None):

        assert (w_Phi0 is     None) or (w_Phi is     None)
        assert (w_Phi0 is not None) or (w_Phi is not None)

        # dWpordJs = self.get_dWpordJs()

        assert(0)

        return jac_form
