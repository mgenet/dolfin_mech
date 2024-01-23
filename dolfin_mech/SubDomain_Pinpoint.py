#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Felipe Álvarez-Barrientos, 2019-2021                                 ###
###                                                                          ###
### Pontificia Universidad Católica, Santiago, Chile                         ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2020-2024                                    ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

################################################################################

class PinpointSubDomain(dolfin.SubDomain):



    def __init__(self, coords, tol=None):

        self.coords = numpy.asarray(coords)
        self.tol = tol if tol is not None else 1e-3

        dolfin.SubDomain.__init__(self)



    def move(self, coords):

        self.coords[:] = coords



    def inside(self, x, on_boundary):

        return (numpy.linalg.norm(x - self.coords) < self.tol)



    def check_inside(self, mesh):

        x_lst = []
        for x in mesh.coordinates():
            if self.inside(x, True):
                x_lst.append(x)
        return x_lst
