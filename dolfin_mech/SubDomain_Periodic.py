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
###                                                                          ###
### Inspired from https://comet-fenics.readthedocs.io/en/latest/             ###
###                        demo/periodic_homog_elas/periodic_homog_elas.html ###
###                                                                          ###
################################################################################

import dolfin
import numpy as np

################################################################################

class PeriodicSubDomain(dolfin.SubDomain):



    def __init__(self,
            dim,
            bbox,
            vertices=None,
            tol=None):

        self.dim = dim

        self.xmin = bbox[0]
        self.xmax = bbox[1]
        if (self.dim >= 2):
            self.ymin = bbox[2]
            self.ymax = bbox[3]
        if (self.dim >= 3):
            self.zmin = bbox[4]
            self.zmax = bbox[5]

        if (tol is None):
            self.tol = (self.xmax - self.xmin)**2
            if (self.dim >= 2):
                self.tol += (self.ymax - self.ymin)**2
            if (self.dim >= 3):
                self.tol += (self.zmax - self.zmin)**2
            self.tol = (self.tol)**(1/2)
            self.tol *= 1e-2
        else:
            self.tol = tol
       
        dolfin.SubDomain.__init__(self, map_tol=self.tol)

        if (self.dim == 2):
            self.inside = self.inside_2D
            self.map = self.map_2D
        elif (self.dim == 3):
            self.inside = self.inside_3D
            self.map = self.map_3D
        else:
            assert (0),\
                "dim must be 2 or 3. Aborting."

    
        if (vertices is None):
            vertices = np.array([[self.xmin, self.ymin],
                                 [self.xmax, self.ymin],
                                 [self.xmax, self.ymax],
                                 [self.xmin, self.ymax]])
        # if (vertices is not None):
        self.vv = vertices
        self.a1 = self.vv[1,:]-self.vv[0,:] # first vector generating periodicity
        self.a2 = self.vv[3,:]-self.vv[0,:] # second vector generating periodicity
        # check if UC vertices form indeed a parallelogram
        assert np.linalg.norm(self.vv[2,:] - self.vv[3,:] - self.a1) <= self.tol
        assert np.linalg.norm(self.vv[2,:] - self.vv[1,:] - self.a2) <= self.tol



    # def inside_2D(self, x, on_boundary):
    #     return bool(on_boundary and (dolfin.near(x[0], self.xmin, self.tol) or dolfin.near(x[1], self.ymin, self.tol)) and not (dolfin.near(x[0], self.xmax, self.tol) or dolfin.near(x[1], self.ymax, self.tol)))



    def inside_2D(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the
        # bottom-right or top-left vertices
        return bool((dolfin.near(x[0], self.vv[0,0] + x[1]*self.a2[0]/self.vv[3,1], self.tol) or
                    dolfin.near(x[1], self.vv[0,1] + x[0]*self.a1[1]/self.vv[1,0], self.tol)) and
                    (not ((dolfin.near(x[0], self.vv[1,0], self.tol) and dolfin.near(x[1], self.vv[1,1], self.tol)) or
                    (dolfin.near(x[0], self.vv[3,0], self.tol) and dolfin.near(x[1], self.vv[3,1], self.tol)))) and on_boundary)



    def inside_3D(self, x, on_boundary):
        return bool(on_boundary and (dolfin.near(x[0], self.xmin, self.tol) or dolfin.near(x[1], self.ymin, self.tol) or dolfin.near(x[2], self.zmin, self.tol)) and not (dolfin.near(x[0], self.xmax, self.tol) or dolfin.near(x[1], self.ymax, self.tol) or dolfin.near(x[2], self.zmax, self.tol)))



    # def map_2D(self, x, y):
    #     # Vertex
    #     if  (dolfin.near(x[0], self.xmax, self.tol) \
    #     and  dolfin.near(x[1], self.ymax, self.tol))\
    #     or  (dolfin.near(x[0], self.xmax, self.tol) \
    #     and  dolfin.near(x[1], self.ymin, self.tol))\
    #     or  (dolfin.near(x[0], self.xmin, self.tol) \
    #     and  dolfin.near(x[1], self.ymax, self.tol)):
    #         y[0] = self.xmin
    #         y[1] = self.ymin
    #     # Edges
    #     elif dolfin.near(x[0], self.xmax, self.tol):
    #         y[0] = self.xmin
    #         y[1] = x[1]
    #     elif dolfin.near(x[1], self.ymax, self.tol):
    #         y[0] = x[0]
    #         y[1] = self.ymin
    #     # FAB: Else clause is there to have the point always end up mapped somewhere, cf. https://fenicsproject.discourse.group/t/periodic-boundary-class-for-2-target-domains/99/2
    #     else:
    #         y[0] = -1.
    #         y[1] = -1.



    def map_2D(self, x, y):
        if dolfin.near(x[0], self.vv[2,0], self.tol) and dolfin.near(x[1], self.vv[2,1], self.tol): # if on top-right corner
            y[0] = x[0] - (self.a1[0]+self.a2[0])
            y[1] = x[1] - (self.a1[1]+self.a2[1])
        elif dolfin.near(x[0], self.vv[1,0] + x[1]*self.a2[0]/self.vv[2,1], self.tol): # if on right boundary
            y[0] = x[0] - self.a1[0]
            y[1] = x[1] - self.a1[1]
        else:   # should be on top boundary
            y[0] = x[0] - self.a2[0]
            y[1] = x[1] - self.a2[1]



    def map_3D(self, x, y):
        # Vertex
        if  (dolfin.near(x[0], self.xmax, self.tol) \
        and  dolfin.near(x[1], self.ymax, self.tol) \
        and  dolfin.near(x[2], self.zmax, self.tol))\
        or  (dolfin.near(x[0], self.xmax, self.tol) \
        and  dolfin.near(x[1], self.ymax, self.tol) \
        and  dolfin.near(x[2], self.zmin, self.tol))\
        or  (dolfin.near(x[0], self.xmax, self.tol) \
        and  dolfin.near(x[1], self.ymin, self.tol) \
        and  dolfin.near(x[2], self.zmax, self.tol))\
        or  (dolfin.near(x[0], self.xmax, self.tol) \
        and  dolfin.near(x[1], self.ymin, self.tol) \
        and  dolfin.near(x[2], self.zmin, self.tol))\
        or  (dolfin.near(x[0], self.xmin, self.tol) \
        and  dolfin.near(x[1], self.ymax, self.tol) \
        and  dolfin.near(x[2], self.zmax, self.tol))\
        or  (dolfin.near(x[0], self.xmin, self.tol) \
        and  dolfin.near(x[1], self.ymax, self.tol) \
        and  dolfin.near(x[2], self.zmin, self.tol))\
        or  (dolfin.near(x[0], self.xmin, self.tol) \
        and  dolfin.near(x[1], self.ymin, self.tol) \
        and  dolfin.near(x[2], self.zmax, self.tol)):
            y[0] = self.xmin
            y[1] = self.ymin
            y[2] = self.zmin
        # Edges
        elif (dolfin.near(x[0], self.xmax, self.tol) \
        and   dolfin.near(x[1], self.ymax, self.tol))\
        or   (dolfin.near(x[0], self.xmax, self.tol) \
        and   dolfin.near(x[1], self.ymin, self.tol))\
        or   (dolfin.near(x[0], self.xmin, self.tol) \
        and   dolfin.near(x[1], self.ymax, self.tol)):
            y[0] = self.xmin
            y[1] = self.ymin
            y[2] = x[2]
        elif (dolfin.near(x[0], self.xmax, self.tol) \
        and   dolfin.near(x[2], self.zmax, self.tol))\
        or   (dolfin.near(x[0], self.xmax, self.tol) \
        and   dolfin.near(x[2], self.zmin, self.tol))\
        or   (dolfin.near(x[0], self.xmin, self.tol) \
        and   dolfin.near(x[2], self.zmax, self.tol)):
            y[0] = self.xmin
            y[1] = x[1]
            y[2] = self.zmin
        elif (dolfin.near(x[1], self.ymax, self.tol) \
        and   dolfin.near(x[2], self.zmax, self.tol))\
        or   (dolfin.near(x[1], self.ymax, self.tol) \
        and   dolfin.near(x[2], self.zmin, self.tol))\
        or   (dolfin.near(x[1], self.ymin, self.tol) \
        and   dolfin.near(x[2], self.zmax, self.tol)):
            y[0] = x[0]
            y[1] = self.ymin
            y[2] = self.zmin
        # Faces
        elif dolfin.near(x[0], self.xmax, self.tol):
            y[0] = self.xmin
            y[1] = x[1]
            y[2] = x[2]
        elif dolfin.near(x[1], self.ymax, self.tol):
            y[0] = x[0]
            y[1] = self.ymin
            y[2] = x[2]
        elif dolfin.near(x[2], self.zmax, self.tol):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = self.zmin
        # FAB: Else clause is there to have the point always end up mapped somewhere, cf. https://fenicsproject.discourse.group/t/periodic-boundary-class-for-2-target-domains/99/2
        else:
            y[0] = -1000.
            y[1] = -1000.
            y[2] = -1000.
