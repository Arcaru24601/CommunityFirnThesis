#!/usr/bin/env python
"""
Functions to solve the diffusion equation
"""
import time

import numpy as np
from scipy import interpolate
import scipy.integrate
from scipy.sparse import spdiags
import scipy.sparse.linalg as splin
from constants import *
import sys
from numba import jit, njit
import time as t


def solver(a_U, a_D, a_P, b):
    """
    function for solving matrix problem

    :param a_U:
    :param a_D:
    :param a_P:
    :param b:

    :return phi_t:
    """

    nz = np.size(b)

    diags = (np.append([a_U, -a_P], [a_D], axis=0))
    cols = np.array([1, 0, -1])

    big_A = spdiags(diags, cols, nz, nz, format='csc')

    big_A = big_A.T


    rhs = -b

    phi_t = splin.spsolve(big_A, rhs)

    return phi_t


####!!!!
def transient_solve_TR(z_edges, Z_P, nt, dt, Gamma_P, phi_0, nz_P, nz_fv, phi_s, tot_rho, c_vol, mode):
    """
    transient 1-d diffusion finite volume method
    :param c_vol:
    :param tot_rho:
    :param z_edges:
    :param Z_P: this is the same as self.z?
    :param nt:
    :param dt:
    :param Gamma_P: firn conductivity K_firn, in Patankar denoted as k
    :param phi_0:
    :param nz_P:
    :param nz_fv:
    :param phi_s:
    :param mode: This is just for me to understand what time the solver needs for different modes ('firnair',
                'isotopeDiffusion', and 'diffusion')
    :return phi_t:
    """

    phi_t = phi_0

    for i_time in range(nt):
        dZ = np.diff(z_edges)  # width of nodes
        deltaZ_u = np.diff(Z_P)
        deltaZ_u = np.append(deltaZ_u[0], deltaZ_u)
        deltaZ_d = np.diff(Z_P)
        deltaZ_d = np.append(deltaZ_d, deltaZ_d[-1])

        # these seem to be interpolation factors: ratio of distances associated with an interface
        f_u = 1 - (Z_P[:] - z_edges[0:-1]) / deltaZ_u[:]
        f_d = 1 - (z_edges[1:] - Z_P[:]) / deltaZ_d[:]

        Gamma_U = np.append(Gamma_P[0], Gamma_P[0: -1])
        Gamma_D = np.append(Gamma_P[1:], Gamma_P[-1])

        Gamma_u = 1 / ((1 - f_u) / Gamma_P + f_u / Gamma_U)  # Patankar eq. 4.9
        Gamma_d = 1 / ((1 - f_d) / Gamma_P + f_d / Gamma_D)

        S_C = 0
        S_C = S_C * np.ones(nz_P)

        D_u = (Gamma_u / deltaZ_u)
        D_d = (Gamma_d / deltaZ_d)

        b_0 = S_C * dZ  # first term of Patankar eq. 4.41d

        a_U = D_u  # Patankar eq. 4.41a,b
        a_D = D_d  # Patankar eq. 4.41a,b

        a_P_0 = c_vol * dZ / dt  # (new) Patankar eq. 4.41c
        #######################################

        S_P = 0.0
        a_P = a_U + a_D + a_P_0 - S_P * dZ

        #######################################
        ### Boundary conditions:
        ### type 1 is a specified value, type 2 is a specified gradient
        ### (units for gradient are degrees/meter)
        bc_u_0 = phi_s  # need to pay attention to surface boundary for gas
        bc_type_u = 1
        bc_u = np.concatenate(([bc_u_0], [bc_type_u]))

        bc_d_0 = 0
        bc_type_d = 2
        bc_d = np.concatenate(([bc_d_0], [bc_type_d]))
        #########################################

        b = b_0 + a_P_0 * phi_t  # Patankar 4.41d

        # Upper boundary
        a_P[0] = 1
        a_U[0] = 0
        a_D[0] = 0
        b[0] = bc_u[0]

        # Down boundary
        a_P[-1] = 1
        a_D[-1] = 0
        a_U[-1] = 1
        b[-1] = deltaZ_u[-1] * bc_d[0]

        phi_t = solver(a_U, a_D, a_P, b)


    return phi_t


###################################
### end transient_solve_TR ########
###################################


