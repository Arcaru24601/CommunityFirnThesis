import numpy as np
from constants import *
from numba import jit, njit, gdb
import time as t
import matplotlib.pyplot as plt


@njit
def w(z_airdict, Tz_airdict, dt_airdict, por_op_airdict, advection_type_airdict,
      w_firn_airdict, z_co_airdict, z_edges, rho_edges, Z_P, dPdz, por_cl_edges, dscl):
    """
    Function for downward advection of air and also calculates total air content.
    """
    if advection_type_airdict == 'Darcy':
        por_op_edges = np.interp(z_edges, z_airdict, por_op_airdict)
        T_edges = np.interp(z_edges, z_airdict, Tz_airdict)
        p_star = por_op_edges * np.exp(M_AIR * GRAVITY * z_edges / (R * T_edges))
        dPdz_edges = np.interp(z_edges, z_airdict, dPdz)

        perm = 10.0 ** (-7.7) * p_star ** 3.4  # Freitag, 2002
        visc = 1.5e-5  # kg m^-1 s^-1, dynamic viscosity, source?
        flux = -1.0 * perm / visc * dPdz_edges  # units m/s

        w_ad = flux / p_star / dt_airdict

    elif advection_type_airdict == 'Christo':

        por_op_edges = np.interp(z_edges, Z_P, por_op_airdict)
        w_firn_edges = np.interp(z_edges, Z_P, w_firn_airdict)  # units m/s
        T_edges = np.interp(z_edges, Z_P, Tz_airdict)
        C = np.exp(M_AIR * GRAVITY * z_edges / (R * T_edges))

        op_ind = np.where(z_edges <= z_co_airdict)[0]  # indices of all nodes with open porosity (shallower than CO)
        op_ind2 = np.where(z_edges <= z_co_airdict + 20)[0]  # a bit deeper
        co_ind = op_ind[-1]
        cl_ind1 = np.where(z_edges > z_co_airdict)[0]  # closed indices
        cl_ind = np.intersect1d(cl_ind1, op_ind2)

        Xi_up = por_op_edges[op_ind2] / np.reshape(por_op_edges[op_ind2], (-1, 1))
        Xi_down = (1 + np.log(np.reshape(w_firn_edges[op_ind2], (-1, 1)) / w_firn_edges[op_ind2]))
        Xi = Xi_up / Xi_down  # Equation 5.10 in Christo's thesis; Xi[i,j] is the pressure increase (ratio) for bubbles at depth[i] that were trapped at depth[j]

        integral_matrix = (Xi.T * dscl[op_ind2] * C[op_ind2]).T
        integral_matrix_sum = integral_matrix.sum(axis=1)

        p_ratio = np.zeros_like(z_edges)
        p_ratio[op_ind] = integral_matrix_sum[op_ind]  # 5.11
        p_ratio[cl_ind] = p_ratio[co_ind] * Xi[cl_ind, co_ind]  # 5.12
        p_ratio[cl_ind[-1] + 1:] = p_ratio[cl_ind[-1]]

        flux = w_firn_edges[co_ind - 1] * p_ratio[co_ind - 1] * por_cl_edges[co_ind - 1]

        velocity = np.minimum(w_firn_edges,
                              ((flux + 1e-10 - w_firn_edges * p_ratio * por_cl_edges) / (por_op_edges + 1e-10 * C)))

        w_ad = (velocity - w_firn_edges)

    elif advection_type_airdict == 'zero':
        w_ad = np.zeros_like(rho_edges)

    return w_ad

@njit
def A(P):
    """Power-law scheme, Patankar eq. 5.34"""
    A = np.maximum((1 - 0.1 * np.abs(P)) ** 5, np.zeros(P.size))
    return A


@njit
def F_upwind(F):
    """ Upwinding scheme """
    F_upwind = np.maximum(F, 0)
    return F_upwind


# ----------------------------------------------------------------------------------------------------------------------
# define all the variables

z_airdict = np.load('z_airdict.npy')
Tz_airdict = np.load('Tz_airdict.npy')
dt_airdict = 31557600.0
por_op_airdict = np.load('por_op_airdict.npy')
por_tot_airdict = np.load('por_tot_airdict.npy')
pressure_airdict = np.load('pressure_airdict.npy')
advection_type_airdict = 'Christo'
por_cl_airdict = np.load('por_cl_airdict.npy')
w_firn_airdict = np.load('w_firn_airdict.npy')
z_co_airdict = 62.43274777905841
z_edges = np.load('z_edges.npy')
rho_edges = np.load('rho_edges.npy')
Z_P = np.load('Z_P.npy')
dZ = np.load('dZ.npy')
dPdz = np.load('dPdz.npy')
por_cl_edges = np.load('por_cl_edges.npy')
dscl = np.load('dscl.npy')

# run w()
time_ = []
for i in range(10000):
    start = t.time_ns()
    w(z_airdict, Tz_airdict, dt_airdict, por_op_airdict, advection_type_airdict,
      w_firn_airdict, z_co_airdict, z_edges, rho_edges, Z_P, dPdz, por_cl_edges, dscl)
    end = t.time_ns()
    time_.append(end - start)

time__ = np.array(time_)
np.save('time_w', time__)

print('mean: ', np.mean(time__))
print('mean [1:]: ', np.mean(time__[1:]))

# w.inspect_types()
print(z_edges)