#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import sys
from dolfin import *

from mpi4py import MPI as pyMPI
comm = MPI.comm_world
rank = comm.Get_rank()

import pfc_H2



parameters['allow_extrapolation'] = True

fcn_sp_data_array = []
sol_array = []
alpha_array = []
mesh_data_array = []


for k in range(0,3):
    
    N = 8*2**(k+3)
    print('N = ',N)
    
    # file_name = 'mesh_size_'+str(N)
#     print(file_name)
    fcn_sp_data, sol, alpha, mesh_data = pfc_H2.pfc_function(N)
    fcn_sp_data_array.append(fcn_sp_data)
    sol_array.append(sol)
    alpha_array.append(alpha)
    mesh_data_array.append(mesh_data)
    
file = open('PFC_error_and_rates.txt','w')

phi_error_array = []
phi_error_array_H2 = []
phi_error_array_L2 = []
phi_error_array_L2_fenics = []

for k in range(0,2):
    
    N = 8*2**k
    N1 = 8*2**(k+1)
    
    # Pull info from arrays for coarse mesh
    sol_coarse_mesh = sol_array[k]
    phi_sol_coarse_mesh = sol_coarse_mesh.split(True)[0]
    
    # Pull info from arrays for fine mesh
    fcn_sp_data_fine_mesh = fcn_sp_data_array[k+1]
    sol_fine_mesh = sol_array[k+1]
    phi_sol_fine_mesh = sol_fine_mesh.split(True)[0]
    mu_sol_fine_mesh = sol_fine_mesh.split(True)[1]
    alpha_fine_mesh = alpha_array[k+1]
    mesh_data_fine_mesh = mesh_data_array[k+1]
    

    # Define mesh size for penalty term
    h = CellDiameter(mesh_data_fine_mesh)
    h_avg = (h('+') + h('-'))/2.0
    #print('h = ',float(h_avg))
    n = FacetNormal(mesh_data_fine_mesh)

    # Interpolate the coarse mesh solution
    sol_interp = interpolate(sol_coarse_mesh, fcn_sp_data_fine_mesh) #Note sol_prev_mesh here is still the previous mesh solution new!

    phi_interp = sol_interp.split(True)[0]
    mu_interp = sol_interp.split(True)[1]

    # Compute the error
    phi_diff = phi_sol_fine_mesh - phi_interp

    phi_error_calc = dot(div(grad(phi_diff)), div(grad(phi_diff)))*dx + (alpha_fine_mesh('+')/h_avg)*inner(jump(grad(phi_diff),n), jump(grad(phi_diff),n))*dS
   # phi_error_calc_H2 = dot(div(grad(phi_diff)), div(grad(phi_diff)))*dx
    phi_error_calc_L2 = dot(phi_diff, phi_diff)*dx + (alpha_fine_mesh('+')/h_avg)*inner(jump(grad(phi_diff),n), jump(grad(phi_diff),n))*dS
    
    phi_error = assemble(phi_error_calc)
    phi_error_array.append(sqrt(abs(phi_error)))
    
    # phi_error_H2 = assemble(phi_error_calc_H2)
#     phi_error_array_H2.append(sqrt(abs(phi_error_H2)))
    
    phi_error_L2 = assemble(phi_error_calc_L2)
    phi_error_array_L2.append(sqrt(abs(phi_error_L2)))
    
    
    # Write to a file
    file.write('n_coarse = '+str(N)+' to n_fine = '+str(N1)+' error = '+str(phi_error))
    file.write("\n")
    
    # file.write('n_coarse = '+str(N)+' to n_fine = '+str(N1)+' L2 error = '+str(phi_error_L2))
#     file.write("\n")

    plt.figure()
    plt.plot(sol_fine_mesh,label='Solution')
    
    plt.figure()
    plt.plot(phi_interp,label='Interpolated Solution')
    
phi_err_rate_array = []
phi_err_rate_array_H2 = []
phi_err_rate_array_L2 = []
phi_err_rate_array_L2_fenics = []


for k in range(0,1):
    
    N = 8*2**k
    N1 = 8*2**(k+1)
    N2 = 8*2**(k+2)
    rate = phi_error_array[k]/phi_error_array[k+1]/2
    phi_err_rate_array.append(rate)
    
    # rate_H2 = phi_error_array_H2[k]/phi_error_array_H2[k+1]/2
#     phi_err_rate_array_H2.append(rate_H2)
    
    rate_L2 = phi_error_array_L2[k]/phi_error_array_L2[k+1]/2
    phi_err_rate_array_L2.append(rate_L2)
    
    
    print(rate)
    print(rate_L2)
    file.write('aIP n_coarse = '+str(N)+' to n_fine = '+str(N1)+' over n_coarse = '+str(N1)+' to n_fine = '+str(N2)+'rate = '+str(rate))
    file.write("\n")
    # file.write('H2 n_coarse = '+str(N)+' to n_fine = '+str(N1)+' over n_coarse = '+str(N1)+' to n_fine = '+str(N2)+' H2rate = '+str(rate_H2))
#     file.write("\n")
    
    
print(phi_error_array)
#print(phi_error_array_H2)
print(phi_err_rate_array)
#print(phi_err_rate_array_H2)
print(phi_err_rate_array_L2)

file.write('Energy Norm Rates = ')
file.write("\n")
file.write(''.join(str(elem) for elem in phi_err_rate_array))
file.write("\n")
#file.write('Energy Norm Rates No Penalty Term = ')
#file.write("\n")
#file.write(''.join(str(elem) for elem in phi_err_rate_array_H2))
#file.write("\n")
file.write('L2 Norm Rates With Penalty Term = ')
file.write("\n")
file.write(''.join(str(elem) for elem in phi_err_rate_array_L2))






file.close()