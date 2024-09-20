"""
This demo illustrates how to use of DOLFIN for solving the 
Phase-Field-Crystal equation,(taken from Wise FD-multigrid paper, 2009)"""
"""A quick search for switch! gives you all the possible changes to be made in this code.
Modified by Diegel on October 3, 2019.
"""
######################################################################
# Copyright (C) 2009 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2009-06-20
# Last changed: 2013-11-20
# Begin demo

import random
import os
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mpi4py import MPI as pyMPI
comm = MPI.comm_world
rank = comm.Get_rank()


def pfc_function(N):
    
    
    xy_lim = 201.0
	# Class representing the intial conditions #todo: change the initial conditions!
	#class InitialConditions(Expression): 
	
	# Note that eps is used in the definition of values[1] below!!!  Switch: Initial Conditions
    class InitialConditions(UserExpression):
        def __init__(self, **kwargs):
            random.seed(2 + MPI.rank(MPI.comm_world))
            super().__init__(**kwargs)
		#random.seed(2 + rank)
        def eval(self, values, x):
#            values[0] = 0.07 - 0.02*cos((2*pi*(x[0]-12))/32)*sin((2*pi*(x[1]-1))/32) \
#            + 0.02*cos((pi*(x[0]+10))/32)*cos((pi*(x[0]+10))/32)*cos((pi*(x[1]+3))/32)*cos((pi*(x[1]+3))/32) \
#            - 0.01*sin(4*pi*x[0]/32)*sin(4*pi*x[0]/32)*sin(4*pi*(x[1]-6)/32)*sin(4*pi*(x[1]-6)/32)
#            values[1] = (.025 - 1)*((cos((pi*(x[0] - 12))/16)*sin(2*pi*(x[1] - 1)))/50 + \
#            (sin((pi*x[0])/8)**2*sin((pi*(x[1] - 6))/8)**2)/100 - (cos((pi*(x[0] + 10))/32)**2*cos((pi*(x[1] + 3))/32)**2)/50 - 7/100) \
#            - ((cos((pi*(x[0] - 12))/16)*sin(2*pi*(x[1] - 1)))/50 + (sin((pi*x[0])/8)**2*sin((pi*(x[1] - 6))/8)**2)/100 \
#            - (cos((pi*(x[0] + 10))/32)**2*cos((pi*(x[1] + 3))/32)**2)/50 - 7/100)**3 \
#            - (pi**4*cos((pi*x[0])/8)**2*cos((pi*(x[1] - 6))/8)**2)/51200 - (pi**2*cos((pi*x[0])/8)**2*sin((pi*(x[1] - 6))/8)**2)/1600 \
#            - (pi**2*sin((pi*x[0])/8)**2*cos((pi*(x[1] - 6))/8)**2)/1600 + (pi**4*cos((pi*x[0])/8)**2*sin((pi*(x[1] - 6))/8)**2)/25600 \
#            + (pi**4*sin((pi*x[0])/8)**2*cos((pi*(x[1] - 6))/8)**2)/25600 + (pi**2*sin((pi*x[0])/8)**2*sin((pi*(x[1] - 6))/8)**2)/800 \
#            - (3*pi**4*sin((pi*x[0])/8)**2*sin((pi*(x[1] - 6))/8)**2)/51200 - (pi**2*cos((pi*(x[0] + 10))/32)**2*cos((pi*(x[1] + 3))/32)**2)/6400 \
#            + (3*pi**4*cos((pi*(x[0] + 10))/32)**2*cos((pi*(x[1] + 3))/32)**2)/6553600 + (pi**2*cos((pi*(x[0] + 10))/32)**2*sin((pi*(x[1] + 3))/32)**2)/12800 \
#            + (pi**2*cos((pi*(x[1] + 3))/32)**2*sin((pi*(x[0] + 10))/32)**2)/12800 - (pi**4*cos((pi*(x[0] + 10))/32)**2*sin((pi*(x[1] + 3))/32)**2)/3276800 \
#            - (pi**4*cos((pi*(x[1] + 3))/32)**2*sin((pi*(x[0] + 10))/32)**2)/3276800 + (pi**4*sin((pi*(x[0] + 10))/32)**2*sin((pi*(x[1] + 3))/32)**2)/6553600 \
#            + (41*pi**2*cos((pi*(x[0] - 12))/16)*sin(2*pi*(x[1] - 1)))/256 - (42025*pi**4*cos((pi*(x[0] - 12))/16)*sin(2*pi*(x[1] - 1)))/131072
            if x[0] < 75.0 and x[0] > 60.0 and x[1] < 75.0 and x[1] > 60.0:
                values[0] = 0.285 + .446*(cos(0.66*x[1]/sqrt(3))*cos(.66*x[0])-0.5*cos(2*0.66*x[1]/sqrt(3)))
            elif x[0] < 40.0 and x[0] > 25.0 and x[1] < 40.0 and x[1] > 25.0:
                values[0] = 0.285 + \
                .446*(cos(0.66*(sin(pi/4)*x[0]+cos(pi/4)*x[1])/sqrt(3))*cos(.66*(cos(pi/4)*x[0]-sin(pi/4)*x[1])) \
                    -0.5*cos(2*0.66*(sin(pi/4)*x[0]+cos(pi/4)*x[1])/sqrt(3)))
            elif x[0] < 100.0 and x[0] > 85.0 and x[1] < 40.0 and x[1] > 25.0:
                values[0] = 0.285 + \
                .446*(cos(0.66*(sin(-pi/4)*x[0]+cos(-pi/4)*x[1])/sqrt(3))*cos(.66*(cos(-pi/4)*x[0]-sin(-pi/4)*x[1])) \
                    -0.5*cos(2*0.66*(sin(-pi/4)*x[0]+cos(-pi/4)*x[1])/sqrt(3)))
            else:
                values[0] = 0.285
            values[1]=0.0
        def value_shape(self):
            return (2,)
			# Class for interfacing with the Newton solver
    class PhaseFieldCrystalEquation(NonlinearProblem):
        def __init__(self, a, L):
            NonlinearProblem.__init__(self)
            self.L = L
            self.a = a
        def F(self, b, x):
            assemble(self.L, tensor=b)
        def J(self, A, x):
            assemble(self.a, tensor=A)
            
    # Sub domain for Periodic boundary condition. 
    # Switch: Turn on or off periodic boundary conditions by commenting or uncommenting.
#    class PeriodicBoundary(SubDomain):
#
#        def inside(self, x, on_boundary):
#            return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary \
#                        or x[1] < DOLFIN_EPS and x[1] > -DOLFIN_EPS and on_boundary)
#        
#        
#        def map(self, x, y):
#            if x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary:
#                y[0] = x[0] - 32.
#                y[1] = x[1] 
#            else:   # near(x[1], 1)
#                y[0] = x[0]
#                y[1] = x[1] - 32.
#           
#
#        def map(self, x, y):
#            if near(x[0], 32) and near(x[1], 32):
#                y[0] = x[0] - 32.
#                y[1] = x[1] - 32.
#            elif near(x[0], 32):
#                y[0] = x[0] - 32.
#                y[1] = x[1]
#            else:   # near(x[1], 1)
#                y[0] = x[0]
#                y[1] = x[1] - 32.
           

    # Create periodic boundary condition
#    PBC = PeriodicBoundary()
            

	# Model parameters
	
    dt = .5*(xy_lim/N) # time step (notice the dependence on the grid points)
    eps = 0.25      #taken from Wise FD-multigrid paper.
    
	# Form compiler options (Suggested by fenics)
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["representation"] = "uflacs" 
    parameters['allow_extrapolation'] = True  # not sure this is really needed

    # Make mesh ghosted for evaluation of DG terms (I no longer think this is needed but kept here just in case.)
    #parameters["ghost_mode"] = "shared_facet"

	# Create mesh and define function spaces
    mesh = RectangleMesh(Point(0.0,0.0) , Point( xy_lim, xy_lim ), N, N,"right/left")
    # Switch: Plot mesh to make sure mesh division is okay
#    plt.figure()
#    plot(mesh,title="Mesh")
#    
    
	# Define function spaces
    P2 = FiniteElement('P',triangle,2)
    V = P2*P2	# Sets up mixed function space
 #   ME = FunctionSpace(mesh,V,constrained_domain=PBC) # Switch: Turn on if using Periodic Boundary conditions
    ME = FunctionSpace(mesh,V)	# Switch: Turn on if NOT using Periodic Boundary conditions
    

	# Define trial and test functions
    du    = TrialFunction(ME)	# du is being used b/c we are using the fenics project internal Newton Solver
    q, v  = TestFunctions(ME)  

	# Define functions
    u = Function(ME)  # current solution u=[c;mu]
    u0 = Function(ME)  # solution from previous converged step

	# Split mixed functions
    c,  mu  = split(u)
    c0, mu0 = split(u0)

	# Create intial conditions and interpolate
    u_init = InitialConditions()
    u.interpolate(u_init)
    u0.interpolate(u_init)
    

	# for the edge terms in the C0-IP part! # 
	# Define mesh size, averaging mesh size, and normal component
    h = CellDiameter(mesh)
    h_avg = ((h('+') + h('-'))/2.0)
    n = FacetNormal(mesh)
 
	# Define penalty parameter. Switch: We should play with the penalty parameter for better results.
    #alpha_dep = 20/((32/N))
    alpha_dep = 20
    alpha = Constant(alpha_dep) 

	# Define the bilinear form for discretizing the fourth order term # 
	# Switch: Is dot(n,dot(grad(grad(c),n),n)) different from div(grad(c))? This needs to be explored!

#    aIP = inner(div(grad(c)), div(grad(v)))*dx \
#    + inner(avg(dot(n,dot(grad(grad(c)),n))), jump(grad(v),n))*dS \
#    + inner(jump(grad(c),n), avg(dot(n,dot(grad(grad(v)),n))))*dS \
#    + (alpha/h_avg)*inner(jump(grad(c),n), jump(grad(v),n))*dS \
#    + inner(dot(n,dot(grad(grad(c)),n)), dot(grad(v), n))*ds \
#    + inner(dot(grad(c), n), dot(n,dot(grad(grad(v)),n)))*ds \
#    + (alpha/h)*inner(dot(grad(c),n), dot(grad(v),n))*ds
      
    aIP = inner(div(grad(c)), div(grad(v)))*dx \
    - inner(avg(div(grad(c))), jump(grad(v),n))*dS \
    - inner(jump(grad(c),n), avg(div(grad(v))))*dS \
    + (alpha/h_avg)*inner(jump(grad(c),n), jump(grad(v),n))*dS \
    - inner(div(grad(c)), dot(grad(v), n))*ds \
    - inner(dot(grad(c), n), div(grad(v)))*ds \
    + (alpha/h)*inner(dot(grad(c),n), dot(grad(v),n))*ds
      
    # Compute the nonlinear term
    c = variable(c)
    dfdc = c**3 + ((1-eps)*c) #nonlinear part!

	# Define weak statement of the equations
    L0 = c*q*dx - c0*q*dx + dt*dot(grad(mu), grad(q))*dx
    L1 = mu*v*dx - dfdc*v*dx + 2*dot(grad(c0), grad(v))*dx - aIP 

    L = L0 + L1

	# Compute directional derivative about u in the direction of du (Jacobian)
    a = derivative(L, u, du)

	# Create nonlinear problem and Newton solver
    problem = PhaseFieldCrystalEquation(a, L)
    solver = NewtonSolver()
    
    # Switch: Set solver parameters. Currently using suggestion from fenics. 
    solver.parameters["linear_solver"] = "mumps" # Mumps is used to speed up solve time and alleviate memory issues.
    #solver.parameters["preconditioner"] = "ilu" # Switch: Still trying to find a preconditioner which works.
    solver.parameters["convergence_criterion"] = "incremental"
    solver.parameters["relative_tolerance"] = 1e-6
    
    # Set-up definition to compute energies
        
    # Set-up definition to compute energies from lower order terms (called basic energy below)
    def basic_energy_func(u, u_,eps,alpha,h_avg,h):
        return .25*inner(dot(u,u_),dot(u,u_))*dx + ((1-eps)/2)*inner(u,u_)*dx \
        - dot(grad(u), grad(u_))*dx 
        
    # Set-up definition to compute energies from fourth order terms (called aIP energy below)
    # Switch: This should be changed if we change the definition of aIP above
    def aIP_energy_func(u, u_,eps,alpha,h_avg,h):
        return .5*(inner(div(grad(u)), div(grad(u_)))*dx \
        - inner(avg(div(grad(u))), jump(grad(u_), n))*dS \
	    - inner(jump(grad(u), n), avg(div(grad(u_))))*dS \
	    + (alpha/h_avg)*inner(jump(grad(u),n), jump(grad(u_),n))*dS
        - inner(div(grad(u)), dot(grad(u_), n))*ds \
	    - inner(dot(grad(u), n), div(grad(u_)))*ds \
	    + (alpha/h)*inner(dot(grad(u),n), dot(grad(u_),n))*ds)

#    alpha_0=Constant(25/((32/N)))
	    
    # Calculate initial energies
    phi0 = u0.split()[0]
    basic_energy_array = []
#    basic_energy_assem = assemble(basic_energy_func(phi0,phi0,eps,alpha_0,h_avg,h))
#    basic_energy_calc = np.array(basic_energy_assem)
#    basic_energy_array.append(basic_energy_calc)
    
    aIP_energy_array = []
#    aIP_energy_assem = assemble(aIP_energy_func(phi0,phi0,eps,alpha_0,h_avg,h))
#    aIP_energy_calc = np.array(aIP_energy_assem)
#    aIP_energy_array.append(aIP_energy_calc)
    
    tot_energy_array = []
#    tot_energy_assem = basic_energy_assem + aIP_energy_assem
#    tot_energy_calc = np.array(tot_energy_assem)
#    tot_energy_array.append(tot_energy_calc)


	# Define path for output files and make directories if they are not already made
    cur_path = os.getcwd()
    print ("The current working directory is %s" % cur_path)
    data_path = cur_path+'/test_small_regions_'+str(alpha_dep)+'/mesh_size'+str(N)+'/'
    try:
        os.makedirs(data_path)
    except OSError:
        print ("Creation of the directory %s failed" % data_path)
    else:
        print ("Successfully created the directory %s " % data_path)
        
        
    # Switch: Plot initial conditions for debugging. Should be turned off once debugging is complete.
    plt.figure()
    contour = plot(u0[0],title="time = 0", cmap=cm.bwr, vmin=-0.5, vmax=1.2)
#    cbar = plt.colorbar(contour,extend="both")
    m = plt.cm.ScalarMappable(cmap=cm.bwr)
    m.set_array(u0[0])
    m.set_clim(-0.5, 1.2)
    plt.colorbar(m, format='%.2f')
#    plt.clim(.04, .1)
#    plt.show()
    plt.savefig(data_path+'Initial.png')
        
    # Open file to save solution data at each time step
    sol_file = File(data_path+"pfc.pvd", "compressed")

	# Begin solving problem
	# Step in time
    t = 0.0
    T = 1000
    i = 0
    
    Nsteps = int(T/dt) # Switch: Used for debugging. Print number of time steps to screen.
    print(Nsteps)

    while (t < T):
        i+=1
        t += dt
        u0.vector()[:] = u.vector()
        phi0 = u0.split()[0]
        
        solver.solve(problem, u.vector())
        sol_file << (u.split()[0], t)
        
        phi = u.split()[0]
        basic_energy_assem = assemble(basic_energy_func(phi,phi,eps,alpha,h_avg,h))
        basic_energy_calc = np.array(basic_energy_assem)
        scaled_basic_energy = basic_energy_calc/(xy_lim*xy_lim)
        basic_energy_array.append(scaled_basic_energy)
        
        aIP_energy_assem = assemble(aIP_energy_func(phi,phi,eps,alpha,h_avg,h))
        aIP_energy_calc = np.array(aIP_energy_assem)
        scaled_aIP_energy_calc = aIP_energy_calc/(xy_lim*xy_lim)
        aIP_energy_array.append(scaled_aIP_energy_calc)
        
        tot_energy_assem = basic_energy_assem + aIP_energy_assem
        tot_energy_calc = np.array(tot_energy_assem)
        scaled_tot_energy_calc = tot_energy_calc/(xy_lim*xy_lim)
        tot_energy_array.append(scaled_tot_energy_calc)


    
    
    
    # Switch: Used for debugging. Does solution look okay?
    plt.figure()
    contour = plot(phi,title="time = 1000", cmap=cm.bwr, vmin=-0.5, vmax=1.2)
#    cbar = plt.colorbar(contour,extend="both")
    m = plt.cm.ScalarMappable(cmap=cm.bwr)
    m.set_array(phi)
    m.set_clim(-0.5, 1.2)
    plt.colorbar(m, format='%.2f')
#    plt.show()
    plt.savefig(data_path+'Solution.png')
    
#    ax.set_label('cbar_label',rotation=270)
#    ax.set_ticklabels(labels,update_ticks=True)


    # Switch: Plot energies. 
    num_time_steps = len(tot_energy_array)
    time = np.linspace(dt,T,num_time_steps)

    plt.figure()
    plt.plot(time, aIP_energy_array,label='aIP energy')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("aIP Energy")
    #plt.ylim(.03,.04)
    #plt.title('Parameter values are: dt = %d, h = 32/%d, alpha = %d' %)
    #plt.show()
    plt.savefig(data_path+'aIP_energy.png')
    
    plt.figure()
    plt.plot(time, basic_energy_array,label='basic energy')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Basic Energy")
    #plt.ylim(.03,.04)
    #plt.title('Parameter values are: dt = %d, h = 32/%d, alpha = %d' %)
    #plt.show()
    plt.savefig(data_path+'basic_energy.png')
    
    plt.figure()
    plt.plot(time, tot_energy_array)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    plt.title("Scaled Energy")
    plt.xlabel("Time")
    plt.ylabel("Scaled Total Energy")
    #plt.ylim(.03,.04)
    #plt.title('Parameter values are: dt = %d, h = 32/%d, alpha = %d' %)
    #plt.show()
    plt.savefig(data_path+'total_energy.png')

	# function output is function space data, solution, penalty parameter, and mesh data 
    return ME, u, alpha, mesh
        

