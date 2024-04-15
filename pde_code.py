from pde_solver_client import client

pde_code = """
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import FunctionSpace
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD

# Create mesh
domain = mesh.create_unit_square(comm, 8, 8, mesh.CellType.quadrilateral)

# Define FunctionSpace
V = FunctionSpace(domain, ("Lagrange", 1))

# Define Dirichlet boundary condition function
uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

# Create the Dirichlet boundary condition
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

# Define trial and test function
import ufl
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define source term
from dolfinx import default_scalar_type
f = fem.Constant(domain, default_scalar_type(-6))

# Define variational problem
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Form and solve the linear system
from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Compute the error
V2 = fem.FunctionSpace(domain, ("Lagrange", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
error_max = np.max(np.abs(uD.x.array - uh.x.array))

# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")
"""
output= client(pde_code)
print(output)
