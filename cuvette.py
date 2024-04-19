import firedrake as fd
from firedrake.output import VTKFile


def refine_bary(coarse_mesh):
    """Return barycentric refinement of given input mesh"""
    from petsc4py import PETSc
    coarse_dm = coarse_mesh.topology_dm
    transform = PETSc.DMPlexTransform().create(comm=coarse_dm.getComm())
    transform.setType(PETSc.DMPlexTransformType.REFINEALFELD)
    transform.setDM(coarse_dm)
    transform.setUp()
    fine_dm = transform.apply(coarse_dm)
    fine_mesh = fd.Mesh(fine_dm)
    return fine_mesh


coarse_mesh = fd.Mesh("mesh.msh", dim=2)
mesh = refine_bary(coarse_mesh)

k = 2
V = fd.VectorFunctionSpace(mesh, 'CG', k) # displacement
Ev = fd.VectorFunctionSpace(mesh, "CG", k) # velocity
Ep = fd.FunctionSpace(mesh, "DG", k - 1) # pressure
W = fd.MixedFunctionSpace([Ev, Ep])

x, y = fd.SpatialCoordinate(mesh)
expr_x_outer = x / fd.sqrt(x**2 + y**2) - x
expr_y_outer = y / fd.sqrt(x**2 + y**2) - y
expr_x_inner = 0.5 * x / fd.sqrt(x**2 + y**2) - x
expr_y_inner = 0.5 * y / fd.sqrt(x**2 + y**2) - y

bc_x_outer = fd.DirichletBC(V.sub(0), expr_x_outer, (11, 12, 13, 14))
bc_y_outer = fd.DirichletBC(V.sub(1), expr_y_outer, (11, 12, 13, 14))
bc_x_inner = fd.DirichletBC(V.sub(0), expr_x_inner, (21, 22, 23, 24))
bc_y_inner = fd.DirichletBC(V.sub(1), expr_y_inner, (21, 22, 23, 24))

bcs_u = [
    bc_x_outer,
    bc_y_outer,
    bc_x_inner,
    bc_y_inner,
]

dx = fd.dx(degree=4)

w = fd.Function(W)
v_hat, p = fd.split(w)
phi_v_hat, phi_p = fd.TestFunctions(W)
u = fd.Function(V)
phi_u = fd.TestFunction(V)
v_hat_bndry = fd.Function(V)
v_bndry = fd.Function(V)
phi_v_bndry = fd.TestFunction(V)

form = fd.inner(fd.grad(u), fd.grad(phi_u)) * dx
jacobian = fd.derivative(form, u)

problem = fd.NonlinearVariationalProblem(form, u, bcs=bcs_u, J=jacobian)

lu = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_max_it": 12,
    "snes_rtol": 1e-11,
    "snes_atol": 5e-10,
    "snes_linesearch_type": "basic",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}
solver = fd.NonlinearVariationalSolver(problem, solver_parameters=lu)
solver.solve()

fileu = VTKFile("u.pvd")
u.rename("displacement")
fileu.write(u)

# Material params
mu = fd.Constant(1.5)

expr_x_outer_bndry = 0
expr_y_outer_bndry = 0
expr_x_inner_bndry = 0.5 * y / fd.sqrt(x**2 + y**2)
expr_y_inner_bndry = - 0.5 * x / fd.sqrt(x**2 + y**2)

bc_x_outer = fd.DirichletBC(V.sub(0), expr_x_outer_bndry, (11, 12, 13, 14))
bc_y_outer = fd.DirichletBC(V.sub(1), expr_y_outer_bndry, (11, 12, 13, 14))
bc_x_inner = fd.DirichletBC(V.sub(0), expr_x_inner_bndry, (21, 22, 23, 24))
bc_y_inner = fd.DirichletBC(V.sub(1), expr_y_inner_bndry, (21, 22, 23, 24))

bcs_v_bndry = [
    bc_x_outer_bndry,
    bc_y_outer_bndry,
    bc_x_inner_bndry,
    bc_y_inner_bndry,
]

form_bndry = fd.inner(fd.grad(v_bndry), fd.grad(phi_v_bndry)) * dx
jacobian_bndry = fd.derivative(form_bndry, v_bndry)
problem_bndry = fd.NonlinearVariationalProblem(form_bndry, v_bndry, bcs=bcs_v_bndry, J=jacobian_bndry)
solver_bndry = fd.NonlinearVariationalSolver(problem_bndry, solver_parameters=lu)
solver_bndry.solve()
I = fd.Identity(2)
F = I + fd.grad(u)
J = fd.det(F)
v_hat_bndry = fd.project(J * fd.inv(F) * v_bndry, fd.VectorFunctionSpace(mesh, "CG", k))

filev_hat_bndry = fd.File(f"output/v_hat_bndry.pvd")
filev_hat_bndry.write(v_bndry)

bc_outer = fd.DirichletBC(W.sub(0), (0., 0.), (11, 12, 13, 14))
bc_inner = fd.DirichletBC(W.sub(0), v_hat_bndry, (21, 22, 23, 24))

bcs_v_hat = [
    bc_outer,
    bc_inner,
]

# Build functions
# full ALE transformation
I = fd.Identity(2)

F = I + fd.grad(u)
J = fd.det(F)

v = 1 / J * F * v_hat
phi_v = 1 / J * F * phi_v_hat

L = fd.grad(v)

inv_F = fd.inv(F)
L = L * inv_F
D = 0.5 * (L + L.T)
T = - p * I + 2.0 * mu * D

# Data Forces
force = fd.Constant((0, 0))

Eq1 = fd.div(v_hat) * phi_p * fd.dx

rho = 1000
Eq2 = (
    + J * fd.inner(T * inv_F.T, fd.grad(phi_v)) * fd.dx
    - J * rho * fd.inner(force, phi_v) * fd.dx
)

Eq = Eq1 + Eq2

jacobian = fd.derivative(Eq, w)

problem = fd.NonlinearVariationalProblem(Eq, w, bcs=bcs_v_hat, J=jacobian)
lu = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_max_it": 12,
    "snes_rtol": 1e-11,
    "snes_atol": 5e-10,
    "snes_linesearch_type": "basic",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}
solver = fd.NonlinearVariationalSolver(problem, solver_parameters=lu)
solver.solve()

v_hat, p  = w.subfunctions
v = fd.project(1 / J * F * v_hat, fd.VectorFunctionSpace(mesh, "CG", k))
v.rename("velocity")
p.rename("pressure")

filev = fd.File(f"output/v.pvd")
filep = fd.File(f"output/p.pvd")

filev.write(v)
filep.write(p)
