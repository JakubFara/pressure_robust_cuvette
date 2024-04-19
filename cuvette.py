import firedrake as fd

# mesh = fd.Mesh("square_with_hole.msh", dim=2)
mesh = fd.Mesh("mesh.msh")
# mesh = fd.UnitSquareMesh(2,2)

k = 2
V = fd.VectorFunctionSpace(mesh, 'CG', k)

x, y = fd.SpatialCoordinate(mesh)
expr_x_outer = x / fd.sqrt(x**2 + y**2)
expr_y_outer = y / fd.sqrt(x**2 + y**2)
expr_x_inner = 0.5 * x / fd.sqrt(x**2 + y**2)
expr_y_inner = 0.5 * y / fd.sqrt(x**2 + y**2)

bc_x_outer = fd.DirichletBC(V.sub(0), expr_x_outer, (11, 12, 13, 14))
bc_y_outer = fd.DirichletBC(V.sub(1), expr_y_outer, (11, 12, 13, 14))
bc_x_inner = fd.DirichletBC(V.sub(0), expr_x_inner, (21, 22, 23, 24))
bc_y_inner = fd.DirichletBC(V.sub(1), expr_y_inner, (21, 22, 23, 24))

bcs = [
    bc_x_outer,
    bc_y_outer,
    bc_x_inner,
    bc_y_inner,
]

dx = fd.dx(degree=4)

phi_u = fd.TestFunction(V)
u = fd.Function(V)

form = fd.inner(fd.grad(u), fd.grad(phi_u)) * dx

J = fd.derivative(form, u)

problem = fd.NonlinearVariationalProblem(form, u, bcs=bcs, J=J)

lu = {"mat_type": "aij",
      "snes_type": "newtonls",
      "snes_monitor": None,
      "snes_converged_reason": None,
      "snes_max_it": 12,
      "snes_rtol": 1e-11,
      "snes_atol": 5e-10,
      "snes_linesearch_type": "basic",
      "ksp_type": "preonly",
      "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps"}
solver = fd.NonlinearVariationalSolver(problem, solver_parameters=lu)
solver.solve()
u.rename("displacement")

fileu = fd.File(f"out/u.pvd")
u.rename("displacement")
fileu.write(u, time=0)
