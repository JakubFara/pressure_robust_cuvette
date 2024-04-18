import firedrake as fd


mesh = fd.Mesh("square_with_hole.msh")

k = 2
V = fd.VectorFunctionSpace(mesh, 'CG', k)

x = fd.SpatialCoordinate(mesh)
print(x[0]. x[1])
expr_x_outer = x[0] / fd.sqrt(x[0]**2 + x[1]**2)
expr_y_outer = x[1] / fd.sqrt(x[0]**2 + x[1]**2)
expr_x_inner = 0.5 * x[0] / fd.sqrt(x[0]**2 + x[1]**2)
expr_y_inner = 0.5 * x[1] / fd.sqrt(x[0]**2 + x[1]**2)

bc_x_outer = fd.DirichletBC(V.sub(0), expr_x_outer, 1)
bc_y_outer = fd.DirichletBC(V.sub(1), expr_y_outer, 1)
bc_x_inner = fd.DirichletBC(V.sub(0), expr_x_inner, 2)
bc_y_inner = fd.DirichletBC(V.sub(1), expr_y_inner, 2)

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
