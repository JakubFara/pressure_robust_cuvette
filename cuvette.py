import firedrake as fd


mesh = fd.Mesh("square_with_hole.msh")

k = 2
V = FunctionSpace(m, 'CG', k)

x = SpatialCoordinate(mesh)
expr = x[0] / df.sqrt(x[0]**2 + x[1]**2)
bc_bot_v = fd.DirichletBC(V.sub(0).sub(0), expr, 3)
