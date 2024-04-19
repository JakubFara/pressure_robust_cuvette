#!/usr/bin/env python3
import sys
import gmsh
import numpy as np


gmsh.initialize()
h = 0.02

factory = gmsh.model.geo
outer_points = [
    factory.add_point(-1, -1, 0.0, h),
    factory.add_point(1, -1, 0.0, h),
    factory.add_point(1, 1, 0.0, h),
    factory.add_point(- 1, 1, 0.0, h),
]

outer_lines = [
    factory.add_line(outer_points[0], outer_points[1]),
    factory.add_line(outer_points[1], outer_points[2]),
    factory.add_line(outer_points[2], outer_points[3]),
    factory.add_line(outer_points[3], outer_points[0]),
]

inner_points = [
    factory.add_point(-0.5, -0.5, 0.0, h),
    factory.add_point(0.5, -0.5, 0.0, h),
    factory.add_point(0.5, 0.5, 0.0, h),
    factory.add_point(- 0.5, 0.5, 0.0, h),
]

inner_lines = [
    factory.add_line(inner_points[0], inner_points[1]),
    factory.add_line(inner_points[1], inner_points[2]),
    factory.add_line(inner_points[2], inner_points[3]),
    factory.add_line(inner_points[3], inner_points[0]),
]

loop_outer = factory.add_curve_loop(outer_lines)
loop_inner = factory.add_curve_loop(inner_lines)

gmsh.model.geo.addPlaneSurface([loop_outer, loop_inner], 1)

gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(2, [2], 1)
gmsh.model.addPhysicalGroup(1, outer_lines, 1)  # wall
gmsh.model.addPhysicalGroup(1, inner_lines, 2)  # wall

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
# gmsh.model.mesh.generate(1)
# Write mesh data:
gmsh.write(f"square_with_hole.msh")

# Creates  graphical user interface
if "close" not in sys.argv:
    gmsh.fltk.run()

# It finalize the Gmsh API
gmsh.finalize()
