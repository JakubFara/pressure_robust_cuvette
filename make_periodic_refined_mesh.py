import numpy as np

import ufl

from pyop2.mpi import COMM_WORLD
from firedrake.utils import IntType, RealType, ScalarType

from firedrake import (
    VectorFunctionSpace,
    Function,
    Constant,
    par_loop,
    dx,
    WRITE,
    READ,
    assemble,
    Interpolate,
    FiniteElement,
    interval,
    tetrahedron,
    CylinderMesh,
    Mesh,
)
from firedrake.cython import dmcommon
from firedrake import mesh
from firedrake import function
from firedrake import functionspace
from firedrake.petsc import PETSc

from pyadjoint.tape import no_annotations

distribution_parameters_noop = {"partition": False,
                                "overlap_type": (mesh.DistributedMeshOverlapType.NONE, 0)}
reorder_noop = False


def refine_bary(coarse_mesh):
    """Return barycentric refinement of given input mesh"""
    from petsc4py import PETSc
    coarse_dm = coarse_mesh.topology_dm
    transform = PETSc.DMPlexTransform().create(comm=coarse_dm.getComm())
    transform.setType(PETSc.DMPlexTransformType.REFINEALFELD)
    transform.setDM(coarse_dm)
    transform.setUp()
    fine_dm = transform.apply(coarse_dm)
    fine_mesh = Mesh(fine_dm)
    return fine_mesh


def _postprocess_periodic_mesh(coords, comm, distribution_parameters, reorder, name, distribution_name, permutation_name):
    dm = coords.function_space().mesh().topology.topology_dm
    dm.removeLabel("pyop2_core")
    dm.removeLabel("pyop2_owned")
    dm.removeLabel("pyop2_ghost")
    dm.removeLabel("exterior_facets")
    dm.removeLabel("interior_facets")
    V = coords.function_space()
    dmcommon._set_dg_coordinates(dm,
                                 V.finat_element,
                                 V.dm.getLocalSection(),
                                 coords.dat._vec)
    return mesh.Mesh(
        dm,
        comm=comm,
        distribution_parameters=distribution_parameters,
        reorder=reorder,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )


@PETSc.Log.EventDecorator()
def PartiallyPeriodicRefinedRectangleMesh(
    nx,
    ny,
    Lx,
    Ly,
    direction="x",
    quadrilateral=False,
    reorder=None,
    distribution_parameters=None,
    diagonal=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """Generates RectangleMesh that is periodic in the x or y direction.

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :kwarg direction: The direction of the periodicity.
    :kwarg quadrilateral: (optional), creates quadrilateral mesh.
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg diagonal: (optional), one of ``"crossed"``, ``"left"``, ``"right"``.
        Not valid for quad meshes.
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.

    If direction == "x" the boundary edges in this mesh are numbered as follows:

    * 1: plane y == 0
    * 2: plane y == Ly

    If direction == "y" the boundary edges are:

    * 1: plane x == 0
    * 2: plane x == Lx
    """

    if direction not in ("x", "y"):
        raise ValueError("Unsupported periodic direction '%s'" % direction)

    # handle x/y directions: na, La are for the periodic axis
    na, nb, La, Lb = nx, ny, Lx, Ly
    if direction == "y":
        na, nb, La, Lb = ny, nx, Ly, Lx

    if na < 3:
        raise ValueError(
            "2D periodic meshes with fewer than 3 cells in each direction are not currently supported"
        )

    coarse_mesh = CylinderMesh(
        na,
        nb,
        1.0,
        1.0,
        longitudinal_direction="z",
        quadrilateral=quadrilateral,
        reorder=reorder_noop,
        distribution_parameters=distribution_parameters_noop,
        diagonal=diagonal,
        comm=comm,
        name=name,
        distribution_name=distribution_name,
        permutation_name=permutation_name,
    )
    m = refine_bary(coarse_mesh)
    coord_family = "DQ" if quadrilateral else "DG"
    cell = "quadrilateral" if quadrilateral else "triangle"
    coord_fs = VectorFunctionSpace(
        m, FiniteElement(coord_family, cell, 1, variant="equispaced"), dim=2
    )
    old_coordinates = m.coordinates
    new_coordinates = Function(
        coord_fs, name=mesh._generate_default_mesh_coordinates_name(name)
    )

    # make x-periodic mesh
    # unravel x coordinates like in periodic interval
    # set y coordinates to z coordinates
    domain = "{[i, j, k, l]: 0 <= i, k < old_coords.dofs and 0 <= j < new_coords.dofs and 0 <= l < 3}"
    instructions = f"""
    <{RealType}> Y = 0
    <{RealType}> pi = 3.141592653589793
    <{RealType}> oc[k, l] = real(old_coords[k, l])
    for i
        Y = Y + oc[i, 1]
    end
    for j
        <{RealType}> nc0 = atan2(oc[j, 1], oc[j, 0]) / (pi* 2)
        nc0 = nc0 + 1 if nc0 < 0 else nc0
        nc0 = 1 if nc0 == 0 and Y < 0 else nc0
        new_coords[j, 0] = nc0 * Lx[0]
        new_coords[j, 1] = old_coords[j, 2] * Ly[0]
    end
    """

    cLx = Constant(La)
    cLy = Constant(Lb)

    par_loop(
        (domain, instructions),
        dx,
        {
            "new_coords": (new_coordinates, WRITE),
            "old_coords": (old_coordinates, READ),
            "Lx": (cLx, READ),
            "Ly": (cLy, READ),
        },
    )

    if direction == "y":
        # flip x and y coordinates
        operator = np.asarray([[0, 1], [1, 0]])
        new_coordinates.dat.data[:] = np.dot(new_coordinates.dat.data, operator.T)

    return _postprocess_periodic_mesh(new_coordinates,
                                      comm,
                                      distribution_parameters,
                                      reorder,
                                      name,
                                      distribution_name,
                                      permutation_name)
