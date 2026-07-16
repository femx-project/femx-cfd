from pathlib import Path

import femx


MESH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "meshes"
    / "demo"
    / "2d_square_res_low.msh"
)


def make_problem():
    model = femx.NavierStokesModel(
        MESH,
        num_steps=1,
        dt=0.25,
        rho=1.0,
        mu=0.01,
    )
    prob = femx.NavierStokesProblem(model)
    prob.add_bc(femx.DirichletBC("wall_top", "velocity", (0.0, 0.0)))
    prob.add_bc(femx.DirichletBC("wall_bottom", "velocity", (0.0, 0.0)))
    prob.add_bc(femx.DirichletBC("wall_right", "pressure", 0.0))
    prob.add_ctr(
        femx.NormalVelocityControl(
            "wall_left",
            normal=(1.0, 0.0),
            times=(0.0,),
        )
    )

    y = model.mesh.coordinates[prob.ctr_mesh_node_ids, 1]
    pts = [(x, val) for x in (-0.4, -0.25, 0.0, 0.25) for val in y]
    op = femx.VelocityObservationOperator(
        model,
        points=pts,
        times=(0.25,),
        components=(0, 1),
    )
    return prob, op


def truth_parameter(prob):
    inlet = prob.model.mesh.boundary(prob.ctr.boundary)
    y = prob.model.mesh.coordinates[prob.ctr_mesh_node_ids, 1]
    center = 0.5 * (inlet.coordinates[:, 1].min() + inlet.coordinates[:, 1].max())
    radius = 0.5 * (inlet.coordinates[:, 1].max() - inlet.coordinates[:, 1].min())
    return 0.2 * (1.0 - ((y - center) / radius) ** 2)
