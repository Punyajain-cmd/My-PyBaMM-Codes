import pybamm
import numpy as np
import matplotlib.pyplot as plt

model = pybamm.BaseModel("Single Particle Model")

# Define variable
c_e = pybamm.Variable("Concentration of Negative electrode", domain="negative particle")

# Parameters
D = 3.9e-14  # m^2/s
R = 1e-5     # m
j = 1.4      # A/m^2
F = 96485.33289  # C/mol
c_e0 = 2.5e4  # mol/m^3

# PDE definition
N = D * pybamm.grad(c_e)
dc_edt = pybamm.div(N)
model.rhs = {c_e: dc_edt}

# Initial condition
model.initial_conditions = {c_e: pybamm.Scalar(c_e0)}

# Boundary conditions
model.boundary_conditions = {
    c_e: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (pybamm.Scalar(-j / (F * D)), "Neumann")
    }
}

# Model variables
model.variables = {
    "Concentration of Negative electrode": c_e,
    "Flux of Lithium ions in Negative electrode": N,
    "Rate of change of concentration in negative electrode": dc_edt
}

# Geometry
geometry = {
    "negative particle": {
        "r_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(R)}
    }
}

# Mesh
submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
var_pts = {"r_n": 20}
spatial_methods = {"negative particle": pybamm.FiniteVolume()}

mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# Solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 3600, 100)
solution = solver.solve(model, t)


# If solution is a list, get the first Solution object
if isinstance(solution, list):
    solution = solution[0]

if solution is None:
    raise RuntimeError("Solver did not return a solution. Please check the model setup and solver configuration.")

# Extract solution
c_e_sol = solution["Concentration of Negative electrode"]

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(solution.t, c_e_sol(solution.t, r=R))
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Concentration at surface (r = R)")

r_vals = np.linspace(0, R, 100)
ax2.plot(r_vals, c_e_sol(1000, r=r_vals))
ax2.set_xlabel("r")
ax2.set_ylabel("Concentration at t = 1000 s")

plt.tight_layout()
plt.show()
