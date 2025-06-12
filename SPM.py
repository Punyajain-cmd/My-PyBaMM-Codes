import pybamm
import matplotlib.pyplot as plt
import numpy as np

# Define the model
model = pybamm.BaseModel("Single Particle Model")

# Variables for concentrations
c_n = pybamm.Variable("Concentration in negative particle", domain="negative particle")
c_p = pybamm.Variable("Concentration in positive particle", domain="positive particle")

# Parameters from Chen2020-compatible names
D_n = pybamm.Parameter("Negative electrode diffusivity [m2.s-1]")
D_p = pybamm.Parameter("Positive electrode diffusivity [m2.s-1]")
R_n = pybamm.Parameter("Radius of negative electrode particle [m]")
R_p = pybamm.Parameter("Radius of positive electrode particle [m]")
I = pybamm.FunctionParameter("Current function [A]", {"Time [s]": pybamm.t})
F = pybamm.Parameter("Faraday constant [C.mol-1]")
c_n0 = pybamm.Parameter("Initial concentration in negative electrode [mol.m-3]")
c_p0 = pybamm.Parameter("Initial concentration in positive electrode [mol.m-3]")
eps_n = pybamm.Parameter("Maximum concentration in negative electrode [mol.m-3]")
eps_p = pybamm.Parameter("Maximum concentration in positive electrode [mol.m-3]")
L_n = pybamm.Parameter("Thickness of negative electrode [m]")
L_p = pybamm.Parameter("Thickness of positive electrode [m]")
A = pybamm.Parameter("Electrode width [m]") * pybamm.Parameter("Electrode height [m]")

# Geometry-dependent variables
a_n = 3 * eps_n / R_n
a_p = 3 * eps_p / R_p
j_n = I / (F * A * a_n * L_n)
j_p = -I / (F * A * a_p * L_p)

# PDEs for diffusion
N_n = -D_n * pybamm.grad(c_n)
N_p = -D_p * pybamm.grad(c_p)
dc_n_dt = -pybamm.div(N_n)
dc_p_dt = -pybamm.div(N_p)

model.rhs = {
    c_n: dc_n_dt,
    c_p: dc_p_dt,
}

model.initial_conditions = {
    c_n: c_n0,
    c_p: c_p0,
}

model.boundary_conditions = {
    c_n: {"left": (pybamm.Scalar(0), "Neumann"), "right": (-j_n, "Neumann")},
    c_p: {"left": (pybamm.Scalar(0), "Neumann"), "right": (-j_p, "Neumann")},
}

# Surface concentrations
c_n_surf = pybamm.surf(c_n)
c_p_surf = pybamm.surf(c_p)

# Max concentrations
c_n_max = pybamm.Parameter("Maximum concentration in negative electrode [mol.m-3]")
c_p_max = pybamm.Parameter("Maximum concentration in positive electrode [mol.m-3]")

# Reaction rates
k_n = pybamm.Parameter("Negative electrode reaction rate [m.s-1]") 
k_p = pybamm.Parameter("Positive electrode reaction rate [m.s-1]") 

# Normalized stoichiometry
x_n_s = c_n_surf / c_n_max
x_p_s = c_p_surf / c_p_max

# OCPs
U_n = pybamm.FunctionParameter("Negative electrode OCP [V]", {"stoichiometry": x_n_s})
U_p = pybamm.FunctionParameter("Positive electrode OCP [V]", {"stoichiometry": x_p_s})

# Constants and temperature
R = pybamm.constants.R
T = pybamm.Parameter("Ambient temperature [K]")

# Exchange current densities
i_0_n = k_n * F * pybamm.sqrt(c_n) * pybamm.sqrt(c_n_surf) * pybamm.sqrt(c_n_max - c_n_surf)
i_0_p = k_p * F * pybamm.sqrt(c_p) * pybamm.sqrt(c_p_surf) * pybamm.sqrt(c_p_max - c_p_surf)

# Overpotentials
eta_n = (2 * R * T / F) * pybamm.arcsinh(j_n * F / (2 * i_0_n))
eta_p = (2 * R * T / F) * pybamm.arcsinh(j_p * F / (2 * i_0_p))

# Voltage calculation (fixed using your method)
U_n_plus_eta = pybamm.surf(U_n + eta_n)
U_p_plus_eta = pybamm.surf(U_p + eta_p)
V = U_p_plus_eta - U_n_plus_eta

model.variables = {
    "Terminal voltage [V]": V,
    "Concentration in negative particle": c_n,
    "Concentration in positive particle": c_p,
    "Surface concentration in negative particle": c_n_surf,
    "Surface concentration in positive particle": c_p_surf,
}

# Load parameters
params = pybamm.ParameterValues("Chen2020")

# Add missing or custom parameters
params.update({
    "Negative electrode diffusivity [m2.s-1]": 3.9e-14,
    "Positive electrode diffusivity [m2.s-1]": 1e-13,
    "Negative electrode reaction rate [m.s-1]": 1e-3,
    "Positive electrode reaction rate [m.s-1]": 1e-3,
    "Maximum concentration in negative electrode [mol.m-3]": 30500,
    "Maximum concentration in positive electrode [mol.m-3]": 51500,
    "Radius of negative electrode particle [m]": 1e-5,
    "Radius of positive electrode particle [m]": 1e-5,
    "Thickness of negative electrode [m]": 50e-6,
    "Thickness of positive electrode [m]": 50e-6,
    "Electrode width [m]": 0.1,
    "Electrode height [m]": 0.1,
    "Initial concentration in negative electrode [mol.m-3]": 0.8 * 30500,
    "Initial concentration in positive electrode [mol.m-3]": 0.2 * 51500,
}, check_already_exists=False)

# Geometry
r_n = pybamm.SpatialVariable("r_n", domain="negative particle")
r_p = pybamm.SpatialVariable("r_p", domain="positive particle")
geometry = {
    "negative particle": {r_n: {"min": pybamm.Scalar(0), "max": R_n}},
    "positive particle": {r_p: {"min": pybamm.Scalar(0), "max": R_p}},
}
params.process_geometry(geometry)

# Mesh and discretisation
submesh_types = {
    "negative particle": pybamm.Uniform1DSubMesh,
    "positive particle": pybamm.Uniform1DSubMesh,
}
var_pts = {"r_n": 20, "r_p": 20}
spatial_methods = {
    "negative particle": pybamm.FiniteVolume(),
    "positive particle": pybamm.FiniteVolume(),
}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
params.process_model(model)

disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# Solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 3600, 100)
solution = solver.solve(model, t)

# Plotting
if isinstance(solution, list):
    solution = solution[0]

if solution is None:
    raise RuntimeError("Solver did not return a solution.")

V_sol = solution["Terminal voltage [V]"]
c_n_surf_sol = solution["Surface concentration in negative particle"]
c_p_surf_sol = solution["Surface concentration in positive particle"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
ax1.plot(solution.t, V_sol(solution.t))
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Voltage [V]")

ax2.plot(solution.t, c_n_surf_sol(solution.t), label="c_n_surf")
ax2.plot(solution.t, c_p_surf_sol(solution.t), label="c_p_surf")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Surface concentrations")
ax2.legend()

plt.tight_layout()
plt.show()
