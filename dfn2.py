import pybamm
import matplotlib.pyplot as plt
import numpy as np

# Initialize model
model = pybamm.BaseModel("Revised DFN Model")

neg_domain = "negative electrode"
sep_domain = "separator" 
pos_domain = "positive electrode"
particle_domains = {
    "negative": "negative particle",
    "positive": "positive particle"
}

# Solid-phase variables (electrode particles)
c_s_n = pybamm.Variable("Negative particle concentration", domain=particle_domains["negative"])
c_s_p = pybamm.Variable("Positive particle concentration", domain=particle_domains["positive"])
D_s_n = pybamm.Parameter("Negative particle diffusivity")
D_s_p = pybamm.Parameter("Positive particle diffusivity")
R_n = pybamm.Parameter("Negative particle radius")
R_p = pybamm.Parameter("Positive particle radius")

# Electrolyte variables
c_e = pybamm.Variable("Electrolyte concentration", domain=[neg_domain, sep_domain, pos_domain])
phi_e = pybamm.Variable("Electrolyte potential", domain=[neg_domain, sep_domain, pos_domain])

# Electrode potentials
phi_s_n = pybamm.Variable("Negative electrode potential", domain=neg_domain)
phi_s_p = pybamm.Variable("Positive electrode potential", domain=pos_domain)

# Current densities
j_n = pybamm.Variable("Negative electrode reaction current density", domain=neg_domain)
j_p = pybamm.Variable("Positive electrode reaction current density", domain=pos_domain)

# Common parameters
F = pybamm.constants.F
R = pybamm.constants.R
T = pybamm.Parameter("Temperature [K]")
epsilon_e = pybamm.Parameter("Electrolyte porosity")
t_plus = pybamm.Parameter("Cation transference number")

# Solid phase equations
N_s_n = -D_s_n * pybamm.grad(c_s_n)
dcdt_s_n = -pybamm.div(N_s_n) / R_n**2

N_s_p = -D_s_p * pybamm.grad(c_s_p)
dcdt_s_p = -pybamm.div(N_s_p) / R_p**2

# Boundary conditions for solid phase
model.boundary_conditions = {
    c_s_n: {
        "left": (0, "Neumann"),
        "right": (-j_n/(F * D_s_n), "Neumann"),
    },
    c_s_p: {
        "left": (0, "Neumann"),
        "right": (-j_p/(F * D_s_p), "Neumann"),
    }
}

# Electrolyte diffusion
D_e_eff = pybamm.Parameter("Effective electrolyte diffusivity")
N_e = -D_e_eff * epsilon_e**1.5 * pybamm.grad(c_e)

# Combine j_n and j_p for full domain
j = pybamm.concatenation(
    j_n,
    pybamm.FullBroadcast(0, "separator", auxiliary_domains={"secondary": "current collector"}),
    j_p
)

dcdt_e = (pybamm.div(N_e) + (1 - t_plus) * j/F) / epsilon_e

# Electrolyte potential
kappa_eff = pybamm.Parameter("Effective ionic conductivity")
phi_e_equation = pybamm.div(kappa_eff * pybamm.grad(phi_e)) + pybamm.div(kappa_eff * (2 * R * T / F) * (1 - t_plus) * pybamm.grad(pybamm.log(c_e))) + j

# Electrode potentials
sigma_n = pybamm.Parameter("Negative electrode conductivity")
sigma_p = pybamm.Parameter("Positive electrode conductivity")

i_s_n = -sigma_n * pybamm.grad(phi_s_n)
i_s_p = -sigma_p * pybamm.grad(phi_s_p)

# Butler-Volmer kinetics

k_n = pybamm.Parameter("Negative electrode rate constant")
k_p = pybamm.Parameter("Positive electrode rate constant")
alpha = 0.5


phi_e_n = pybamm.Variable("Electrolyte potential (n)", domain="negative electrode")
phi_e_s = pybamm.Variable("Electrolyte potential (s)", domain="separator")
phi_e_p = pybamm.Variable("Electrolyte potential (p)", domain="positive electrode")

phi_e = pybamm.concatenation(phi_e_n, phi_e_s, phi_e_p)


c_s_n_surf = pybamm.surf(c_s_n)
U_n_func = pybamm.FunctionParameter("Negative electrode OCP [V]", {"c_s_n": pybamm.surf(c_s_n)})
eta_n = phi_s_n - phi_e_n - U_n_func
pybamm.surf(c_s_p)
c_s_p_surf = pybamm.surf(c_s_p)
U_p_func = pybamm.FunctionParameter("Positive electrode OCP [V]", {"c_s_p": pybamm.surf(c_s_p)})
eta_p = phi_s_p - phi_e_p - U_p_func

c_e_n=pybamm.PrimaryBroadcast(c_e, neg_domain)
# Correct use of PrimaryBroadcast for c_e
j_n = (
    k_n * c_e_n**alpha
    * (1 - c_s_n_surf / pybamm.Parameter("Max negative concentration"))**alpha
    * (
        pybamm.exp(alpha * F * eta_n / (R * T))
        - pybamm.exp(-alpha * F * eta_n / (R * T))
    )
)

j_p = (
    k_p * pybamm.PrimaryBroadcast(c_e, pos_domain)**alpha
    * (1 - c_s_p_surf / pybamm.Parameter("Max positive concentration"))**alpha
    * (
        pybamm.exp(alpha * F * eta_p / (R * T))
        - pybamm.exp(-alpha * F * eta_p / (R * T))
    )
)



model.boundary_conditions.update({
    phi_s_n: {"left": (pybamm.Scalar(0), "Dirichlet")},
    phi_s_p: {"right": (pybamm.Parameter("Applied voltage"), "Dirichlet")},
    c_e: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (pybamm.Scalar(0), "Neumann")
    }
})

# Voltage calculation
V = pybamm.boundary_value(phi_s_p, "right") - pybamm.boundary_value(phi_s_n, "left") - pybamm.boundary_value(phi_e, "right")

model.variables = {
    "Negative particle concentration": c_s_n,
    "Positive particle concentration": c_s_p,
    "Electrolyte concentration": c_e,
    "Terminal voltage": V,
    "Negative current density": j_n,
    "Positive current density": j_p
}

model.rhs = {
    c_s_n: dcdt_s_n,
    c_s_p: dcdt_s_p,
    c_e: dcdt_e,
    phi_e: phi_e_equation
}

model.initial_conditions = {
    c_s_n: pybamm.Parameter("Initial negative concentration"),
    c_s_p: pybamm.Parameter("Initial positive concentration"),
    c_e: pybamm.Parameter("Initial electrolyte concentration"),
    phi_e: pybamm.Scalar(0)
}

params = pybamm.ParameterValues("Chen2020")
params.update({
    "Negative electrode diffusivity [m2.s-1]": 3.9e-14,
    "Positive electrode diffusivity [m2.s-1]": 1e-13,
    "Negative electrode reaction rate [m.s-1]": 1e-3,
    "Positive electrode reaction rate [m.s-1]": 1e-3,
    "Maximum concentration in negative electrode [mol.m-3]": 30500,
    "Maximum concentration in positive electrode [mol.m-3]": 51500,
    "Radius of negative electrode particle [m]": 1e-5,
    "Radius of positive electrode particle [m]": 1e-5,
    "Negative electrode thickness [m]": 50e-6,
    "Separator thickness [m]": 25e-6,
    "Positive electrode thickness [m]": 50e-6,
    "Initial concentration in negative electrode [mol.m-3]": 0.8 * 30500,
    "Initial concentration in positive electrode [mol.m-3]": 0.2 * 51500,
}, check_already_exists=False)

L_n = pybamm.Parameter("Negative electrode thickness [m]")
L_s = pybamm.Parameter("Separator thickness [m]")
L_p = pybamm.Parameter("Positive electrode thickness [m]")

x = pybamm.SpatialVariable("x", domain=["negative electrode", "separator", "positive electrode"])
r_n = pybamm.SpatialVariable("r_n", domain="negative particle")
r_p = pybamm.SpatialVariable("r_p", domain="positive particle")

geometry = {
    "negative electrode": {x: {"min": 0, "max": L_n}},
    "separator": {x: {"min": L_n, "max": L_n + L_s}},
    "positive electrode": {x: {"min": L_n + L_s, "max": L_n + L_s + L_p}},
    "negative particle": {r_n: {"min": 0, "max": 1}},
    "positive particle": {r_p: {"min": 0, "max": 1}}
}

var_pts = {
    "x": 80,
    "r_n": 20,
    "r_p": 20
}

submesh_types = {
    "negative electrode": pybamm.Uniform1DSubMesh,
    "separator": pybamm.Uniform1DSubMesh,
    "positive electrode": pybamm.Uniform1DSubMesh,
    "negative particle": pybamm.Uniform1DSubMesh,
    "positive particle": pybamm.Uniform1DSubMesh
}

spatial_methods = {
    "negative electrode": pybamm.FiniteVolume(),
    "separator": pybamm.FiniteVolume(),
    "positive electrode": pybamm.FiniteVolume(),
    "negative particle": pybamm.FiniteVolume(),
    "positive particle": pybamm.FiniteVolume()
}

mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)
params.process_model(model)

solver = pybamm.ScipySolver()
t = np.linspace(0, 3600, 100)
solution = solver.solve(model, t)

if isinstance(solution, list):
    solution = solution[0]

if solution is None:
    raise RuntimeError("Solver did not return a solution.")

c_s_n_sol = solution["Negative particle concentration"]
c_e_sol = solution["Electrolyte concentration"]
V_sol = solution["Terminal voltage"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

r_n = params["Radius of negative electrode particle [m]"]
ax1.plot(solution.t, c_s_n_sol(solution.t, r=r_n))
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Surface Concentration (mol/m³)")
ax1.set_title("Negative Particle Surface Concentration")

x_plot = np.linspace(0, 1, 100)
ax2.plot(x_plot, c_e_sol(1000, x=x_plot))
ax2.set_xlabel("Normalized Position")
ax2.set_ylabel("Electrolyte Concentration (mol/m³)")
ax2.set_title("Electrolyte Profile @ 1000s")

ax3.plot(solution.t, V_sol(solution.t))
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Terminal Voltage (V)")
ax3.set_title("Discharge Curve")

plt.tight_layout()
plt.show()
