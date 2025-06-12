import pybamm
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd

# Initialize model
# Initialize model
model = pybamm.BaseModel("Revised DFN Model")
params = pybamm.ParameterValues("Chen2020")
#print(list(params.keys()))


neg_domain = "negative electrode"
sep_domain = "separator" 
pos_domain = "positive electrode"
particle_domains = {
    "negative": "negative particle",
    "positive": "positive particle"
}
auxillary_domains = {
    "negative" : "negative electrode",
    "positive" : "Positive electrode"
}


# Solid-phase variables (electrode particles)
c_s_n = pybamm.Variable("Negative particle concentration", domain=particle_domains["negative"], auxiliary_domains={"secondary": auxillary_domains["negative"]})
c_s_p = pybamm.Variable("Positive particle concentration", domain=particle_domains["positive"], auxiliary_domains={"secondary": auxillary_domains["positive"]})
# Change all parameter definitions to include units
D_s_n = pybamm.Parameter("Negative particle diffusivity [m2.s-1]")  
D_s_p = pybamm.Parameter("Positive particle diffusivity [m2.s-1]")
R_n = pybamm.Parameter("Negative particle radius [m]")
R_p = pybamm.Parameter("Positive particle radius [m]")


# Electrolyte variables
c_e_n = pybamm.Variable("Negative Electrolyte concentration", domain=neg_domain)
c_e_p = pybamm.Variable("Positive Electrolyte concentration", domain=pos_domain)
c_e_s  = pybamm.Variable("Separator Electrolyte concentration", domain=sep_domain)
c_e = pybamm.concatenation(c_e_n,c_e_p,c_e_s)

phi_e_n = pybamm.Variable("Negative Electrolyte potential", domain=neg_domain)
phi_e_p = pybamm.Variable("Positive Electrolyte potential", domain=pos_domain)
phi_e_s = pybamm.Variable("Separator Electrolyte potential", domain=sep_domain)

phi_e = pybamm.concatenation(phi_e_n,phi_e_p,phi_e_s)

# Electrode potentials
phi_s_n = pybamm.Variable("Negative electrode potential", domain="negative electrode")
phi_s_p = pybamm.Variable("Positive electrode potential", domain="positive electrode")





# Common parameters
F = pybamm.constants.F
R = pybamm.constants.R
T = pybamm.Parameter("Initial temperature [K]")
t_plus = pybamm.Parameter("Cation transference number")



# ========== Butler-Volmer Kinetics ==========
k_n = pybamm.Scalar(1e-10)  # Example value in [m/s]
k_p = pybamm.Scalar(3e-11)  # Example value in [m/s]



alpha = 0.5  # Symmetric transfer coefficients


c_s_n_surf = pybamm.surf(c_s_n)
c_s_p_surf = pybamm.surf(c_s_p)
c_s_n_max = params["Maximum concentration in negative electrode [mol.m-3]"]
c_s_p_max = params["Maximum concentration in positive electrode [mol.m-3]"]
x_n = c_s_n_surf / c_s_n_max
x_p = c_s_p_surf / c_s_p_max


# Open-circuit potentials (scalar parameters)
U_n = pybamm.FunctionParameter("Negative electrode OCP [V]", {"Electrode stoichiometry": x_n})
U_p = pybamm.FunctionParameter("Positive electrode OCP [V]", {"Electrode stoichiometry": x_p})



# Now subtract on same domain (electrode)
U_n_broadcast = pybamm.PrimaryBroadcast(U_n, ["negative particle"])
eta_n = phi_s_n - phi_e_n - U_n


U_p_broadcast = pybamm.PrimaryBroadcast(U_p, ["positive particle"])
eta_p = phi_s_p - phi_e_p - U_p




j_n = (
    k_n * (c_e_n*c_s_n_surf)**alpha 
    * (c_s_n_max - c_s_n_surf)**alpha
    * (pybamm.exp(alpha * F * eta_n / (R * T)) 
       - pybamm.exp(-alpha * F * eta_n / (R * T)))
)


j_p = (
    k_p * (c_e_p*c_s_p_surf)**alpha
    * (c_s_p_max - c_s_p_surf)**alpha
    * (pybamm.exp(alpha * F * eta_p / (R * T)) 
       - pybamm.exp(-alpha * F * eta_p / (R * T)))
)


j = pybamm.concatenation(
    j_n,  # in negative electrode
    pybamm.FullBroadcast(0, "separator", auxiliary_domains={"secondary": "current collector"}),
    j_p   # in positive electrode
)


# ========== Solid Phase Equations ==========
# Negative particle diffusion
N_s_n = -D_s_n * pybamm.grad(c_s_n)
dcdt_s_n = -pybamm.div(N_s_n) / R_n**2

# Positive particle diffusion
N_s_p = -D_s_p * pybamm.grad(c_s_p)
dcdt_s_p = -pybamm.div(N_s_p) / R_p**2

# Boundary conditions for solid phase
model.boundary_conditions = {
    c_s_n: {
        "left": (0, "Neumann"),  # Symmetry at center
        "right": (-j_n/(F * D_s_n), "Neumann"),  # Flux at surface
    },
    c_s_p: {
        "left": (0, "Neumann"),
        "right": (-j_p/(F * D_s_p), "Neumann"),
    }
}

# Electrolyte diffusion
epsilon_e = pybamm.Parameter("Negative electrode porosity")

# Electrolyte diffusion (domain-specific effective diffusivities)
D_e_eff_n = pybamm.FunctionParameter(
    "Electrolyte diffusivity [m2.s-1]",
    {"Electrolyte concentration": c_e_n, "Temperature [K]": T}
)
D_e_eff_s = pybamm.FunctionParameter(
    "Electrolyte diffusivity [m2.s-1]",
    {"Electrolyte concentration": c_e_s, "Temperature [K]": T}
)
D_e_eff_p = pybamm.FunctionParameter(
    "Electrolyte diffusivity [m2.s-1]",
    {"Electrolyte concentration": c_e_p, "Temperature [K]": T}
)

N_e_n = -D_e_eff_n * epsilon_e**1.5 * pybamm.grad(c_e_n)
N_e_s = -D_e_eff_s * epsilon_e**1.5 * pybamm.grad(c_e_s)
N_e_p = -D_e_eff_p * epsilon_e**1.5 * pybamm.grad(c_e_p)


# Electrolyte concentration time derivative (both electrodes contribute)
dcdt_e_n = (pybamm.div(N_e_n) + (1 - t_plus)*(j_n)/F) / epsilon_e
dcdt_e_p = (pybamm.div(N_e_p) + (1 - t_plus)*(j_p)/F) / epsilon_e


# Electrolyte potential
# Negative electrode parameters
epsilon_n = pybamm.Parameter("Negative electrode porosity")
brugg_n = pybamm.Parameter("Negative electrode Bruggeman coefficient (electrolyte)")
kappa_n = pybamm.FunctionParameter(
    "Electrolyte conductivity [S.m-1]",
    {"Electrolyte concentration": c_e_n, "Temperature [K]": T}
)
kappa_eff_n = kappa_n * epsilon_n ** brugg_n


# Positive electrode parameters
epsilon_p = pybamm.Parameter("Positive electrode porosity")
brugg_p = pybamm.Parameter("Positive electrode Bruggeman coefficient (electrolyte)")
kappa_p = pybamm.FunctionParameter(
    "Electrolyte conductivity [S.m-1]",
    {"Electrolyte concentration": c_e_p, "Temperature [K]": T}
)
kappa_eff_p = kappa_p * epsilon_p ** brugg_p


# Electrolyte potential equations (domain-specific)
phi_e_n_equation = pybamm.div(kappa_eff_n * pybamm.grad(phi_e_n)) + \
                   pybamm.div(kappa_eff_n * (2 * R * T / F) * (1 - t_plus) * pybamm.grad(pybamm.log(c_e_n))) + \
                   j_n

phi_e_p_equation = pybamm.div(kappa_eff_p * pybamm.grad(phi_e_p)) + \
                   pybamm.div(kappa_eff_p * (2 * R * T / F) * (1 - t_plus) * pybamm.grad(pybamm.log(c_e_p))) + \
                   j_p


# ========== Electrode Potentials ==========

sigma_n = pybamm.Parameter("Negative electrode conductivity [S.m-1]")
sigma_p = pybamm.Parameter("Positive electrode conductivity [S.m-1]")

# Ohm's law in solid phase
j_s_n = pybamm.div(-sigma_n * pybamm.grad(phi_s_n))
j_s_p = pybamm.div(-sigma_p * pybamm.grad(phi_s_p))



model.boundary_conditions = {
    phi_s_n: {"left": (pybamm.Scalar(0), "Dirichlet")},  # Grounded at current collector
    phi_s_p: {"right": (pybamm.Scalar(4), "Dirichlet")},
    c_e_n:  {"left": (pybamm.Scalar(0), "Neumann")},  # Insulated at negative current collector
    c_e_p:  {"right": (pybamm.Scalar(0), "Neumann")}  # Insulated at positive current collector

}


# separator equations

# For electrolyte concentration in separator
dcdt_e_s = (pybamm.div(N_e_s)) / epsilon_e  # No reaction current in separator

# Separator
epsilon_s = pybamm.Parameter("Separator porosity")
brugg_s = pybamm.Parameter("Separator Bruggeman coefficient (electrolyte)")
kappa_s = pybamm.FunctionParameter(
    "Electrolyte conductivity [S.m-1]",
    {"Electrolyte concentration": c_e_s, "Temperature [K]": T}
)
kappa_eff_s = kappa_s * epsilon_s ** brugg_s

# For electrolyte potential in separator
phi_e_s_equation = pybamm.div(kappa_eff_s * pybamm.grad(phi_e_s)) + \
                 pybamm.div(kappa_eff_s * (2 * R * T / F) * (1 - t_plus) * \
                 pybamm.grad(pybamm.log(c_e_s)))
                               
 
 #separator boundary Condition
                 
model.boundary_conditions[c_e_s] = {
    "left": (pybamm.Scalar(0), "Neumann"),
    "right": (pybamm.Scalar(0), "Neumann")
}

model.boundary_conditions[phi_e_s] = {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (pybamm.Scalar(0), "Neumann")
    }


# Voltage calculation
V = pybamm.boundary_value(phi_s_p, "right") - pybamm.boundary_value(phi_s_n, "left") - pybamm.boundary_value(phi_e, "right")

# Add variables to model
model.variables = {
    "Negative particle concentration": c_s_n,
    "Positive particle concentration": c_s_p,
    "Electrolyte concentration": c_e,
    "Terminal voltage": V,
    "Negative current density": j_n,
    "Positive current density": j_p
}

# Add governing equations
model.rhs = {
    c_s_n: dcdt_s_n,
    c_s_p: dcdt_s_p,
    c_e_p: dcdt_e_p,
    c_e_n: dcdt_e_n,
    c_e_s: dcdt_e_s
}


model.algebraic = {
    phi_s_n: j_s_n,
    phi_s_p: j_s_p,
    phi_e_s: phi_e_s_equation,
    phi_e_n: phi_e_n_equation,  
    phi_e_p: phi_e_p_equation,  
    
}


# Set initial conditions
model.initial_conditions = {
    c_s_n: pybamm.Parameter("Initial concentration in negative electrode [mol.m-3]"),
    c_s_p: pybamm.Parameter("Initial concentration in positive electrode [mol.m-3]"),
    c_e_n: pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]"),
    c_e_p: pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]"),
    c_e_s: pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]"),
    phi_e_p: pybamm.Scalar(0),
    phi_e_s: pybamm.Scalar(0),
    phi_e_n: pybamm.Scalar(0),
    phi_s_n: pybamm.Scalar(0),  # Initial guess for negative electrode potential
    phi_s_p: pybamm.Scalar(4)  # Matches Dirichlet BC
}


params.update({
    "Electrolyte diffusivity [m2.s-1]": 2.6e-10,
    "Electrolyte conductivity [S.m-1]": 1.1,
    # Electrode parameters
    "Negative electrode diffusivity [m2.s-1]": 3.9e-14,
    "Positive electrode diffusivity [m2.s-1]": 1e-13,
    "Negative electrode reaction rate [m.s-1]": 1e-3,
    "Positive electrode reaction rate [m.s-1]": 1e-3,
    "Maximum concentration in negative electrode [mol.m-3]": 30500,
    "Maximum concentration in positive electrode [mol.m-3]": 51500,
    "Radius of negative electrode particle [m]": 1e-5,
    "Radius of positive electrode particle [m]": 1e-5,
    "Negative electrode porosity": 0.3,  # From Chen2020 parameters
    "Separator porosity": 0.4,
    "Positive electrode porosity": 0.3,
    
    # Geometry parameters
    "Negative electrode thickness [m]": 50e-6,
    "Separator thickness [m]": 25e-6,  # Added separator thickness
    "Positive electrode thickness [m]": 50e-6,
    
    # Initial conditions
    "Initial concentration in negative electrode [mol.m-3]": 0.8 * 30500,
    "Initial concentration in positive electrode [mol.m-3]": 0.2 * 51500,
}, check_already_exists=False)

params.process_model(model)
params = pybamm.ParameterValues("Chen2020")
sim = pybamm.Simulation(model, parameter_values=params)
t_eval = np.linspace(0,3600,100)
sim.solve(0, t_eval)
sim.plot()

