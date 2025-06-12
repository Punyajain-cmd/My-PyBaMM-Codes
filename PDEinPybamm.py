import pybamm
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd

model = pybamm.BaseModel("Model with PDEs")

c = pybamm.Variable("Concentration", domain="Negative electrode")    # Here the concentration is defined as a variable name "c"                                                                  # within the negative electrode domain

c_s =pybamm.Variable("Concentration in solid", domain="Negative electrode")

c_e = pybamm.Variable("Concentration in electrolyte", domain="Negative electrode")

# now to define the equations it is recommended to define intermediate equations and then using them we can define the final equations

N = -pybamm.grad(c)
dcdt = -pybamm.div(N)

N_s = -pybamm.grad(c_s)
dc_sdt = -pybamm.div(N_s)

N_e = -pybamm.grad(c_e)
dc_edt = -pybamm.div(N_e)


model.rhs = {
    c: dcdt,
    c_s: dc_sdt,
    c_e: dc_edt
}

# IN abve equations, in order to define model equations we have used the flux equation as an intermediate equation.
# All the above equations are defined in the negative particle domain and all of them are "Poissons Equations"



# starting with the initial conditions and the boundary conditions


model.initial_conditions = {
    c: pybamm.Scalar(1),
    c_s: pybamm.Scalar(1),
    c_e: pybamm.Scalar(1)
}

# Boundary conditions

model.boundary_conditions = {
    c: {"left": (pybamm.Scalar(0), "Neumann"), "right": (pybamm.Scalar(2), "Neumann")},
    c_s: {"left": (pybamm.Scalar(0), "Neumann"), "right": (pybamm.Scalar(1), "Neumann")},
    c_e: {"left": (pybamm.Scalar(0), "Neumann"), "right": (pybamm.Scalar(1), "Neumann")}
    }

# Here the boundary condtions take a constant value but un general case they can be a function as well.

# Variables to be used in the model
model.variables = {
    "Concentration": c,
    "Concentration in solid": c_s,
    "Concentration in electrolyte": c_e,
    "Flux": N,
    "Flux in solid": N_s,
    "Flux in electrolyte": N_e
}

# Now we need to define the Geometery and the mesh for the model
r = pybamm.SpatialVariable("r", domain=["Negative electrode"], coord_sys="cylindrical polar")    # Here we are defining the spatial variable "r" in the negative electrode domain in cylindrical polar coordinates.

# Defining the geometry


geometry = {
    "Negative electrode": {
        "r_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
    }
}

# Mesh and Discretisation
submesh_types = {"Negative electrode": pybamm.Uniform1DSubMesh}
var_pts = {"r_n": 30}  # 30 points for spatial variable "r_n"
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)


'''
The above SubMesh type does not require a parameter value, So we can pass it directly.
But some SunMesh types requires a parameter value like "pybamm.exponential1DSubMesh" which cluster the points close to one of the boundaries or both depending on the parameter value.

For example, to create a mesh with more nodes clustered to the right (i.e. the surface in the particle problem)
'''
# exp_mesh = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": "right", "stretch": 2})


# Now we can discretise the model using the mesh we have created
spatial_methods = {"Negative electrode": pybamm.FiniteVolume()}
discretised_model = pybamm.Discretisation(mesh, spatial_methods)
discretised_model.process_model(model)


# Now we can solve the model using the discretised model

'''
sim = pybamm.Simulation(discretised_model)
sim.solve(t_eval=np.linspace(0, 10, 100))
sim.plot("Concentration", time_unit="s")
'''

solver = pybamm.ScipySolver()
t = np.linspace(0, 10, 100)
solution = solver.solve(model, t)

# If solution is a list, get the first Solution object
if isinstance(solution, list):
    solution = solution[0]

if solution is None:
    raise RuntimeError("Solver did not return a solution. Please check the model setup and solver configuration.")

# post-process, so that the solution can be called at any time t or space r
# (using interpolation)
c = solution["Concentration"]

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(solution.t, c(solution.t, r=1))
ax1.set_xlabel("t")
ax1.set_ylabel("Surface concentration")
r = np.linspace(0, 1, 100)
ax2.plot(r, c(t=0.5, r=r))
ax2.set_xlabel("r")
ax2.set_ylabel("Concentration at t=0.5")
plt.tight_layout()
plt.show()
