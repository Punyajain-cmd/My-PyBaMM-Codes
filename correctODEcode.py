import pybamm
import numpy as np
import matplotlib.pyplot as plt

# Define variables
x = pybamm.Variable("x")
y = pybamm.Variable("y")
x_n = pybamm.Variable("x_n")
x_p = pybamm.Variable("x_p")

# Define parameters
a = pybamm.Parameter("a")
Q_n = pybamm.Parameter("Q_n")
Q_p = pybamm.Parameter("Q_p")
R = pybamm.Parameter("Resistance [Ohm]")

# Define function parameters
I_t = pybamm.FunctionParameter("I_t", {"Time [s]": pybamm.t})
U_n = pybamm.FunctionParameter("U_n", {"x_n": x_n})
U_p = pybamm.FunctionParameter("U_p", {"x_p": x_p})

# Create model
model = pybamm.BaseModel()

# Define ODEs
model.rhs = {
    x: -a * x,
    x_n: -I_t / Q_n,
    x_p: I_t / Q_p
}

# Define initial conditions
model.initial_conditions = {
    x: pybamm.Scalar(0),
    x_n: pybamm.Scalar(2.5),
    x_p: pybamm.Scalar(2.5)
}

# Define output variables
V_t = U_p - U_n - R * I_t
model.variables = {
    "x": x,
    "x_n": x_n,
    "x_p": x_p,
    "V_t": V_t
}

# Define events
model.events = [
    pybamm.Event("Stop at x_n = 0", x_n),
    pybamm.Event("Stop at x_n = 1", x_n - 1),
    pybamm.Event("Stop at x_p = 0", x_p),
    pybamm.Event("Stop at x_p = 1", x_p - 1)
]

# Define parameter values with both constants and functions
parameter_values = pybamm.ParameterValues({
    "a": 1,
    "Q_n": 1,
    "Q_p": 1,
    "Resistance [Ohm]": 1,
    "U_n": lambda x_n: 0.5,
    "U_p": lambda x_p: 0.5,
    "I_t": lambda t: 1.0 + 0.5 * np.sin(100 * t)
})

# Create and run simulation
sim = pybamm.Simulation(model, parameter_values=parameter_values)
#sim.solve(t_eval=np.linspace(0, 10, 100))
sim.solve([0, 1])
sim.plot(["V_t", "x_n", "x_p"])
