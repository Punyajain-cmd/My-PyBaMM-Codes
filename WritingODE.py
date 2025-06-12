import pybamm
import numpy as np
import matplotlib.pyplot as plt
# import pybamm.ODE  # Removed: pybamm has no ODE submodule
import scipy



x= pybamm.Variable("x")                                         # define a variable using pybamm.Varaible class
y = pybamm.Variable("y")
x_n = pybamm.Variable("x_n")
x_p = pybamm.Variable("x_p")


a = pybamm.Parameter("a")                                       #define a parameter using pybamm.Parameter class

P = pybamm.FunctionParameter("P", {"Time [s]" : pybamm.t})

I_t = pybamm.FunctionParameter("I_t", {"Time [s]" : pybamm.t})
U_n = pybamm.FunctionParameter("U_n", {"x_n": x_n})
U_p = pybamm.FunctionParameter("U_p", {"x_p": x_p})
Q_p = pybamm.Parameter("Q_p")
Q_n = pybamm.Parameter("Q_n")
R = pybamm.Parameter("Resistance [Ohm]")
V_t = pybamm.Variable("Terminal voltage [V]")
#V_t = pybamm.FunctionParameter("Terminal voltage [V]", {"x_n": x_n, "x_p": x_p, "U_n": U_n, "U_p": U_p, "R": R})

'''
i = pybamm.FunctionParameter("Current function [A]", {"Time [s]": pybamm.t})
x_n_0 = pybamm.Parameter("Initial negative electrode stoichiometry")
x_p_0 = pybamm.Parameter("Initial positive electrode stoichiometry")
U_p = pybamm.FunctionParameter("Positive electrode OCV", {"x_p": x_p})
U_n = pybamm.FunctionParameter("Negative electrode OCV", {"x_n": x_n})
Q_n = pybamm.Parameter("Negative electrode capacity [A.h]")
Q_p = pybamm.Parameter("Positive electrode capacity [A.h]")
R = pybamm.Parameter("Electrode resistance [Ohm]")
'''

model = pybamm.BaseModel("User-defined ODE model")
model.rhs = {x : -a * x}

#model.algebraic = {y : x-y}  we dont need algebraic equations in this case
model.initial_conditions = {x: 0}
model.variables = {"x" : x}
# model.variables = {y : y}  # we dont need this variable in this case

# above is the basic structure of a pybamm model

# Defiing the Reservoir model

model.rhs[x_n] = -1 * (I_t / Q_n)
model.rhs[x_p] = 1 * (I_t / Q_p)
model.initial_conditions = {x_n: 0.5, x_p: 0.5}
model.variables = {
	"x_n": x_n,
	"V_t": U_p - U_n - R * I_t,
    "x_p": x_p
}
# now we have defined all the necessary components of the Reservoir model

# visualizing the expressions of the model

model.rhs[x_n].visualise("x_n rhs.png")
model.rhs[x_p].visualise("x_p rhs.png")

print("x_n rhs expression: ", model.rhs[x_n])
# above we are printing the expression as a string
print("x_p rhs expression: ", model.rhs[x_p])


# the variable children returns the children node value of the parent node
model.rhs[x_n].children[0].visualise("x_n rhs children.png")
model.rhs[x_n].children[0]

model.rhs[x_n].children[0].children[0].children[0]

# defining events in the model

stop_at_t_equal_3 = pybamm.Event("Stop at t = 3", pybamm.t - 3)   # This event will trigger when the time reaches 3 seconds
model.events = [stop_at_t_equal_3]

# defining the events for the reservoir model
stop_at_x_n_equal_0 = pybamm.Event("Stop at x_n = 0", x_n)
stop_at_x_n_equal_1 = pybamm.Event("Stop at x_n =1", x_n - 1)

stop_at_x_p_equal_0 =pybamm.Event("Stop at x_p = 0", x_p)
stop_at_x_p_equal_1 = pybamm.Event("Stop at x_p =1", x_p - 1)

model.events = [stop_at_x_n_equal_0, stop_at_x_n_equal_1, stop_at_x_p_equal_0, stop_at_x_p_equal_1]

'''
model.events = [
    pybamm.Event("Minimum negative stoichiometry", x_n - 0),
    pybamm.Event("Maximum negative stoichiometry", 1 - x_n),
    pybamm.Event("Minimum positive stoichiometry", x_p - 0),
    pybamm.Event("Maximum positive stoichiometry", 1 - x_p),
]
'''

# defining the values of the parameters
parameter_values = pybamm.ParameterValues({
    "a": 1,
    "Q_n": 1,
    "Q_p": 1,   
    "Resistance [Ohm]": 1,
    "U_n": 0.5,  # This is a constant OCV of 0.5 V for the negative electrode
    "U_p": 0.5,  # This is a constant OCV of 0.5 V for the positive electrode
    "Time [s]": 0,  # Initial time is set to 0 seconds
})

# defining the values of the function parameters

parameter_values = pybamm.ParameterValues({
    "I_t": lambda t: 1.0 + 0.5 * np.sin(100 * t)
})  # Constant current of 1 A

'''
param = pybamm.ParameterValues({
    "Current function [A]": lambda t: 1 + 0.5 * pybamm.sin(100*t),
    "Initial negative electrode stoichiometry": 0.9,
    "Initial positive electrode stoichiometry": 0.1,
    "Negative electrode capacity [A.h]": 1,
    "Positive electrode capacity [A.h]": 1,
    "Electrode resistance [Ohm]": 0.1,
    "Positive electrode OCV": nmc_LGM50_ocp_Chen2020,
    "Negative electrode OCV": graphite_LGM50_ocp_Chen2020,
})


This is a simple example of how to define Paramerter Values in Pybamm
'''

sim = pybamm.Simulation(model, parameter_values=parameter_values)
sim.solve([0,1])
sim.plot(["V_t", "Q_n", "Q_p"], time_unit = "s")
