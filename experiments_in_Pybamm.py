import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy 
import scipy.optimize


#pybamm.set_logging_level("NOTICE")                 # set the logging level to Notice to avoid too many messages
# there are many logging levels in pybamm, you can set it to "Debug", "Info", "Warning", "Error", "Critical" or "None" or "Notice"
# The levels of logging in PyBaMM (from most to least detailed) are:
# - "DEBUG"    (detailed diagnostic info)
# - "INFO"     (general progress information)
# - "NOTICE"   (important messages, but less than INFO)
# - "WARNING"  (something might be wrong)
# - "ERROR"    (something went wrong)
#
# "Notice" is a good balance: it shows essential messages without being too verbose.
# Define an experiment using the PyBaMM Experiment class


'''
experiment = pybamm.Experiment([(    "Discharge at 1C until 2.5V",
    "Rest for 1 hour",
    "Charge at 1C until 4.2V",
    "Hold at 4.2V until C/50")]*3)
# this will perform the experiment for one cycle only
# we can multiply the list by an integer to perform the experiment that many times



model = pybamm.lithium_ion.SPMe()
parameters = pybamm.ParameterValues("Chen2020")
simulation = pybamm.Simulation(model, parameter_values=parameters, experiment=experiment)
#simulation.solve([0, 3600])  we dont need to specify the time here because the experiment automatically defines the times steps based on the experiment steps
simulation.solve()  # Solve the experiment

# Plot the results
simulation.plot(
    output_variables=[
        "Terminal voltage [V]",
        "Current [A]",
    ]
)

simulation.solution # this will give us the solution for each cycle
simulation.solution.cycles[1]  # following the zero-based indexing, this will give us the solution for the first cycle
simulation.solution.cycles[1].plot(["Terminal voltage [V]", "Current [A]"]) # this will plot the terminal voltage for the first cycle

simulation.solution.cycles[1].steps # this shows all the steps in the first cycle (based on the experiment defined above and based on the zero indexing)

simulation.solution.cycles[1].steps[0].plot(["Terminal voltage [V]", "Current [A]"]) # this will plot a particular step in the first cycle
'''


# now we will get the power in the picture as well


#pybamm.ParameterValues("Nominal cell capacity [A.h]")
# this will give us the nominal cell capacity in Ah, we can use it to calculate the C-rate
# for example, if the nominal capacity is 2 Ah, then 1C is 2 

'''
experiment = pybamm.Experiment([(
"Discharge at 5A until 2.5V",                       # here round  brackets are used to  define one complete cycle with three steps
"Charge at 15W until 4.2V",                         # if we remove the round brackets, it will be considered it as three separate cycle 
"Hold at 4.2V until 0.01A")])
model = pybamm.lithium_ion.SPMe()
parameters = pybamm.ParameterValues("Chen2020")

simulation1 = pybamm.Simulation(model, parameter_values=parameters, experiment=experiment)
simulation1.solve()
simulation1.plot(["Terminal voltage [V]", "Current [A]","Terminal power [W]"])  # Plot the terminal voltage, current, and power


# This will plot the terminal voltage and current for the experiment defined above
simulation1.solution.cycles[0].steps[0].plot(["Terminal voltage [V]", "Current [A]", "Terminal power [W]"])  # Plot the first step of the first cycle

# so this was the introduction of the terminal power into the plots,
'''



'''
# now we will use the terminal power to calculate the energy and plot it
experiment = pybamm.Experiment([(
    "Discharge at 5A until 2.5V",
    "Charge at 15W until 4.2V",
    "Hold at 4.2V until 0.01A")])
model = pybamm.lithium_ion.SPMe()
parameters = pybamm.ParameterValues("Chen2020")
simulation2 = pybamm.Simulation(model, parameter_values=parameters, experiment=experiment)
simulation2.solve()
# Calculate the energy from the terminal power
time = simulation2.solution.t
power = simulation2.solution["Terminal power [W]"](time)  # Get the terminal power as a function of time, we extracting the terminal power from the solution
energy = np.trapz(power, time)  # Integrate power over time to get energy
plt.figure()
plt.plot(time, power, label="Terminal Power [W]")
plt.xlabel("Time [s]")
plt.ylabel("Power [W]")
plt.title("Terminal Power Over Time")
plt.legend()
plt.grid()
plt.figure()
plt.plot(time, np.cumsum(power) * (time[1] - time[0]), label="Cumulative Energy [J]")
plt.xlabel("Time [s]")
plt.ylabel("Cumulative Energy [J]")
plt.title("Cumulative Energy Over Time")
plt.legend()
plt.grid()
plt.show()
# Plot the results
simulation2.plot(["Terminal voltage [V]", "Current [A]", "Terminal power [W]"])
# Plot the first step of the first cycle
simulation2.solution.cycles[0].steps[0].plot(["Terminal voltage [V]", "Current [A]", "Terminal power [W]"])
# Show the cumulative energy plot
plt.figure()
plt.plot(time, np.cumsum(power) * (time[1] - time[0]), label="Cumulative Energy [J]")
plt.xlabel("Time [s]")
plt.ylabel("Cumulative Energy [J]")
plt.title("Cumulative Energy Over Time")
plt.legend()
plt.grid()
plt.show()
# Show the terminal power plot
plt.figure()
plt.plot(time, power, label="Terminal Power [W]")
plt.xlabel("Time [s]")
plt.ylabel("Power [W]")
plt.title("Terminal Power Over Time")
plt.legend()
plt.grid()
plt.show()
# Show the terminal voltage and current plot
simulation2.solution.cycles[0].steps[0].plot(["Terminal voltage [V]", "Current [A]"])
# Show the terminal voltage and current plot for the first step of the first cycle
'''

# now we will focus on adding pulses to our experiment


model = pybamm.lithium_ion.SPMe()
parameters = pybamm.ParameterValues("Chen2020")
'''
cycles = [
    "Discharge at 1C until 2.5V"
] + [
    "Charge at 2C for 1 minute (1 second period)",
    "Rest for 5 minute (10 second period)"
] * 4 + [
    "Charge at 15W until 4.2V",
    "Hold at 4.2V until 0.01A"
]

experiment = pybamm.Experiment(cycles)
simulation = pybamm.Simulation(model, parameter_values=parameters, experiment=experiment)
simulation.solve()

sol = simulation.solution

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(sol.cycles[0].steps)))

for i, (step, color) in enumerate(zip(sol.cycles[0].steps, colors)):
    plt.plot(
        step["Time [h]"].data,
        step["Terminal voltage [V]"].data,
        label=f"Step {i+1}",
        color=color,
        marker='o',
        markersize=3,
        linewidth=1
    )

plt.xlabel("Time [h]", fontsize=12)
plt.ylabel("Terminal Voltage [V]", fontsize=12)
plt.title("Terminal Voltage over Steps in First Cycle", fontsize=14)
plt.legend(loc='best', fontsize=9)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

# but here we are going to get a discontinuous plot because the steps are not continuous, they are separated by rest periods and we are plotting eac step separately

# so we need to plot the steps in a continuous way

plt.figure(figsize=(10, 6))
for step in sol.cycles[0].steps:
    plt.plot(
        step["Time [h]"].data,
        step["Terminal voltage [V]"].data,
        marker='o',
        markersize=3,
        linewidth=1
    )
plt.xlabel("Time [h]", fontsize=12)
plt.ylabel("Terminal Voltage [V]", fontsize=12)
plt.title("Terminal Voltage over Steps in First Cycle (Continuous)", fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()


pulses = sol.cycles[0].steps[1]
for i in range(2, 9):
    pulses += sol.cycles[0].steps[i]
plt.plot(pulses["Time [h]"].data , pulses["Terminal voltage [V]"].data, marker='o', markersize=3, linewidth=1)
plt.xlabel("Time [h]", fontsize=12)
plt.ylabel("Terminal Voltage [V]", fontsize=12)
plt.title("Pulses in Terminal Voltage", fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

pulses.plot(["Terminal voltage [V]", "Current [A]"])
'''

cycles = [["Discharge at C/20 for 1 hour", "Rest for 1 hour"]] * 20
flat_cycles = [step for cycle in cycles for step in cycle]
experiment = pybamm.Experiment(list(flat_cycles))  # Explicitly cast to list[str]
simulation = pybamm.Simulation(model, parameter_values=parameters, experiment=experiment)
solution = simulation.solve()
simulation.plot(["Terminal voltage [V]", "Current [A]"])

# this above code will perform the experiment for 20 cycles, each cycle consisting of a discharge of C/20 for one hour followed by a rest of one hour
# so the above solution of the given experiment is a GITT solution (Galvanostatic Intermittent Titration Technique)


# plot the first step of the first cycle 
solution.cycles[0].steps[1].plot(["Terminal voltage [V]", "Current [A]"])

# now we are going to extract resistance from each step
sol = simulation.solution
def extract_resistance(step):
    return step["Local ECM resistance [Ohm]"].data[-1]

resistance = np.array([extract_resistance(step) for i, step in enumerate(solution.cycles[0].steps) if i % 2 == 0])
resistance_means = [np.mean(r) for r in resistance]

plt.plot(np.arange(len(resistance_means)), resistance_means, marker='o', markersize=3, linewidth=1, color='red')

plt.xlabel("Step index")
plt.ylabel("Mean EMS Resistance [Ohm]")
plt.title("EMS Resistance for Even Steps (Averaged)")
plt.grid(True)
plt.show()


# Model and parameter setup
model = pybamm.lithium_ion.SPMe()
parameters = pybamm.ParameterValues("Chen2020")

# Define the experiment: charge and hold
experiment = pybamm.Experiment([
    "Charge at 1C until 4.2V",
    "Hold at 4.2V until C/50"
])

# List to store solutions for different initial SOCs
sols = []

# Initial SOCs to simulate
init_socs = [0, 0.1, 0.2, 0.4, 0.6, 0.8]

# Loop through initial SOCs and run simulations
for initial_soc in init_socs:
    sim = pybamm.Simulation(model, parameter_values=parameters, experiment=experiment)
    sol = sim.solve(initial_soc=initial_soc)  # solve simulation for this initial SOC
    sols.append(sol)  # store the solution

# Plot the solutions with dynamic plotting
pybamm.dynamic_plot(sols, labels=[f"Initial SOC = {soc}" for soc in init_socs])

# Explanation:
# 1. We initialize an empty list `sols` to store the solutions.
# 2. We define a list of initial states of charge (SOCs).
# 3. For each initial SOC, we create a Simulation and solve it.
# 4. Each solution is appended to the `sols` list.
# 5. We use dynamic_plot to visualize the different scenarios.
#
# In your original code, the mistake was using `sols = sim.solve(...)`, which overwrote the `sols` list each time.
# Then trying to call `sols.append(sol)`, which failed because `sols` was now a Solution object, not a list.
# The fix: `sol = sim.solve(...)` and then `sols.append(sol)`. No overwriting of `sols`!


fig, ax = plt.subplots(1, 2)

for sol in sols:
    cc = sol.cycles[0].steps[0]
    cv = sol.cycles[0].steps[1]
    t_cc = cc["Time [s]"].data
    t_cv = cv["Time [s]"].data + t_cc[-1] # Adjust time for cv step to start after cc step
    ax[0].plot(t_cc-t_cv[0], cc["Terminal voltage [V]"].data)
    ax[0].set_title("CC step terminal voltage [V]")
    ax[1].plot(t_cv-t_cv[0], cv["Terminal voltage [V]"].data) 
    ax[1].set_title("CV Step terminal voltage [V]")
    ax[0].set_xlabel("Time [s]")
    ax[1].set_xlabel("Time [s]")
    ax[0].set_ylabel("Terminal voltage [V]")
    ax[1].set_ylabel("Terminal voltage [V]")
    ax[0].legend()
    ax[1].legend()
plt.tight_layout()
plt.show()
# The above code plots the terminal voltage for both the CC and CV steps of each simulation.
# It uses subplots to show both steps side by side for each initial SOC.
# The x-axis is adjusted so that the CV step starts at zero time after the CC step ends.
# The y-axis shows the terminal voltage in volts.



