import numpy 
import pandas as pd
import pybamm
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.integrate
import scipy.interpolate


# watch the youtube video for get an idea of some advanced examples in PyBaMM
# study why tesla cells are different from most of the competitors cells and why are they so much better

# now we will se how to load equations in a latex like format using PyBaMM

eqn = 4 + pybamm.t * 5
eqn.visualise("tree.png")
eqn.to_equation("latex.txt")
eqn.visualise("tree.pdf")
eqn.to_equation()
eqn.to_equation("latex")


model = pybamm.lithium_ion.SPM()
model.latexify()
model.latexify(newline=False)[2]  # this gives the output for all the equations in a list format with out a new line and 
                                    #that outside 2 written gives the 2md equation of the model from the list

str(model.latexify(newline=False)[2])   # adding the the whole above line of code inside a string gives the latex output for the 2nd equation of the model                                   
model.visualise("tree.png")
model.to_equation("latex.txt")


spme = pybamm.lithium_ion.SPMe()

spme.latexify("spme.png") # this will save the latex equations in a png file with all the equations and model parameters


spme.latexify(newline=True)[2]  # this gives the output for all the equations in a list format with a new line and 
                                    #that outside 2 written gives the 2md equation of the model from the list


