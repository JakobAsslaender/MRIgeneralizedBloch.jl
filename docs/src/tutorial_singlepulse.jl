#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/tutorial_singlepulse.ipynb)

# # Simulation of a Single RF Pulse

# The core of generalized Bloch model is implemented in the function [`apply_hamiltonian_gbloch!(∂m∂t, m, mfun, p, t)`](@ref), which calculates the derivative `∂m/∂t` for a given magnetization vector `m` and stores it in-place in the the variable `∂m∂t`. The function interface is written in a format that can be fed directly into a differential equation solver of the [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) package.

# We need the following packages for this tutorial:
using MRIgeneralizedBloch
using DifferentialEquations
using SpecialFunctions
using QuadGK
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide #!nb

# and we define the properties a coupled spin system:
m0s = 0.15
R1f = 0.5 # 1/s
R2f = 13 # 1/s
R1s = 3 # 1/s
T2s = 12e-6 # s
Rx = 17; # 1/s

# Here, we simulate a 100μs inversion pulse:
α = π
TRF = 200e-6; # s

# Further, we assume a perfectly calibrated, on-resonant RF-pulse:
B1 = 1
ω0 = 0; # rad/s

# as well as a [super-Lorentzian lineshape](http://dx.doi.org/10.1002/mrm.1910330404). We interpolate the corresponding Green's function to improve performance:
G = interpolate_greens_function(greens_superlorentzian, 0, TRF / T2s);

# ## Rectangular RF-Pulses
# First, we simulate a rectangular RF-pulse where we can assume a constant `ω1` during the entire simulation:
ω1 = α/TRF;

# ### Isolated Semi-Solid Spin Pool
# The first example shows how to simulate an isolated semi-solid spin pool for which the magnetization vector is defined by `m = [zs; 1]`. The appended `1` facilitates a more compact implementation of longitudinal relaxation to a non-zero thermal equilibrium. Here, we initialize the magnetization with the thermal equilibrium:
m0 = [m0s; 1];

# The generalized Bloch model is a so-called integro-differential equation where the derivative ``∂m/∂t`` at the time ``t_1`` does not just depend on ``m(t_1)``, but on ``m(t)`` for ``t \in [0, t_1]``. This is solved with a [delay differential equation (DDE) solver](https://diffeq.sciml.ai/stable/tutorials/dde_example/) that stores an interpolated *history function* `mfun(p, t)`, which we use in the [`apply_hamiltonian_gbloch!`](@ref) function to evaluate the integral. This history function has to be initialized with
mfun(p, t) = m0;

# For slight performance improvements, we could also initialize the the history function with `mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? m0[idxs] : m0`. This syntax allows for direct indexing of the history function in [`apply_hamiltonian_gbloch!`](@ref), which improves performance. Following the syntax of the [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) package, we can define and solve the differential equation:

param = (ω1, B1, ω0, R1s, T2s, G) # defined by apply_hamiltonian_gbloch!
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
sol = solve(prob)

# The function [`apply_hamiltonian_gbloch!`](@ref) is implemented such that it concludes from `param = (ω1, B1, ω0, R1s, T2s, G)` that you are only supplying the relaxation properties of the semi-solid spin pool and hence it simulates the spin dynamics of an isolated semi-solid spin pool. The DifferentialEquations.jl package also implements a plot function for the solution objects
p = plot(sol, xlabel="t [s]", ylabel="zˢ(t)", idxs=1, labels=:none)
#md Main.HTMLPlot(p) #hide


# ### Coupled Spin System
# For a coupled spin system, the magnetization vector is defined as `m = [xf; yf; zf; zs; 1]` and the thermal equilibrium magnetization is given by:
m0 = [0; 0; 1-m0s; m0s; 1];

# To indicate to the `apply_hamiltonian_gbloch!` function that we would like to simulate a coupled spin system, we simple provide it with the properties of both pools in the following format:
param = (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G);

# Thereafter, we can use the same function calls as above to simulate the spin dynamics:
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
sol = solve(prob)
p = plot(sol, xlabel="t [s]", ylabel="m(t)", idxs=1:4, labels=["xᶠ" "yᶠ" "zᶠ" "zˢ"])
#md Main.HTMLPlot(p) #hide


# ## Shaped RF-Pulses
# The function `apply_hamiltonian_gbloch!` also allows for the simulation of RF-pulses with arbitrary shapes. In order to simulate shaped RF-pulses, ω₁ has to be defined as a function that takes time as an input and returns ω₁ at this particular point in time. For example, we can define a `sinc`-pulse:
NSideLobes = 1
f_ω1(t) = sinc(2(NSideLobes+1) * t/TRF - (NSideLobes+1)) * α / (sinint((NSideLobes+1)π) * TRF/π / (NSideLobes+1));

# `NSideLobes` defines here the number of side lobes on each side as can be seen in the following plot.
tᵢ = 0:1e-6:TRF
p = plot(tᵢ, f_ω1.(tᵢ), xlabel="t [s]", ylabel="ω₁(t)", labels=:none)
#md Main.HTMLPlot(p) #hide

# With numerical integration we can check if the RF-pulse has the correct flip angle:
quadgk(f_ω1, 0, TRF)[1] / α

# ### Isolated Semi-Solid Spin Pool
# In order to calculate the spin dynamics of an isolated semi-solid spin pool during a shaped RF-pulse, we use the same function call as we did in the Section [Rectangular RF-Pulses](@ref) with the only difference that the first element of `param` is of type `Function`:
m0 = [m0s; 1]
param = (f_ω1, B1, ω0, R1s, T2s, G)
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
sol = solve(prob)
p = plot(sol, xlabel="t [s]", ylabel="zˢ(t)", idxs=1, labels=:none)
#md Main.HTMLPlot(p) #hide


# ### Coupled Spin System
# And the same simulation can be done for a coupled spin system:
param = (f_ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G)
m0 = [0; 0; 1-m0s; m0s; 1];
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
sol = solve(prob)
p = plot(sol, xlabel="t [s]", ylabel="m(t)", idxs=1:4, labels=["xᶠ" "yᶠ" "zᶠ" "zˢ"])
#md Main.HTMLPlot(p) #hide

# More details on the interface, including the linear approximation of the generalized Bloch model can found in the following scripts that replicate all simulations, data analyses, and figures of the generalized Bloch paper.