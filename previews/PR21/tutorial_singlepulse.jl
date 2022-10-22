#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/tutorial_singlepulse.ipynb)

# # Simulation of a Single RF Pulse

# The core of generalized Bloch model is implemented in the function [`apply_hamiltonian_gbloch!(∂m∂t, m, mfun, p, t)`](@ref), which calculates the derivative `∂m/∂t` for a given magnetization vector `m` and stores it in-place in the variable `∂m∂t`. The function interface is written in a format that can be fed directly into a differential equation solver of the [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) package.

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

# For most parts of this tutorial, we assume a perfectly calibrated, on-resonant RF-pulse:
B1 = 1
ω0 = 0; # rad/s

# as well as a [super-Lorentzian lineshape](http://dx.doi.org/10.1002/mrm.1910330404). We interpolate the corresponding Green's function in the range `TRF ∈ [0, 1000 ⋅ T2s]` to improve performance:
G = interpolate_greens_function(greens_superlorentzian, 0, 1000);

# ## Rectangular RF-Pulses
# First, we simulate a rectangular RF-pulse with a constant `ω1`:
α = π # rad
TRF = 200e-6 # s
ω1 = α/TRF; # rad/s

# ### Isolated Semi-Solid Spin Pool
# The first example demonstrates how to simulate an isolated semi-solid spin pool for which the magnetization vector is defined by `m = [zs; 1]`. The appended `1` facilitates a more compact implementation of longitudinal relaxation to a non-zero thermal equilibrium. Here, we initialize the magnetization with the thermal equilibrium:
m0 = [m0s; 1];

# The generalized Bloch model is a so-called integro-differential equation where the derivative ``∂m/∂t`` at the time ``t_1`` does not just depend on ``m(t_1)``, but on ``m(t)`` with ``t \in [0, t_1]``. This is solved with a [delay differential equation (DDE) solver](https://diffeq.sciml.ai/stable/tutorials/dde_example/) that stores an interpolated *history function* `mfun(p, t)`, which we use in the [`apply_hamiltonian_gbloch!`](@ref) function to evaluate the integral. This history function has to be initialized with
mfun(p, t) = m0;

# For slight performance improvements, we could also initialize the history function with `mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? m0[idxs] : m0`. This syntax allows for direct indexing of the history function in [`apply_hamiltonian_gbloch!`](@ref), which improves performance. Following the syntax of the [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) package, we can define and solve the differential equation:

param = (ω1, B1, ω0, R1s, T2s, G) # defined by apply_hamiltonian_gbloch!
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
sol = solve(prob)

# The function [`apply_hamiltonian_gbloch!`](@ref) is implemented such that it concludes from `param = (ω1, B1, ω0, R1s, T2s, G)` that you are only supplying the relaxation properties of the semi-solid spin pool and hence it simulates the spin dynamics of an isolated semi-solid spin pool. The DifferentialEquations.jl package also implements a plot function for the solution object, which can use to display the result:
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
# The function `apply_hamiltonian_gbloch!` also allows for the simulation of RF-pulses with arbitrary shapes. To this end, ω₁(t) has to be defined as a function that takes time in seconds as an input and returns ω₁ at this particular point in time. For example, we can define a `sinc`-pulse:
NSideLobes = 1
f_ω1(t) = sinc(2(NSideLobes+1) * t/TRF - (NSideLobes+1)) * α / (sinint((NSideLobes+1)π) * TRF/π / (NSideLobes+1))
p = plot(f_ω1, 0, TRF, xlabel="t [s]", ylabel="ω₁(t)", labels=:none)
#md Main.HTMLPlot(p) #hide

# `NSideLobes` defines here the number of side lobes on each side as can be seen in the plot. With numerical integration we can check if the RF-pulse has the correct flip angle:
quadgk(f_ω1, 0, TRF)[1] / α

# ### Isolated Semi-Solid Spin Pool
# In order to calculate the spin dynamics of an isolated semi-solid spin pool during a shaped RF-pulse, we define the same tuple `param` as we did in Section [Rectangular RF-Pulses](@ref) with the only difference that the first element is a subtype of the abstract type `Function`:
m0 = [m0s; 1]
param = (f_ω1, B1, ω0, R1s, T2s, G)
typeof(f_ω1) <: Function

# With this definition of `param`, we can use the same function call as we did before:
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
sol = solve(prob)
p = plot(sol, xlabel="t [s]", ylabel="zˢ(t)", idxs=1, labels=:none)
#md Main.HTMLPlot(p) #hide


# ### Coupled Spin System
# We can perform the same change to `param` to simulate a coupled spin system:
param = (f_ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G)
m0 = [0; 0; 1-m0s; m0s; 1]
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
sol = solve(prob)
p = plot(sol, xlabel="t [s]", ylabel="m(t)", idxs=1:4, labels=["xᶠ" "yᶠ" "zᶠ" "zˢ"])
#md Main.HTMLPlot(p) #hide

# Double click on zˢ in the legend to isolate the semi-solid spin pool in the plot and compare the simulation to the last section.

# ## ω₀-Sweep or Adiabatic RF-Pulses
# The [`apply_hamiltonian_gbloch!(∂m∂t, m, mfun, p, t)`](@ref) function is also implemented for RF-pulses with a varying RF frequency ω₀(t) as, e.g., used in adiabatic pulses. In order to simulate such pulses, the first element of `param` has to be `ω₁(t)::Function` like for [Shaped RF-Pulses](@ref) and, additionally, the third element has to be `φ::Function` instead of `ω₀::Number`. Notice two differences here: first, the (abstract) type `Function` instead of `Number` will tell the compiler to use the adiabatic-pulse implementation. Second, this implementation requires the phase of the RF-pulse as function of time φ(t) instead of the frequency because φ(t) ≠ ω₀(t) ⋅ t, if ω₀ is a function of time.

# To show the interface at a practical example, we can defined a [hyperbolic secant adiabatic inversion pulse](https://doi.org/10.1006/jmre.1998.1441):
TRF = 10.24e-3 # s
γ = 267.522e6 # gyromagnetic ratio in rad/s/T
ω₁ᵐᵃˣ = 13e-6 * γ # rad/s
μ = 5 # shape parameter in rad
β = 674.1 # shape parameter in 1/s

f_ω1(t) = ω₁ᵐᵃˣ * sech(β * (t - TRF/2)) # rad/s
f_ω0(t) = -μ * β * tanh(β * (t - TRF/2)) # rad/s
f_φ(t)  = -μ * log(cosh(β * t) - sinh(β*t) * tanh(β*TRF/2)); # rad

# This pulse a hyperbolic secant amplitude:
p = plot(f_ω1, 0, TRF, xlabel="t [s]", ylabel="ω₁(t) [rad/s]", labels=:none)
#md Main.HTMLPlot(p) #hide

# and hyperbolic tangent frequency sweep
p = plot(f_ω0, 0, TRF, xlabel="t [s]", ylabel="ω₀(t) [rad/s]", labels=:none)
#md Main.HTMLPlot(p) #hide

# As explained above, we actually don't use the frequency in the implementation. Instead, we use the RF-phase ``φ(t) = \int_0^t ω₀(t) dt``:
p = plot(f_φ, 0, TRF, xlabel="t [s]", ylabel="φ(t) [rad]", labels=:none)
#md Main.HTMLPlot(p) #hide

# This interface, of course, also allows for the simulation of an isolated semi-solid spin pool with above described modifications to `param`. For brevity, however, we here directly simulate a coupled spin pool starting from thermal equilibrium:
m0 = [0, 0, 1-m0s, m0s, 1]
p = (f_ω1, B1, f_φ, m0s, R1f, R2f, Rx, R1s, T2s, G)
sol = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0.0, TRF), p))
p = plot(sol, xlabel="t [s]", ylabel="m(t)", idxs=1:4, labels=["xᶠ" "yᶠ" "zᶠ" "zˢ"])
#md Main.HTMLPlot(p) #hide

# This simulation shows the intended inversion of the free spin pool and a saturation of `m0s` to roughly 7% of its thermal equilibrium magnetization (double click on the corresponding legend entry). The transversal magnetization of the free pool exhibits some oscillations and at this point I should highlight a distinct difference between the implementation for adiabatic RF-pulses the above described implementation for constant ω₀: the latter case uses a frame of references that rotates with the RF-frequency about the z-axis, i.e. the RF-pulses rotate the magnetization  with ω₁ about the y-axis and, additionally, the magnetizations rotates with ω₀ about the z-axis. The implementation of the adiabatic pulses uses a rotating frame of reference that is on resonance with the Larmor frequency of the spin isochromat and angle of the RF-pulse changes with ω₀(t). To simulate off-resonance, we can simply add a static value to above function or, more precisely, add a phase slope to φ:
Δω0 = 1000 # rad/s
f_φ_or(t) = f_φ(t) + Δω0 * t; # rad

# We can, additionally, change `B1` to demonstrate the robustness of adiabatic pulses:
B1 = 1.2 # 20% miss-calibration
p = (f_ω1, B1, f_φ_or, m0s, R1f, R2f, Rx, R1s, T2s, G)
sol = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0.0, TRF), p))
p = plot(sol, xlabel="t [s]", ylabel="m(t)", idxs=1:4, labels=["xᶠ" "yᶠ" "zᶠ" "zˢ"])
#md Main.HTMLPlot(p) #hide

# While the spin dynamics during the pulse is changed, the final magnetization of the free pool is approximately the same compared for the on-resonant isochromat with `B1 = 1`. The final magnetization of the semi-solid spin pool is, like before, close to zero, but a close look reveals a small negative zˢ-magnetization (double click on zˢ in the plot's legend).

# ## Transversal magnetization of the semi-solid pool
# Throughout this tutorial, we only ever calculated and plotted the longitudinal magnetization of the semi-solid spin pool. This is foremost a result of way we formulate and solve the generalized Bloch equations (cf. Eq. (9) in the [paper](https://doi.org/10.1002/mrm.29071)). But this implementation is also reflective of the standard use-case in magnetization transfer, where we are foremost interested in the longitudinal magnetization of the semi-solid spin pool and its effect on the free spin pool. If required, it is, however, easily possible to calculate the transversal magnetization with Eqs. (4-5) from the paper:
ωx(t) = -B1 * f_ω1(t) * sin(f_φ_or(t))
ωy(t) =  B1 * f_ω1(t) * cos(f_φ_or(t))
zs_gBloch(t) = sol(t)[4]
xs_gBloch(t) = quadgk(τ -> G((t - τ) / T2s) * ωx(τ) * zs_gBloch(τ), 0, t)[1]
ys_gBloch(t) = quadgk(τ -> G((t - τ) / T2s) * ωy(τ) * zs_gBloch(τ), 0, t)[1];

# The last two lines calculate the numerical integral of the Green's function multiplied by the oscillating RF-fields. Similar [code](https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl/blob/master/src/DiffEq_Hamiltonians.jl) is also used in the implementation of `apply_hamiltonian_gbloch!`. Plotting these functions reveals the spin dynamics of the semi-solid spin pool:
p = plot(xs_gBloch, 0, TRF, xlabel="t [s]", ylabel="m(t)", label="xˢ")
plot!(p, ys_gBloch, 0, TRF, label="yˢ")
plot!(p, zs_gBloch, 0, TRF, label="zˢ")
#md Main.HTMLPlot(p) #hide


# More details on the interface, including the linear approximation of the generalized Bloch model can found in the following scripts that replicate all simulations, data analyses, and figures of the generalized Bloch paper.