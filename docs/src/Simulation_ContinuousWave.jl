#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/Simulation_ContinuousWave.ipynb)

# # Continuous Wave Simulation
# The following code replicates the continuous wave simulation of Fig. 2 and is slightly more comprehensive in the sense that all discussed models are simulated. 

# For these simulations we need the following packages:

using MRIgeneralizedBloch
using DifferentialEquations
using QuadGK
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide #!nb
#nb plotlyjs(ticks=:native);

# and we simulate an isolated semi-solid spin pool with the following parameters:
R₁ = 1.0 # 1/s
T₂ˢ = 10e-6 # s

version = "v2" #src
ω₁ = 2000π # rad/s
ω₀ = 200π # rad/s
Tʳᶠ = 2e-3 # s
#src version = "v1"
#src ω₁ = 1e2 * 2π # rad/s
#src ω₀ = 1e3 * 2π # rad/s
#src Tʳᶠ = 1 # s

t = range(0, Tʳᶠ, length=1001) # time points for plotting
tspan = (0.0, Tʳᶠ); # simulation range

# These parameters correspond to Fig. 2b, the parameters for replicating Fig. 2a are `ω₁ = 200π`, `ω₀ = 2000π`, and `Tʳᶠ = 1`. 

# ## Lorentzian Lineshape
# In this script, we simulate the three lineshapes separately, starting with the Lorentzian lineshape for which the Bloch model provides a ground truth. 

# ### Bloch Model
# We can formulate the [Bloch model](http://dx.doi.org/10.1103/PhysRev.70.460) as 
# ```math
# \partial_t \begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix} = \begin{pmatrix} 
# -R_2 & -ω_0 & ω_1 & 0 \\ 
# ω_0 & -R_2 & 0 & 0 \\  
# -ω_1 & 0 & -R_1 & R_1 \\ 
# 0 & 0 & 0 & 0
# \end{pmatrix} \begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix} ,
# ```
# where the matrix is the Hamiltonian of the Bloch model. For a constant ``ω_0`` and ``ω_1``, we can evaluate the Bloch model by taking the  matrix exponential of its Hamiltonian:

H(ω₁, ω₀, R₂, R₁) = [-R₂  -ω₀  ω₁  0; 
                       ω₀ -R₂   0  0;
                      -ω₁   0 -R₁ R₁;
                        0   0   0  0]

z_Bloch = similar(t)
for i = 1:length(t)
    (_, _, z_Bloch[i], _) = exp(H(ω₁, ω₀, 1 / T₂ˢ, R₁) * t[i]) * [0; 0; 1; 1]
end

# ### Henkelman's Steady-State Solution
# When assuming an isolated semi-solid pool, Eq. (9) in [*Henkelman, R. Mark, et al. "Quantitative interpretation of magnetization transfer." Magnetic resonance in medicine 29.6 (1993): 759-766*](https://doi.org/10.1002/mrm.1910290607) reduces to

g_Lorentzian(ω₀) = T₂ˢ / π / (1 + (T₂ˢ * ω₀)^2)
z_steady_state_Lorentzian = R₁ / (R₁ + π * ω₁^2 * g_Lorentzian(ω₀))

# where `g_Lorentzian(ω₀)` is the Lorentzian lineshape. 

# ### Graham's Single Frequency Approximation
# The lineshape is also used to calculate [Graham's single frequency approximation](http://doi.org/10.1002/jmri.1880070520), which describes an exponential decay with the RF-induced saturation rate `Rʳᶠ`:

Rʳᶠ = π * ω₁^2 * g_Lorentzian(ω₀)
z_Graham_Lorentzian = @. (Rʳᶠ * exp(-t * (R₁ + Rʳᶠ)) + R₁) / (R₁ + Rʳᶠ);

# ### Sled's Model
# [Sled's model](http://dx.doi.org/10.1006/jmre.2000.2059) is given by the ordinary differential equation (ODE)
# ```math
# \partial_t z(t) = \left(-\pi \int_0^t G(t-τ) \omega_1(τ)^2 dτ \right) z(t)  + R_1 (1-z),
# ```
# where ``G(t-τ)`` is the Green's function. The Hamiltonian of this ODE is implemented in [`apply_hamiltonian_sled!`](@ref) and we solve this ODE with the [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) package:

z₀ = [1.0] # initial z-magnetization
param = (ω₁, 1, ω₀, R₁, T₂ˢ, greens_lorentzian) # defined by apply_hamiltonian_sled!
prob = ODEProblem(apply_hamiltonian_sled!, z₀, tspan, param)
z_Sled_Lorentzian = solve(prob);


# ### Generalized Bloch Model
# The generalized Bloch model is given by the integro-differential equation (IDE)
# ```math
# \partial_t z(t) = - ω_1(t) \int_0^t G(t,τ) ω_1(τ) z(τ) dτ + R_1 (1 - z(t)) ,
# ```
# or by
# ```math
# \partial_t z(t) = - ω_y(t) \int_0^t G(t,τ) ω_y(τ) z(τ) dτ - ω_x(t) \int_0^t G(t,τ) ω_x(τ) z(τ) dτ + R_1 (1 - z(t)) ,
# ```

# for off-resonant RF-pulses with ``ω_1 = ω_x + i ω_y``. The Hamiltonian of the IDE is implemented in [`apply_hamiltonian_gbloch!`](@ref) and we can solve this IDE with the [delay-differential equation (DDE)](https://diffeq.sciml.ai/stable/tutorials/dde_example/) solver of the [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) package:

zfun(p, t) = [1.0] # initialize history function (will be populated with an interpolation by the DDE solver)

param = (ω₁, 1, ω₀, R₁, T₂ˢ, greens_lorentzian) # defined by apply_hamiltonian_gbloch!
prob = DDEProblem(apply_hamiltonian_gbloch!, z₀, zfun, tspan, param)
z_gBloch_Lorentzian = solve(prob);

# Now that we have solved all five models, we can plot the solutions for comparison:

p = plot(xlabel="t [ms]", ylabel="zˢ(t)")
plot!(p, 1e3t, zero(similar(t)) .+ z_steady_state_Lorentzian, label="Henkelman's steady-state")
plot!(p, 1e3t, z_Graham_Lorentzian, label="Graham's model")
plot!(p, 1e3t, (hcat(z_Sled_Lorentzian(t).u...)'), label="Sled's model")
plot!(p, 1e3t, (hcat(z_gBloch_Lorentzian(t).u...)'), label="generalized Bloch model")
plot!(p, 1e3t, z_Bloch, label="Bloch model")
#md Main.HTMLPlot(p) #hide

# Zooming into the plot, reveals virtually perfect (besides numerical differences) agreement between Bloch and generalized Bloch model and subtle, but existing differences when compared to the other models. Choosing a longer `T₂ˢ` amplifies these differences. 

# ## Gaussian Lineshape
# We can repeat these simulations (with the exception of the Bloch model) for the Gaussian lineshape:

g_Gaussian(ω₀) = T₂ˢ / sqrt(2π) * exp(-(T₂ˢ * ω₀)^2 / 2)
z_steady_state_Gaussian = R₁ / (R₁ + π * ω₁^2 * g_Gaussian(ω₀))

Rʳᶠ = π * ω₁^2 * g_Gaussian(ω₀)
z_Graham_Gaussian = @. (Rʳᶠ * exp(-t * (R₁ + Rʳᶠ)) + R₁) / (R₁ + Rʳᶠ)

param = (ω₁, 1, ω₀, R₁, T₂ˢ, greens_gaussian) # defined by apply_hamiltonian_sled!
prob = ODEProblem(apply_hamiltonian_sled!, z₀, tspan, param)
z_Sled_Gaussian = solve(prob)

prob = DDEProblem(apply_hamiltonian_gbloch!, z₀, zfun, tspan, param)
z_gBloch_Gaussian = solve(prob)

p = plot(xlabel="t [ms]", ylabel="zˢ(t)")
plot!(p, 1e3t, zero(similar(t)) .+ z_steady_state_Gaussian, label="Henkelman's steady-state")
plot!(p, 1e3t, z_Graham_Gaussian, label="Graham' model")
plot!(p, 1e3t, (hcat(z_Sled_Gaussian(t).u...)'), label="Sled's model")
plot!(p, 1e3t, (hcat(z_gBloch_Gaussian(t).u...)'), label="generalized Bloch model")
#md Main.HTMLPlot(p) #hide

# ## Super-Lorentzian Lineshape
# And we can repeat these simulations (with the exception of the Bloch model) for the [super-Lorentzian lineshape](http://dx.doi.org/10.1002/mrm.1910330404), which reveals the most pronounced deviations between the models due to the substantially slower decay of the Green's function:

g_superLorentzian(ω₀) = sqrt(2 / π) * T₂ˢ * quadgk(ct -> exp(-2 * (T₂ˢ * ω₀ / abs(3 * ct^2 - 1))^2) / abs(3 * ct^2 - 1), 0.0, sqrt(1 / 3), 1)[1]
z_steady_state_superLorentzian = R₁ / (R₁ + π * ω₁^2 * g_superLorentzian(ω₀))

Rʳᶠ = π * ω₁^2 * g_superLorentzian(ω₀)
z_Graham_superLorentzian = @. (Rʳᶠ * exp(-t * (R₁ + Rʳᶠ)) + R₁) / (R₁ + Rʳᶠ)

G_superLorentzian = interpolate_greens_function(greens_superlorentzian, 0, Tʳᶠ/T₂ˢ)

param = (ω₁, 1, ω₀, R₁, T₂ˢ, G_superLorentzian)
prob = ODEProblem(apply_hamiltonian_sled!, z₀, tspan, param)
z_Sled_superLorentzian = solve(prob)

prob = DDEProblem(apply_hamiltonian_gbloch!, z₀, zfun, tspan, param)
z_gBloch_superLorentzian = solve(prob)


p = plot(xlabel="t [ms]", ylabel="zˢ(t)")
plot!(p, 1e3t, zero(similar(t)) .+ z_steady_state_superLorentzian, label="Henkelman's steady-state")
plot!(p, 1e3t, z_Graham_superLorentzian, label="Graham's model")
plot!(p, 1e3t, (hcat(z_Sled_superLorentzian(t).u...)'), label="Sled's model")
plot!(p, 1e3t, (hcat(z_gBloch_superLorentzian(t).u...)'), label="generalized Bloch model")
#md Main.HTMLPlot(p) #hide


#src ##################################################################
#src # export data for plotting
#src ##################################################################
using Printf #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/CW_Henkelman_steady_state_", version, ".txt")), "w") #src
write(io, "t_s Lorentzian Gaussian superLorentzian \n") #src
write(io, @sprintf("%1.3e %1.3e %1.3e %1.3e \n", 0, z_steady_state_Lorentzian, z_steady_state_Gaussian, z_steady_state_superLorentzian)) #src
write(io, @sprintf("%1.3e %1.3e %1.3e %1.3e \n", t[end], z_steady_state_Lorentzian, z_steady_state_Gaussian, z_steady_state_superLorentzian)) #src
close(io) #src
    
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/CW_SpinDynamics_", version, ".txt")), "w") #src
write(io, "t_s t_ms z_Bloch z_gBloch_Lorentzian z_gBloch_Gaussian z_gBloch_superLorentzian z_Graham_Lorentzian z_Graham_Gaussian z_Graham_superLorentzian z_Sled_Lorentzian z_Sled_Gaussian z_Sled_superLorentzian \n") #src
for i = 1:length(t) #src
    write(io, @sprintf("%1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e \n", t[i], t[i]*1e3, z_Bloch[i], z_gBloch_Lorentzian(t[i])[1], z_gBloch_Gaussian(t[i])[1], z_gBloch_superLorentzian(t[i])[1], z_Graham_Lorentzian[i], z_Graham_Gaussian[i], z_Graham_superLorentzian[i], z_Sled_Lorentzian(t[i])[1], z_Sled_Gaussian(t[i])[1], z_Sled_superLorentzian(t[i])[1])) #src
end #src
close(io) #src