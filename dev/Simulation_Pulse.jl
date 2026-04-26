#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/Simulation_Pulse.ipynb)

# # RF-Pulse Simulation
# The following code replicates the RF-pulse simulation of Fig. 3 and plots the ``z^s``-magnetization at the end of respective pulse.

# For these simulations we need the following packages:
using MRIgeneralizedBloch
using QuadGK
using DifferentialEquations
using SpecialFunctions
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide #!nb

# and we simulate an isolated semi-solid spin pool with the following parameters:
R₁ = 1 # 1/s
T₂ˢ = 10e-6; # s

# Here, we simulate π-pulses with the following parameters:
α = π
Tʳᶠ = exp.(range(log(2e-7), log(1e-1), length=100)) # s
ω₁ = α ./ Tʳᶠ # rad/s
ω₀ = 0; # rad/s
# Replace first line with `α = π/4` or `α = π/2` to simulate the other two rows of Fig. 3.

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
H(ω₁, ω₀, R₂, R₁) = [-R₂ -ω₀  ω₁  0;
                      ω₀ -R₂   0  0;
                     -ω₁   0 -R₁ R₁;
                       0   0   0  0]

z_Bloch = similar(Tʳᶠ)
for i ∈ eachindex(Tʳᶠ)
    (_, _, z_Bloch[i], _)  = exp(H(ω₁[i], ω₀, 1 / T₂ˢ, R₁) * Tʳᶠ[i]) * [0; 0; 1; 1]
end

# ### Graham's Spectral Model
# [Graham's spectral model](http://doi.org/10.1002/jmri.1880070520) is derived by integrating over the lineshape multiplied by the spectral response function of the RF-pulse. This results in the RF-induced saturation rate `Rʳᶠ` that is used in an exponential model:

Rʳᶠ = @. ω₁^2 * T₂ˢ * ((exp(-Tʳᶠ / T₂ˢ) -1) * T₂ˢ + Tʳᶠ) / Tʳᶠ
z_Graham_spec_Lorentzian = @. (Rʳᶠ * exp(-Tʳᶠ * (R₁ + Rʳᶠ)) + R₁) / (R₁ + Rʳᶠ);

# ### Graham's Single Frequency Approximation
# In the [single frequency approximation](http://doi.org/10.1002/jmri.1880070520), Graham assumes that the RF-pulse has only a single frequency, which reduces `Rʳᶠ` to

g_Lorentzian(ω₀) = T₂ˢ / π ./ (1 .+ (T₂ˢ .* ω₀).^2)
Rʳᶠ = @. π * ω₁^2 * g_Lorentzian(ω₀)
z_Graham_SF_approx_Lorentzian = @. (Rʳᶠ * exp(-Tʳᶠ * (R₁ + Rʳᶠ)) + R₁) / (R₁ + Rʳᶠ);

# where `g_Lorentzian(ω₀)` denotes the Lorentzian lineshape.

# ### Sled's Model
# [Sled's model](http://dx.doi.org/10.1006/jmre.2000.2059) is given by the ordinary differential equation (ODE)
# ```math
# \partial_t z(t) = \left(-\pi \int_0^t G(t-τ) \omega_1(τ)^2 dτ \right) z(t)  + R_1 (1-z),
# ```
# where ``G(t-τ)`` is the Green's function. The Hamiltonian of this ODE is implemented in [`apply_hamiltonian_sled!`](@ref) and the ODE can be solved with the [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) package:

z₀ = [1.0, 1.0] # initial z-magnetization
z_Sled_Lorentzian = similar(Tʳᶠ)
for i ∈ eachindex(Tʳᶠ)
    param = (ω₁[i], 1, ω₀, R₁, T₂ˢ, greens_lorentzian)
    prob = ODEProblem(apply_hamiltonian_sled!, z₀, (0, Tʳᶠ[i]), param)
    z_Sled_Lorentzian[i] = solve(prob).u[end][1]
end

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

z_fun(p, t) = [1.0, 1.0]; # initialize history function (will be populated with an interpolation by the DDE solver)

z_gBloch_Lorentzian = similar(Tʳᶠ)
for i ∈ eachindex(Tʳᶠ)
    param = (ω₁[i], 1, ω₀, R₁, T₂ˢ, greens_lorentzian)
    prob = DDEProblem(apply_hamiltonian_gbloch!, z₀, z_fun, (0, Tʳᶠ[i]), param)
    z_gBloch_Lorentzian[i] = solve(prob, MethodOfSteps(Tsit5()); dtmax=Tʳᶠ[i]/10).u[end][1]
end

# Now we have solved all five models and can plot the solutions for comparison:

p = plot(xaxis=:log, legend=:bottomright, xlabel="Tʳᶠ [s]", ylabel="zˢ(Tʳᶠ)")
plot!(p, Tʳᶠ, z_gBloch_Lorentzian, label="generalized Bloch model")
plot!(p, Tʳᶠ, Tʳᶠ .* 0 .+ cos(α), label="cos(α)")
plot!(p, Tʳᶠ, z_Sled_Lorentzian, label="Sled's model")
plot!(p, Tʳᶠ, z_Graham_spec_Lorentzian, label="Graham's spectral model")
plot!(p, Tʳᶠ, z_Graham_SF_approx_Lorentzian, label="Graham's single frequency approximation")
plot!(p, Tʳᶠ, z_Bloch, label="Bloch model")
#md Main.HTMLPlot(p) #hide

# ## Gaussian Lineshape
# We can repeat these simulations (with the exception of the Bloch model) for the Gaussian lineshape:

Rʳᶠ = @. ω₁^2 * T₂ˢ * (2 * T₂ˢ * (exp(-(Tʳᶠ/T₂ˢ)^2/2)-1) + sqrt(2π) * Tʳᶠ * erf(Tʳᶠ/T₂ˢ/sqrt(2))) / (2 * Tʳᶠ)
z_Graham_spec_Gaussian = @. (Rʳᶠ * exp(-Tʳᶠ * (R₁ + Rʳᶠ)) + R₁) / (R₁ + Rʳᶠ)

g_Gaussian(ω₀) = T₂ˢ / sqrt(2π) * exp(-(T₂ˢ * ω₀)^2 / 2) # lineshape
Rʳᶠ = @. π * ω₁^2 * g_Gaussian(ω₀)
z_Graham_SF_approx_Gaussian = @. (Rʳᶠ * exp(-Tʳᶠ * (R₁ + Rʳᶠ)) + R₁) / (R₁ + Rʳᶠ)

z_Sled_Gaussian = similar(Tʳᶠ)
for i ∈ eachindex(Tʳᶠ)
    param = (ω₁[i], 1, ω₀, R₁, T₂ˢ, greens_gaussian)
    prob = ODEProblem(apply_hamiltonian_sled!, z₀, (0, Tʳᶠ[i]), param)
    z_Sled_Gaussian[i] = solve(prob).u[end][1]
end

z_gBloch_Gaussian = similar(Tʳᶠ)
for i ∈ eachindex(Tʳᶠ)
    param = (ω₁[i], 1, ω₀, R₁, T₂ˢ, greens_gaussian)
    prob = DDEProblem(apply_hamiltonian_gbloch!, z₀, z_fun, (0, Tʳᶠ[i]), param)
    z_gBloch_Gaussian[i] = solve(prob, MethodOfSteps(Tsit5())).u[end][1]
end

p = plot(xaxis=:log, legend=:bottomright, xlabel="Tʳᶠ [s]", ylabel="zˢ(Tʳᶠ)")
plot!(p, Tʳᶠ, z_gBloch_Gaussian, label="generalized Bloch model")
plot!(p, Tʳᶠ, Tʳᶠ .* 0 .+ cos(α), label="cos(α)")
plot!(p, Tʳᶠ, z_Sled_Gaussian, label="Sled's model")
plot!(p, Tʳᶠ, z_Graham_spec_Gaussian, label="Graham's spectral model")
plot!(p, Tʳᶠ, z_Graham_SF_approx_Gaussian, label="Graham's single frequency approximation")
#md Main.HTMLPlot(p) #hide

# ## Super-Lorentzian Lineshape
# Further, we can repeat these simulations for the [super-Lorentzian lineshape](http://dx.doi.org/10.1002/mrm.1910330404) with the exception of Graham's single frequency approximation, as the super-Lorentzian lineshape diverges at ``ω_0 → 0``.

G_superLorentzian = interpolate_greens_function(greens_superlorentzian, 0, maximum(Tʳᶠ)/T₂ˢ)

f_PSD(τ) = quadgk(ct -> (4 / τ / abs(1 - 3 * ct^2) * (exp(- τ^2 / 8 * (1 - 3 * ct^2)^2) - 1) + sqrt(2π) * erf(τ / 2 / sqrt(2) * abs(1 - 3 * ct^2))) / abs(1 - 3 * ct^2), 0.0, 1.0)[1]
Rʳᶠ = @. f_PSD(Tʳᶠ / T₂ˢ) * ω₁^2 * T₂ˢ
z_Graham_spec_superLorentzian = @. (Rʳᶠ * exp(-Tʳᶠ * (R₁ + Rʳᶠ)) + R₁) / (R₁ + Rʳᶠ)

z_Sled_superLorentzian = similar(Tʳᶠ)
for i ∈ eachindex(Tʳᶠ)
    param = (ω₁[i], 1, ω₀, R₁, T₂ˢ, G_superLorentzian)
    prob = ODEProblem(apply_hamiltonian_sled!, z₀, (0, Tʳᶠ[i]), param)
    z_Sled_superLorentzian[i] = solve(prob).u[end][1]
end

z_gBloch_superLorentzian = similar(Tʳᶠ)
for i ∈ eachindex(Tʳᶠ)
    param = (ω₁[i], 1, ω₀, R₁, T₂ˢ, G_superLorentzian)
    prob = DDEProblem(apply_hamiltonian_gbloch!, z₀, z_fun, (0, Tʳᶠ[i]), param)
    z_gBloch_superLorentzian[i] = solve(prob, MethodOfSteps(Tsit5())).u[end][1]
end

p = plot(xaxis=:log, legend=:bottomright, xlabel="Tʳᶠ [s]", ylabel="zˢ(Tʳᶠ)")
plot!(p, Tʳᶠ, z_gBloch_superLorentzian, label="generalized Bloch model")
plot!(p, Tʳᶠ, Tʳᶠ .* 0 .+ cos(α), label="cos(α)")
plot!(p, Tʳᶠ, z_Sled_superLorentzian, label="Sled's model")
plot!(p, Tʳᶠ, z_Graham_spec_superLorentzian, label="Graham's spectral model")
#md Main.HTMLPlot(p) #hide

# This simulation reveals the most pronounced deviations of the generalized Bloch model from established models due to the slower decay of the super-Lorentzian Green's function.

# ### Error Analysis
# Assuming a super-Lorentzian lineshape, we quantify the deviations of Sled's model from the generalized Bloch model:

Tʳᶠᵢ = 1e-3 # s
ω₁ᵢ = α / Tʳᶠᵢ # rad/s
param = (ω₁ᵢ, 1, ω₀, R₁, T₂ˢ, G_superLorentzian)

prob = ODEProblem(apply_hamiltonian_sled!, z₀, (0, Tʳᶠᵢ), param)
z_Sled_superLorentzian_i = solve(prob).u[end][1]

prob = DDEProblem(apply_hamiltonian_gbloch!, z₀, z_fun, (0, Tʳᶠᵢ), param)
z_gBloch_superLorentzian_i = solve(prob, MethodOfSteps(Tsit5())).u[end][1]

z_Sled_superLorentzian_i - z_gBloch_superLorentzian_i

# For ``T_{\text{RF}} = 1``ms, the deviations are small compared to the thermal equilibrium magnetization ``z^s_0 = 1``, but with ``T_{\text{RF}} = 0.1``ms, this deviation becomes sizable:

Tʳᶠᵢ = 1e-4 # s
ω₁ᵢ = α / Tʳᶠᵢ # rad/s
param = (ω₁ᵢ, 1, ω₀, R₁, T₂ˢ, G_superLorentzian)

prob = ODEProblem(apply_hamiltonian_sled!, z₀, (0, Tʳᶠᵢ), param)
z_Sled_superLorentzian_i = solve(prob).u[end][1]

prob = DDEProblem(apply_hamiltonian_gbloch!, z₀, z_fun, (0, Tʳᶠᵢ), param)
z_gBloch_superLorentzian_i = solve(prob, MethodOfSteps(Tsit5())).u[end][1]

z_Sled_superLorentzian_i - z_gBloch_superLorentzian_i



#src ###############################################################
#src # export data for plotting
#src ###############################################################
using Printf #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/Pulse_Response_FA", round(Int, α / pi * 180), "deg.txt")), "w") #src
write(io, "TRF_s z_Bloch z_gBloch_Lorentzian z_gBloch_Gaussian z_gBloch_superLorentzian z_Sled_Lorentzian z_Sled_Gaussian z_Sled_superLorentzian z_Graham_spec_Lorentzian z_Graham_spec_Gaussian z_Graham_spec_superLorentzian z_Graham_SF_approx_Lorentzian z_Graham_SF_approx_Gaussian \n") #src
for i ∈ eachindex(Tʳᶠ) #src
    write(io, @sprintf("%1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e %1.3e \n", #src
    Tʳᶠ[i],  #src
    z_Bloch[i],  #src
    z_gBloch_Lorentzian[i],  #src
    z_gBloch_Gaussian[i],  #src
    z_gBloch_superLorentzian[i],  #src
    z_Sled_Lorentzian[i],  #src
    z_Sled_Gaussian[i],  #src
    z_Sled_superLorentzian[i],  #src
    z_Graham_spec_Lorentzian[i],  #src
    z_Graham_spec_Gaussian[i],  #src
    z_Graham_spec_superLorentzian[i],  #src
    z_Graham_SF_approx_Lorentzian[i],  #src
    z_Graham_SF_approx_Gaussian[i] #src
    )) #src
end #src
close(io) #src