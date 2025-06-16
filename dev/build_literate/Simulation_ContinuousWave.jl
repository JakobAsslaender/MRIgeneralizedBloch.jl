using MRIgeneralizedBloch
using DifferentialEquations
using QuadGK
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide

R₁ = 1.0 # 1/s
T₂ˢ = 10e-6 # s

Tʳᶠ = 2e-3 # s
ω₁ = 2000π # rad/s
ω₀ = 200π # rad/s

t = range(0, Tʳᶠ, length=1001) # time points for plotting
tspan = (0.0, Tʳᶠ); # simulation range

H(ω₁, ω₀, R₂, R₁) = [-R₂  -ω₀  ω₁  0;
                       ω₀ -R₂   0  0;
                      -ω₁   0 -R₁ R₁;
                        0   0   0  0]

z_Bloch = similar(t)
for i = 1:length(t)
    (_, _, z_Bloch[i], _) = exp(H(ω₁, ω₀, 1 / T₂ˢ, R₁) * t[i]) * [0; 0; 1; 1]
end

g_Lorentzian(ω₀) = T₂ˢ / π / (1 + (T₂ˢ * ω₀)^2)
z_steady_state_Lorentzian = R₁ / (R₁ + π * ω₁^2 * g_Lorentzian(ω₀))

Rʳᶠ = π * ω₁^2 * g_Lorentzian(ω₀)
z_Graham_Lorentzian = @. (Rʳᶠ * exp(-t * (R₁ + Rʳᶠ)) + R₁) / (R₁ + Rʳᶠ);

z₀ = [1.0, 1.0] # initial z-magnetization
param = (ω₁, 1, ω₀, R₁, T₂ˢ, greens_lorentzian) # defined by apply_hamiltonian_sled!
prob = ODEProblem(apply_hamiltonian_sled!, z₀, tspan, param)
z_Sled_Lorentzian = solve(prob);

zfun(p, t) = [1.0, 1.0] # initialize history function (will be populated with an interpolation by the DDE solver)

param = (ω₁, 1, ω₀, R₁, T₂ˢ, greens_lorentzian) # defined by apply_hamiltonian_gbloch!
prob = DDEProblem(apply_hamiltonian_gbloch!, z₀, zfun, tspan, param)
z_gBloch_Lorentzian = solve(prob);

p = plot(xlabel="t [ms]", ylabel="zˢ(t)")
plot!(p, 1e3t, zero(similar(t)) .+ z_steady_state_Lorentzian, label="Henkelman's steady-state")
plot!(p, 1e3t, z_Graham_Lorentzian, label="Graham's model")
plot!(p, 1e3t, hcat(z_Sled_Lorentzian(t).u...)[1,:], label="Sled's model")
plot!(p, 1e3t, hcat(z_gBloch_Lorentzian(t).u...)[1,:], label="generalized Bloch model")
plot!(p, 1e3t, z_Bloch, label="Bloch model")

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
plot!(p, 1e3t, hcat(z_Sled_Gaussian(t).u...)[1,:], label="Sled's model")
plot!(p, 1e3t, hcat(z_gBloch_Gaussian(t).u...)[1,:], label="generalized Bloch model")

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
plot!(p, 1e3t, hcat(z_Sled_superLorentzian(t).u...)[1,:], label="Sled's model")
plot!(p, 1e3t, hcat(z_gBloch_superLorentzian(t).u...)[1,:], label="generalized Bloch model")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
