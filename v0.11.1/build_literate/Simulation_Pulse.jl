using MRIgeneralizedBloch
using QuadGK
using DelayDiffEq
using DifferentialEquations
using SpecialFunctions
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide

R₁ = 1 # 1/s
T₂ˢ = 10e-6; # s

α = π
Tʳᶠ = exp.(range(log(2e-7), log(1e-1), length=100)) # s
ω₁ = α ./ Tʳᶠ # rad/s
ω₀ = 0; # rad/s

H(ω₁, ω₀, R₂, R₁) = [-R₂ -ω₀  ω₁  0;
                      ω₀ -R₂   0  0;
                     -ω₁   0 -R₁ R₁;
                       0   0   0  0]

z_Bloch = similar(Tʳᶠ)
for i ∈ eachindex(Tʳᶠ)
    (_, _, z_Bloch[i], _)  = exp(H(ω₁[i], ω₀, 1 / T₂ˢ, R₁) * Tʳᶠ[i]) * [0; 0; 1; 1]
end

Rʳᶠ = @. ω₁^2 * T₂ˢ * ((exp(-Tʳᶠ / T₂ˢ) -1) * T₂ˢ + Tʳᶠ) / Tʳᶠ
z_Graham_spec_Lorentzian = @. (Rʳᶠ * exp(-Tʳᶠ * (R₁ + Rʳᶠ)) + R₁) / (R₁ + Rʳᶠ);

g_Lorentzian(ω₀) = T₂ˢ / π ./ (1 .+ (T₂ˢ .* ω₀).^2)
Rʳᶠ = @. π * ω₁^2 * g_Lorentzian(ω₀)
z_Graham_SF_approx_Lorentzian = @. (Rʳᶠ * exp(-Tʳᶠ * (R₁ + Rʳᶠ)) + R₁) / (R₁ + Rʳᶠ);

z₀ = [1.0, 1.0] # initial z-magnetization
z_Sled_Lorentzian = similar(Tʳᶠ)
for i ∈ eachindex(Tʳᶠ)
    param = (ω₁[i], 1, ω₀, R₁, T₂ˢ, greens_lorentzian)
    prob = ODEProblem(apply_hamiltonian_sled!, z₀, (0, Tʳᶠ[i]), param)
    z_Sled_Lorentzian[i] = solve(prob).u[end][1]
end

z_fun(p, t) = [1.0, 1.0]; # initialize history function (will be populated with an interpolation by the DDE solver)

z_gBloch_Lorentzian = similar(Tʳᶠ)
for i ∈ eachindex(Tʳᶠ)
    param = (ω₁[i], 1, ω₀, R₁, T₂ˢ, greens_lorentzian)
    prob = DDEProblem(apply_hamiltonian_gbloch!, z₀, z_fun, (0, Tʳᶠ[i]), param)
    z_gBloch_Lorentzian[i] = solve(prob, MethodOfSteps(Tsit5()); dtmax=Tʳᶠ[i]/10).u[end][1]
end

p = plot(xaxis=:log, legend=:bottomright, xlabel="Tʳᶠ [s]", ylabel="zˢ(Tʳᶠ)")
plot!(p, Tʳᶠ, z_gBloch_Lorentzian, label="generalized Bloch model")
plot!(p, Tʳᶠ, Tʳᶠ .* 0 .+ cos(α), label="cos(α)")
plot!(p, Tʳᶠ, z_Sled_Lorentzian, label="Sled's model")
plot!(p, Tʳᶠ, z_Graham_spec_Lorentzian, label="Graham's spectral model")
plot!(p, Tʳᶠ, z_Graham_SF_approx_Lorentzian, label="Graham's single frequency approximation")
plot!(p, Tʳᶠ, z_Bloch, label="Bloch model")

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

Tʳᶠᵢ = 1e-3 # s
ω₁ᵢ = α / Tʳᶠᵢ # rad/s
param = (ω₁ᵢ, 1, ω₀, R₁, T₂ˢ, G_superLorentzian)

prob = ODEProblem(apply_hamiltonian_sled!, z₀, (0, Tʳᶠᵢ), param)
z_Sled_superLorentzian_i = solve(prob).u[end][1]

prob = DDEProblem(apply_hamiltonian_gbloch!, z₀, z_fun, (0, Tʳᶠᵢ), param)
z_gBloch_superLorentzian_i = solve(prob, MethodOfSteps(Tsit5())).u[end][1]

z_Sled_superLorentzian_i - z_gBloch_superLorentzian_i

Tʳᶠᵢ = 1e-4 # s
ω₁ᵢ = α / Tʳᶠᵢ # rad/s
param = (ω₁ᵢ, 1, ω₀, R₁, T₂ˢ, G_superLorentzian)

prob = ODEProblem(apply_hamiltonian_sled!, z₀, (0, Tʳᶠᵢ), param)
z_Sled_superLorentzian_i = solve(prob).u[end][1]

prob = DDEProblem(apply_hamiltonian_gbloch!, z₀, z_fun, (0, Tʳᶠᵢ), param)
z_gBloch_superLorentzian_i = solve(prob, MethodOfSteps(Tsit5())).u[end][1]

z_Sled_superLorentzian_i - z_gBloch_superLorentzian_i

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
