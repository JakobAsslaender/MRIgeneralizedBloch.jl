using MRIgeneralizedBloch
using QuadGK
using DifferentialEquations
using SpecialFunctions
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); nothing #hide

α = π
TRF = exp.(range(log(2e-7), log(1e-1), length=100)) # s
ω1 = α ./ TRF # rad/s
ω0 = 0 # rad/s
R1 = 1 # 1/s
T2s = 10e-6 # s
z0 = [1] # initial z-magnetization
z_fun(p, t) = [1.0] # initialize history function (will be populated with an interpolation by the differential equation solver)
nothing #hide

H(ω1, ω0, R2, R1) = [-R2 -ω0  ω1  0;
                      ω0 -R2   0  0;
                     -ω1   0 -R1 R1;
                       0   0   0  0]

z_Bloch = similar(TRF)
for i = 1:length(TRF)
    (_, _, z_Bloch[i], _)  = exp(H(ω1[i], ω0, 1 / T2s, R1) * TRF[i]) * [0; 0; 1; 1]
end

Rrf = @. ω1^2 * T2s * ((exp(-TRF / T2s) -1) * T2s + TRF) / TRF
z_Graham_spec_Lorentzian = @. (Rrf * exp(-TRF * (R1 + Rrf)) + R1) / (R1 + Rrf)
nothing #hide

g_Lorentzian(ω0) = T2s / π ./ (1 .+ (T2s .* ω0).^2)
Rrf = @. π * ω1^2 * g_Lorentzian(ω0)
z_Graham_SF_approx_Lorentzian = @. (Rrf * exp(-TRF * (R1 + Rrf)) + R1) / (R1 + Rrf)
nothing #hide

z_Sled_Lorentzian = similar(TRF)
for i = 1:length(TRF)
    z_Sled_Lorentzian[i] = solve(ODEProblem(apply_hamiltonian_sled!, z0, (0, TRF[i]), (ω1[i], 1, ω0, R1, T2s, greens_lorentzian)), Tsit5())[end][1]
end
nothing #hide

z_gBloch_Lorentzian = similar(TRF)
for i = 1:length(TRF)
    z_gBloch_Lorentzian[i] = solve(DDEProblem(apply_hamiltonian_gbloch!, z0, z_fun, (0, TRF[i]), (ω1[i], 1, ω0, R1, T2s, greens_lorentzian)), MethodOfSteps(DP8()))[end][1]
end

p = plot(xaxis=:log, legend=:bottomright, xlabel="TRF [s]", ylabel="zs(TRF)")
plot!(p, TRF, z_gBloch_Lorentzian, label="generalized Bloch model")
plot!(p, TRF, TRF .* 0 .+ cos(α), label="cos(α)")
plot!(p, TRF, z_Sled_Lorentzian, label="Sled's model")
plot!(p, TRF, z_Graham_spec_Lorentzian, label="Graham's spectral model")
plot!(p, TRF, z_Graham_SF_approx_Lorentzian, label="Graham's single frequency approximation")
plot!(p, TRF, z_Bloch, label="Bloch simulation")

Rrf = @. ω1^2 * T2s * (2 * T2s * (exp(-(TRF/T2s)^2/2)-1) + sqrt(2π) * TRF * erf(TRF/T2s/sqrt(2))) / (2 * TRF)
z_Graham_spec_Gaussian = @. (Rrf * exp(-TRF * (R1 + Rrf)) + R1) / (R1 + Rrf)

g_Gaussian(ω0) = T2s / sqrt(2π) * exp(-(T2s * ω0)^2 / 2) # lineshape
Rrf = @. π * ω1^2 * g_Gaussian(ω0)
z_Graham_SF_approx_Gaussian = @. (Rrf * exp(-TRF * (R1 + Rrf)) + R1) / (R1 + Rrf)

z_Sled_Gaussian = similar(TRF)
for i = 1:length(TRF)
    z_Sled_Gaussian[i] = solve(ODEProblem(apply_hamiltonian_sled!, z0, (0, TRF[i]), (ω1[i], 1, ω0, R1, T2s, greens_gaussian)), Tsit5())[end][1]
end

z_gBloch_Gaussian = similar(TRF)
for i = 1:length(TRF)
    z_gBloch_Gaussian[i] = solve(DDEProblem(apply_hamiltonian_gbloch!, z0, z_fun, (0, TRF[i]), (ω1[i], 1, ω0, R1, T2s, greens_gaussian)), MethodOfSteps(DP8()))[end][1]
end

p = plot(xaxis=:log, legend=:bottomright, xlabel="TRF [s]", ylabel="zs(TRF)")
plot!(p, TRF, z_gBloch_Gaussian, label="generalized Bloch model")
plot!(p, TRF, TRF .* 0 .+ cos(α), label="cos(α)")
plot!(p, TRF, z_Sled_Gaussian, label="Sled's model")
plot!(p, TRF, z_Graham_spec_Gaussian, label="Graham's spectral model")
plot!(p, TRF, z_Graham_SF_approx_Gaussian, label="Graham's single frequency approximation")

G_superLorentzian = interpolate_greens_function(greens_superlorentzian, 0, maximum(TRF)/T2s)

f_PSD(τ) = quadgk(ct -> (4 / τ / abs(1 - 3 * ct^2) * (exp(- τ^2 / 8 * (1 - 3 * ct^2)^2) - 1) + sqrt(2π) * erf(τ / 2 / sqrt(2) * abs(1 - 3 * ct^2))) / abs(1 - 3 * ct^2), 0.0, 1.0)[1]
Rrf = @. f_PSD(TRF / T2s) * ω1^2 * T2s
z_Graham_spec_superLorentzian = @. (Rrf * exp(-TRF * (R1 + Rrf)) + R1) / (R1 + Rrf)

z_Sled_superLorentzian = similar(TRF)
for i = 1:length(TRF)
    z_Sled_superLorentzian[i] = solve(ODEProblem(apply_hamiltonian_sled!, z0, (0, TRF[i]), (ω1[i], 1, ω0, R1, T2s, G_superLorentzian)), Tsit5())[end][1]
end

z_gBloch_superLorentzian = similar(TRF)
for i = 1:length(TRF)
    z_gBloch_superLorentzian[i] = solve(DDEProblem(apply_hamiltonian_gbloch!, z0, z_fun, (0, TRF[i]), (ω1[i], 1, ω0, R1, T2s, G_superLorentzian)), MethodOfSteps(DP8()))[end][1]
end

p = plot(xaxis=:log, legend=:bottomright, xlabel="TRF [s]", ylabel="zs(TRF)")
plot!(p, TRF, z_gBloch_superLorentzian, label="generalized Bloch model")
plot!(p, TRF, TRF .* 0 .+ cos(α), label="cos(α)")
plot!(p, TRF, z_Sled_superLorentzian, label="Sled's model")
plot!(p, TRF, z_Graham_spec_superLorentzian, label="Graham's spectral model")

TRF_i = 1e-3 # s
ω1_i = α / TRF_i # rad/s
z_Sled_superLorentzian_i = solve(ODEProblem(apply_hamiltonian_sled!, z0, (0, TRF_i), (ω1_i, 1, ω0, R1, T2s, G_superLorentzian)), Tsit5())[end][1]
z_gBloch_superLorentzian_i = solve(DDEProblem(apply_hamiltonian_gbloch!, z0, z_fun, (0, TRF_i), (ω1_i, 1, ω0, R1, T2s, G_superLorentzian)), MethodOfSteps(DP8()))[end][1]
z_Sled_superLorentzian_i - z_gBloch_superLorentzian_i

TRF_i = 1e-4 # s
ω1_i = α / TRF_i # rad/s
z_Sled_superLorentzian_i = solve(ODEProblem(apply_hamiltonian_sled!, z0, (0, TRF_i), (ω1_i, 1, ω0, R1, T2s, G_superLorentzian)), Tsit5())[end][1]
z_gBloch_superLorentzian_i = solve(DDEProblem(apply_hamiltonian_gbloch!, z0, z_fun, (0, TRF_i), (ω1_i, 1, ω0, R1, T2s, G_superLorentzian)), MethodOfSteps(DP8()))[end][1]
z_Sled_superLorentzian_i - z_gBloch_superLorentzian_i

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

