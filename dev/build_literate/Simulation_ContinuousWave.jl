using MRIgeneralizedBloch
using DifferentialEquations
using QuadGK
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native);

R1 = 1.0 # 1/s
T2s = 10e-6 # s
z0 = [1.0] # initial z-magnetization
z_fun(p, t) = [1.0] # initialize history function (will be populated with an interpolation by the differential equation solver)

ω1 = 2000π # rad/s
ω0 = 200π # rad/s
TRF = .002 # s


t = range(0, TRF, length=1001) # plot points
tspan = (0.0, TRF); # simulation range

H(ω1, ω0, R2, R1) = [-R2  -ω0  ω1  0;
                       ω0 -R2   0  0;
                      -ω1   0 -R1 R1;
                        0   0   0  0]

z_Bloch = similar(t)
for i = 1:length(t)
    (_, _, z_Bloch[i], _) = exp(H(ω1, ω0, 1 / T2s, R1) * t[i]) * [0; 0; 1; 1]
end

g_Lorentzian(ω0) = T2s / π / (1 + (T2s * ω0)^2)
z_steady_state_Lorentzian = R1 / (R1 + π * ω1^2 * g_Lorentzian(ω0))

Rrf = π * ω1^2 * g_Lorentzian(ω0)
z_Graham_Lorentzian = @. (Rrf * exp(-t * (R1 + Rrf)) + R1) / (R1 + Rrf);

z_Sled_Lorentzian = solve(ODEProblem(apply_hamiltonian_sled!, z0, tspan, (ω1, 1, ω0, R1, T2s, greens_lorentzian)));

z_gBloch_Lorentzian = solve(DDEProblem(apply_hamiltonian_gbloch!, z0, z_fun, tspan, (ω1, 1, ω0, R1, T2s, greens_lorentzian)));

p = plot(xlabel="t [ms]", ylabel="zs(t)")
plot!(p, 1e3t, z_Bloch, label="Bloch model")
plot!(p, 1e3t, zero(similar(t)) .+ z_steady_state_Lorentzian, label="Henkelman's steady-state")
plot!(p, 1e3t, z_Graham_Lorentzian, label="Graham's model")
plot!(p, 1e3t, (hcat(z_Sled_Lorentzian(t).u...)'), label="Sled's model")
plot!(p, 1e3t, (hcat(z_gBloch_Lorentzian(t).u...)'), label="generalized Bloch model")

g_Gaussian(ω0) = T2s / sqrt(2π) * exp(-(T2s * ω0)^2 / 2)
z_steady_state_Gaussian = R1 / (R1 + π * ω1^2 * g_Gaussian(ω0))

Rrf = π * ω1^2 * g_Gaussian(ω0)
z_Graham_Gaussian = @. (Rrf * exp(-t * (R1 + Rrf)) + R1) / (R1 + Rrf)

z_gBloch_Gaussian = solve(DDEProblem(apply_hamiltonian_gbloch!, z0, z_fun, tspan, (ω1, 1, ω0, R1, T2s, greens_gaussian)))

z_Sled_Gaussian = solve(ODEProblem(apply_hamiltonian_sled!, z0, tspan, (ω1, 1, ω0, R1, T2s, greens_gaussian)))

p = plot(xlabel="t [ms]", ylabel="zs(t)")
plot!(p, 1e3t, zero(similar(t)) .+ z_steady_state_Gaussian, label="Henkelman's steady-state")
plot!(p, 1e3t, z_Graham_Gaussian, label="Graham' model")
plot!(p, 1e3t, (hcat(z_Sled_Gaussian(t).u...)'), label="Sled's model")
plot!(p, 1e3t, (hcat(z_gBloch_Gaussian(t).u...)'), label="generalized Bloch model")

g_superLorentzian(ω0) = sqrt(2 / π) * T2s * quadgk(ct -> exp(-2 * (T2s * ω0 / abs(3 * ct^2 - 1))^2) / abs(3 * ct^2 - 1), 0.0, sqrt(1 / 3), 1)[1]
z_steady_state_superLorentzian = R1 / (R1 + π * ω1^2 * g_superLorentzian(ω0))

Rrf = π * ω1^2 * g_superLorentzian(ω0)
z_Graham_superLorentzian = @. (Rrf * exp(-t * (R1 + Rrf)) + R1) / (R1 + Rrf)

G_superLorentzian = interpolate_greens_function(greens_superlorentzian, 0, TRF/T2s)
z_gBloch_superLorentzian = solve(DDEProblem(apply_hamiltonian_gbloch!, z0, z_fun, tspan, (ω1, 1, ω0, R1, T2s, G_superLorentzian)))

z_Sled_superLorentzian = solve(ODEProblem(apply_hamiltonian_sled!, z0, tspan, (ω1, 1, ω0, R1, T2s, G_superLorentzian)))

p = plot(xlabel="t [ms]", ylabel="zs(t)")
plot!(p, 1e3t, zero(similar(t)) .+ z_steady_state_superLorentzian, label="Henkelman's steady-state (super-Lorentzian)")
plot!(p, 1e3t, z_Graham_superLorentzian, label="Graham super-Lorentzian")
plot!(p, 1e3t, (hcat(z_gBloch_superLorentzian(t).u...)'), label="gBloch super-Lorentzian")
plot!(p, 1e3t, (hcat(z_Sled_superLorentzian(t).u...)'), label="Sled super-Lorentzian")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

