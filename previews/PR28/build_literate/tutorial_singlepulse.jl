using MRIgeneralizedBloch
using DifferentialEquations
using SpecialFunctions
using QuadGK
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide

m0s = 0.15
R1f = 0.5 # 1/s
R2f = 13 # 1/s
R1s = 3 # 1/s
T2s = 12e-6 # s
Rx = 17; # 1/s

B1 = 1
ω0 = 0; # rad/s

G = interpolate_greens_function(greens_superlorentzian, 0, 1000);

α = π # rad
TRF = 500e-6 # s
ω1 = α/TRF # rad/s

m0 = [1; 1];

mfun(p, t) = m0;

param = (ω1, B1, ω0, R1s, T2s, G) # defined by apply_hamiltonian_gbloch!
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
z_gBloch = solve(prob)

p = plot(z_gBloch, xlabel="t [s]", ylabel="zˢ(t)", idxs=1, label="g. Bloch")

f_ω1(t) = ω1
Rʳᶠ = graham_saturation_rate_spectral(ω0 -> lineshape_superlorentzian(ω0, T2s), f_ω1, TRF, ω0) # 1/s

z_Graham(t) = (Rʳᶠ * exp(-t * (R1s + Rʳᶠ)) + R1s) / (R1s + Rʳᶠ)
plot!(p, z_Graham, 0, TRF, label="Graham")

z_gBloch(TRF)[1]

z_Graham(TRF)

m0 = [0; 0; 1-m0s; m0s; 1];

param = (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G);

prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
m_gBloch = solve(prob)
p = plot(m_gBloch, xlabel="t [s]", ylabel="m(t)", idxs=1:4, labels=["xᶠ" "yᶠ" "zᶠ" "zˢ"])

TRF = 1e-3 # s
NSideLobes = 1
f_ω1(t) = sinc(2(NSideLobes+1) * t/TRF - (NSideLobes+1)) * α / (sinint((NSideLobes+1)π) * TRF/π / (NSideLobes+1))
p = plot(f_ω1, 0, TRF, xlabel="t [s]", ylabel="ω₁(t)", labels=:none)

quadgk(f_ω1, 0, TRF)[1] / α

param = (f_ω1, B1, ω0, R1s, T2s, G)
typeof(f_ω1) <: Function

m0 = [1; 1]
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
z_gBloch = solve(prob)
p = plot(z_gBloch, xlabel="t [s]", ylabel="zˢ(t)", idxs=1, label="g. Bloch")

Rʳᶠ = graham_saturation_rate_spectral(ω0 -> lineshape_superlorentzian(ω0, T2s), f_ω1, TRF, ω0)

z_Graham(t) = (Rʳᶠ * exp(-t * (R1s + Rʳᶠ)) + R1s) / (R1s + Rʳᶠ)
plot!(p, z_Graham, 0, TRF, label="Graham")

z_gBloch(TRF)[1]

z_Graham(TRF)

param = (f_ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G)
m0 = [0; 0; 1-m0s; m0s; 1]
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
m_gBloch = solve(prob)
p = plot(m_gBloch, xlabel="t [s]", ylabel="m(t)", idxs=1:4, labels=["xᶠ" "yᶠ" "zᶠ" "zˢ"])

TRF = 10.24e-3 # s
γ = 267.522e6 # gyromagnetic ratio in rad/s/T
ω₁ᵐᵃˣ = 13e-6 * γ # rad/s
μ = 5 # shape parameter in rad
β = 674.1 # shape parameter in 1/s

f_ω1(t) = ω₁ᵐᵃˣ * sech(β * (t - TRF/2)) # rad/s
f_ω0(t) = -μ * β * tanh(β * (t - TRF/2)) # rad/s
f_φ(t)  = -μ * log(cosh(β * t) - sinh(β*t) * tanh(β*TRF/2)); # rad

p = plot(f_ω1, 0, TRF, xlabel="t [s]", ylabel="ω₁(t) [rad/s]", labels=:none)

p = plot(f_ω0, 0, TRF, xlabel="t [s]", ylabel="ω₀(t) [rad/s]", labels=:none)

p = plot(f_φ, 0, TRF, xlabel="t [s]", ylabel="φ(t) [rad]", labels=:none)

m0 = [0, 0, 1-m0s, m0s, 1]
p = (f_ω1, B1, f_φ, m0s, R1f, R2f, Rx, R1s, T2s, G)
m_gBloch = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), p))
p = plot(m_gBloch, xlabel="t [s]", ylabel="m(t)", idxs=1:4, labels=["xᶠ" "yᶠ" "zᶠ" "zˢ"])

Δω0 = 1000 # rad/s
f_φ_or(t) = f_φ(t) + Δω0 * t; # rad

B1 = 1.2 # 20% miss-calibration
p = (f_ω1, B1, f_φ_or, m0s, R1f, R2f, Rx, R1s, T2s, G)
m_gBloch = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), p))
p = plot(m_gBloch, xlabel="t [s]", ylabel="m(t)", idxs=1:4, labels=["xᶠ" "yᶠ" "zᶠ" "zˢ"])

ωx(t) = -B1 * f_ω1(t) * sin(f_φ_or(t))
ωy(t) =  B1 * f_ω1(t) * cos(f_φ_or(t))
zs_gBloch(t) = m_gBloch(t)[4]
xs_gBloch(t) = quadgk(τ -> G((t - τ) / T2s) * ωx(τ) * zs_gBloch(τ), 0, t)[1]
ys_gBloch(t) = quadgk(τ -> G((t - τ) / T2s) * ωy(τ) * zs_gBloch(τ), 0, t)[1];

p = plot(xs_gBloch, 0, TRF, xlabel="t [s]", ylabel="m(t)", label="xˢ")
plot!(p, ys_gBloch, 0, TRF, label="yˢ")
plot!(p, zs_gBloch, 0, TRF, label="zˢ")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

