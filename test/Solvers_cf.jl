##
using MRIgeneralizedBloch
using DifferentialEquations
using StaticArrays
using Test
R2slT = precompute_R2sl()

## choose random parameters
Npulse = 500
α = π/3 * sin.(4π * (1:Npulse) / Npulse)
α[1] = π
# TRF = similar(α)
# TRF .= 1e-3 # long TRF to make Graham and gBloch match better
TRF = 100e-6 .+ 400e-6 * rand(Npulse)
TRF[1] = 500e-6

TR = 3.5e-3

B1 = 0.9
ω0 = 300
m0s = 0.2
R1f = 0.35
R1s = 2.5
R2f = 1 / 65e-3
T2s = 11e-6
Rex = 20


## gBloch model with IDE vs Graham vs linear approximation: complex signal
s_gBloch_IDE    = calculatesignal_gbloch_ide(  α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s; Ncyc=4)
s_Graham        = calculatesignal_graham_ode(  α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s; Ncyc=4)
s_gBloch_linear = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT)
@test s_gBloch_IDE ≈ s_Graham        rtol = 5e-2
@test s_gBloch_IDE ≈ s_gBloch_linear rtol = 5e-2

## gBloch model with IDE vs Graham vs linear approximation: magnetixation
s_gBloch_IDE = calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s; Ncyc=4, output=:realmagnetization)
s_Graham     = calculatesignal_graham_ode(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s; Ncyc=4, output=:realmagnetization)
s_gBloch_linear = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT; output=:realmagnetization)
s_gBloch_linear_m = similar(s_gBloch_IDE)
for i ∈ eachindex(s_gBloch_linear)
    s_gBloch_linear_m[i,:] = s_gBloch_linear[i][[1;2;3;5;6]]
end
@test s_gBloch_IDE ≈ s_Graham          rtol = 5e-2
@test s_gBloch_IDE ≈ s_gBloch_linear_m rtol = 5e-2