using Test
using MRIgeneralizedBloch

## define control
Npulse = 100
α = abs.(π/2 * sin.(π/2 * ((0:Npulse-1) / Npulse .+ 0.5)))
TRF = 300e-6 .+ 200e-6 * cos.(π * (1:Npulse) / Npulse)
α[1] = π
TRF[1] = 500e-6
TR = 3.5e-3

## define biophysical parameters
m0s = 0.15
R1f = 0.5 # 1/s
R2f = 17 # 1/s
Rx = 30 # 1/s
R1s = 3 # 1/s
T2s = 12e-6 # s
ω0 = 100 # rad/s
B1 = 0.9 # in units of B1_nominal
R2slT = precompute_R2sl()

## simulate signal
s_linapp = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT)

## perform fits
qM = HSFP_fit(vec(s_linapp), α, TRF, TR; R2slT=R2slT)

## test
@test qM.M0  ≈ 1   rtol = 1e-3
@test qM.m0s ≈ m0s rtol = 1e-3
@test qM.R1f ≈ R1f rtol = 1e-3
@test qM.R2f ≈ R2f rtol = 1e-3
@test qM.Rx  ≈ Rx  rtol = 1e-3
@test qM.R1s ≈ R1s rtol = 1e-3
@test qM.T2s ≈ T2s rtol = 1e-3
@test qM.ω0  ≈ ω0  rtol = 1e-3
@test qM.B1  ≈ B1  rtol = 1e-3