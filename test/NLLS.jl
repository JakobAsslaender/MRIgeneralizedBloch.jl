using Test
using MRIgeneralizedBloch
using LinearAlgebra
R2slT = precompute_R2sl()

## define control
# Npulse = 100
# α = abs.(π/2 * sin.(π/2 * ((0:Npulse-1) / Npulse .+ 0.5)))
# TRF = 300e-6 .+ 200e-6 * cos.(π * (1:Npulse) / Npulse)
# α[1] = π
# TRF[1] = 500e-6
using MAT
control = matread(normpath(joinpath(pathof(MRIgeneralizedBloch), "../../docs/control_MT_v3p2_TR3p5ms_discretized.mat")))
α   = control["alpha"]
TRF = control["TRF"]

TR = 3.5e-3

## test fit with separate R1f and R1s
m0s = 0.15
R1f = 0.5 # 1/s
R2f = 17 # 1/s
Rex = 30 # 1/s
R1s = 3 # 1/s
T2s = 12e-6 # s
ω0 = 100 # rad/s
B1 = 0.9 # in units of B1_nominal

s = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT)
qM = fit_gBloch(vec(s), α, TRF, TR; R2slT=R2slT)

@test qM.M0  ≈ 1   rtol = 1e-3
@test qM.m0s ≈ m0s rtol = 1e-3
@test qM.R1f ≈ R1f rtol = 1e-3
@test qM.R2f ≈ R2f rtol = 1e-3
@test qM.Rex  ≈ Rex  rtol = 1e-3
@test qM.R1s ≈ R1s rtol = 1e-3
@test qM.T2s ≈ T2s rtol = 1e-3
@test qM.ω0  ≈ ω0  rtol = 1e-3
@test qM.B1  ≈ B1  rtol = 1e-3

## compress with some random u
u = randn(ComplexF64, length(s), 100)
u,_,_ = svd(u)
sc = u' * vec(s)

qM = fit_gBloch(sc, α, TRF, TR; R2slT=R2slT, u=u)

@test qM.M0  ≈ 1   rtol = 1e-3
@test qM.m0s ≈ m0s rtol = 1e-3
@test qM.R1f ≈ R1f rtol = 1e-3
@test qM.R2f ≈ R2f rtol = 1e-3
@test qM.Rex  ≈ Rex  rtol = 1e-3
@test qM.R1s ≈ R1s rtol = 1e-3
@test qM.T2s ≈ T2s rtol = 1e-3
@test qM.ω0  ≈ ω0  rtol = 1e-3
@test qM.B1  ≈ B1  rtol = 1e-3

## test fit with apparent R1a
R1a = 1.0 # 1/s
s = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1a, R2f, Rex, R1a, T2s, R2slT)
qM = fit_gBloch(vec(s), α, TRF, TR; fit_apparentR1=true, R2slT=R2slT)

@test qM.R1f ≈ qM.R1s

@test qM.M0  ≈ 1   rtol = 1e-3
@test qM.m0s ≈ m0s rtol = 1e-3
@test qM.R1f ≈ R1a rtol = 1e-3
@test qM.R2f ≈ R2f rtol = 1e-3
@test qM.Rex  ≈ Rex  rtol = 1e-3
@test qM.T2s ≈ T2s rtol = 1e-3
@test qM.ω0  ≈ ω0  rtol = 1e-3
@test qM.B1  ≈ B1  rtol = 1e-3