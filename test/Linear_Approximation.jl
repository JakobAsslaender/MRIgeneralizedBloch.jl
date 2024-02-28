using DifferentialEquations
using BenchmarkTools
using MRIgeneralizedBloch
using MRIgeneralizedBloch: apply_hamiltonian_gbloch!, hamiltonian_linear
using Test

## set variables
α = (.01:.01:1) * π
TRF = 500e-6 # s
ω1 = α ./ TRF # rad/s
m0s = 0.1
m0f = 1-m0s
R1f = 0.3 # 1/s
R1s = 2.0 # 1/s
R2f = 1 / 50e-3 # 1/s
T2s = 10e-6 # s
Rx = 70 # 1/s

## pre-calcualtions
G = interpolate_greens_function(greens_superlorentzian, 0, maximum(TRF) / T2s)
mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5)

print("Time to pre-compute saturation: ")
R2slT = @time precompute_R2sl()
R2sl = R2slT[1]

## init
u0_5D = [0,0,m0f,  m0s,1]
u0_6D = [0,0,m0f,0,m0s,1]

## simulate and plot magnetization at the end of RF-pulses with different flip angles
M_full = zeros(length(ω1), 4)
M_appx = similar(M_full)

for i in eachindex(ω1)
    M_full[i,:] = solve(DDEProblem(apply_hamiltonian_gbloch!, u0_5D, mfun, (0.0, TRF), (ω1[i], 1, 0, m0s, R1f, R2f, Rx, R1s, T2s, G)), MethodOfSteps(DP8())).u[end][1:4]
    u = exp(hamiltonian_linear(ω1[i], 1, 0, TRF, m0s, R1f, R2f, Rx, R1s, R2sl(TRF, α[i], 1, T2s))) * u0_6D
    M_appx[i,:] = u[[1:3;5]]
end

@test M_appx ≈ M_full rtol = 1e-4

## benchmark the different solvers (excecute one line at a time to provoke individual results to be printed)
print("Time to solve the full gene. Bloch IDE for 100us π-pulse:")
@btime solve(DDEProblem(apply_hamiltonian_gbloch!, u0_5D, mfun, (0.0, TRF), (ω1[end], 1, 0, m0s, R1f, R2f, Rx, R1s, T2s, G)), MethodOfSteps(DP8()))

print("Time to solve the linear approximation for 100us π-pulse:")
@btime exp(hamiltonian_linear(ω1[end-1], 1, 0, TRF, m0s, R1f, R2f, Rx, R1s, R2sl(TRF, α[end-1], 1, T2s))) * u0_6D

## ##########################################################################################################
# Test gradients
#############################################################################################################
R2sl, dR2sldT2s, dR2sldB1, dR2sldω1, dR2sldTRF, dR2sldT2sdω1, dR2sldB1dω1, dR2sldT2sTRF, dR2sldB1dTRF = R2slT

B1 = 1
α = π/2
TRF = 400e-6
ω1 = α / TRF

rtol = 1e-4

Δ = 1e-9
@test dR2sldT2s(TRF, ω1*TRF, B1, T2s) ≈ (R2sl(TRF, ω1*TRF, B1, T2s + Δ) - R2sl(TRF, ω1*TRF, B1, T2s)) / Δ rtol = rtol
@test dR2sldTRF(TRF, ω1*TRF, B1, T2s) ≈ (R2sl(TRF+Δ, ω1*(TRF+Δ), B1, T2s) - R2sl(TRF, ω1*TRF, B1, T2s)) / Δ rtol = rtol
@test dR2sldT2sTRF(TRF, ω1*TRF, B1, T2s) ≈ (dR2sldT2s(TRF+Δ, ω1*(TRF+Δ), B1, T2s) - dR2sldT2s(TRF, ω1*TRF, B1, T2s)) / Δ rtol = rtol
@test dR2sldB1dTRF(TRF, ω1*TRF, B1, T2s) ≈ (dR2sldB1(TRF+Δ, ω1*(TRF+Δ), B1, T2s) - dR2sldB1(TRF, ω1*TRF, B1, T2s)) / Δ rtol = rtol

Δ = 1e-6
@test dR2sldB1(TRF, ω1*TRF, B1, T2s) ≈ (R2sl(TRF, ω1*TRF, B1 + Δ, T2s) - R2sl(TRF, ω1*TRF, B1, T2s)) / Δ rtol = rtol
@test dR2sldω1(TRF, ω1*TRF, B1, T2s) ≈ (R2sl(TRF, (ω1+Δ)*TRF, B1, T2s) - R2sl(TRF, ω1*TRF, B1, T2s)) / Δ rtol = rtol

Δ = 1e-2
@test dR2sldT2sdω1(TRF, ω1*TRF, B1, T2s) ≈ (dR2sldT2s(TRF, (ω1+Δ)*TRF, B1, T2s) - dR2sldT2s(TRF, ω1*TRF, B1, T2s)) / Δ rtol = rtol
@test dR2sldB1dω1(TRF, ω1*TRF, B1, T2s) ≈ (dR2sldB1(TRF, (ω1+Δ)*TRF, B1, T2s) - dR2sldB1(TRF, ω1*TRF, B1, T2s)) / Δ rtol = rtol