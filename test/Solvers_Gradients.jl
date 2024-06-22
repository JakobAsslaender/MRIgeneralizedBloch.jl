##
using MRIgeneralizedBloch
using DifferentialEquations
using StaticArrays
using Test
R2slT = precompute_R2sl()

## choose random parameters
Npulse = 500
α = abs.(π/2 * sin.(π/2 * ((0:Npulse-1) / Npulse) .+ π/2))
TRF = 300e-6 .+ 200e-6 * cos.(π * (1:Npulse) / Npulse)
α[1] = π
TRF[1] = 500e-6
ω1 = α ./ TRF
isInversionPulse = α .≈ π
TR = 3.5e-3

B1 = 1
ω0 = 0
m0s = 0.25
R1f = 0.3
R1s = 2
R2f = 1 / 65e-3
T2s = 10e-6
Rx = 20
R1a = 0.7

rtol = 1e-5

## m0s
Δm0s = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=(grad_m0s(),))
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s + Δm0s, R1f, R2f, Rx, R1s, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / Δm0s
@test g ≈ gfd rtol = rtol

## R1f
ΔR1f = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=(grad_R1f(),))
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f + ΔR1f, R2f, Rx, R1s, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / ΔR1f
@test g ≈ gfd rtol = rtol

## R2f
ΔR2f = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=(grad_R2f(),))
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f + ΔR2f, Rx, R1s, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / ΔR2f
@test g ≈ gfd rtol = rtol

## Rx
ΔRx = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=(grad_Rx(),))
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx + ΔRx, R1s, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / ΔRx
@test g ≈ gfd rtol = rtol

## R1s
ΔR1s = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=(grad_R1s(),))
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s + ΔR1s, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / ΔR1s
@test g ≈ gfd rtol = rtol

## T2s
ΔT2s = 1e-9
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=(grad_T2s(),))
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s + ΔT2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / ΔT2s
@test g ≈ gfd rtol = 5e-3 # not as precise because of the Higham's Complex Step Approximation

## ω0
Δω0 = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=(grad_ω0(),))
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0 + Δω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / Δω0
@test g ≈ gfd rtol = rtol

## B1
ΔB1 = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=(grad_B1(),))
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1 + ΔB1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / ΔB1
@test g ≈ gfd rtol = 5e-2 # not as precise because of the Higham's Complex Step Approximation

## R1a
ΔR1a = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1a, R2f, Rx, R1a, T2s, R2slT, grad_list=(grad_R1a(),))
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1a + ΔR1a, R2f, Rx, R1a + ΔR1a, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / ΔR1a
@test g ≈ gfd rtol = rtol