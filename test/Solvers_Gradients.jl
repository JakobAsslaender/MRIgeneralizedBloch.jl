##
using MRIgeneralizedBloch
using DifferentialEquations
using StaticArrays
using Test
R2slT = precompute_R2sl(100e-6, 1e-3, 5e-6, 15e-6, 0, π, 0.7, 1.3)

## choose random parameters
Npulse = 500
α = π/2 * rand() * sin.(8π * rand() * (1:Npulse) / Npulse)
α[1] = π
TRF = 100e-6 .+ 400e-6 * rand(Npulse)
TRF[1] = 500e-6

TR = 3.5e-3

B1 = 0.7 + 0.6 * rand()
ω0 = 1000 * rand()
m0s = 0.4 * rand()
R1f = 0.15 + 0.2 * rand()
R1s = 2 + rand()
R2f = 1 / (40e-3 + 100e-3 * rand())
T2s = 5e-6 + 10e-6 * rand()
Rx = 40 * rand()

R1a = 0.5 + rand()

rtol = 1e-5

## m0s
Δm0s = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=[grad_m0s()])
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s + Δm0s, R1f, R2f, Rx, R1s, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / Δm0s
@test g ≈ gfd rtol = rtol

## R1f
ΔR1f = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=[grad_R1f()])
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f + ΔR1f, R2f, Rx, R1s, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / ΔR1f
@test g ≈ gfd rtol = rtol

## R2f
ΔR2f = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=[grad_R2f()])
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f + ΔR2f, Rx, R1s, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / ΔR2f
@test g ≈ gfd rtol = rtol

## Rx
ΔRx = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=[grad_Rx()])
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx + ΔRx, R1s, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / ΔRx
@test g ≈ gfd rtol = rtol

## R1s
ΔR1s = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=[grad_R1s()])
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s + ΔR1s, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / ΔR1s
@test g ≈ gfd rtol = rtol

## T2s
ΔT2s = 1e-9
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=[grad_T2s()])
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s + ΔT2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / ΔT2s
@test g ≈ gfd rtol = 5e-3 # not as precise because of the Higham's Complex Step Approximation

## ω0
Δω0 = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=[grad_ω0()])
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0 + Δω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / Δω0
@test g ≈ gfd rtol = rtol

## B1
ΔB1 = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=[grad_B1()])
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1 + ΔB1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / ΔB1
@test g ≈ gfd rtol = 5e-3 # not as precise because of the Higham's Complex Step Approximation

## R1a
ΔR1a = 1e-6
s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1a, R2f, Rx, R1a, T2s, R2slT, grad_list=[grad_R1a()])
s1 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1a + ΔR1a, R2f, Rx, R1a + ΔR1a, T2s, R2slT)

g = s0[:,1,2]
gfd = vec(s1 .- s0[:,1,1]) / ΔR1a
@test g ≈ gfd rtol = rtol