##
using MRIgeneralizedBloch
using DifferentialEquations
using StaticArrays
using Test
using FiniteDifferences
R2slT = precompute_R2sl()

## choose random parameters
Npulse = 500
α = abs.(π/2 * sin.(π/2 * ((0:Npulse-1) / Npulse) .+ π/2))
TRF = 300e-6 .+ 200e-6 * cos.(π * (1:Npulse) / Npulse)
α[1] = π
TRF[1] = 500e-6
ω1 = α ./ TRF
grad_moment = [i == 1 ? :crusher : :balanced for i ∈ eachindex(α)]
TR = 3.5e-3

B1 = 1.0
ω0 = 0.0
m0s = 0.25
R1f = 0.3
R1s = 2.0
R2f = 1 / 65e-3
T2s = 10e-6
Rex = 20.0
R1a = 0.7

rtol = 1e-5

## M0
f_M0 = _M0 -> real.(calculatesignal_linearapprox(α, TRF, TR, ω0, B1, _M0, m0s, R1f, R2f, Rex, R1s, T2s, R2slT; grad_moment)[1])
gfd = jacobian(central_fdm(5,1; factor=1e6), f_M0, 1.0)[1]

_, grads = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list=(grad_M0(),); grad_moment)

g = grads[:,1]
@test g ≈ gfd rtol = rtol

## m0s
f_m0s = _m0s -> real.(calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, _m0s, R1f, R2f, Rex, R1s, T2s, R2slT; grad_moment)[1])
gfd = jacobian(central_fdm(5,1; factor=1e6), f_m0s, m0s)[1]

_, grads = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list=(grad_m0s(),); grad_moment)

g = grads[:,1]
@test g ≈ gfd rtol = rtol

## R1f
f_R1f = _R1f -> real.(calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, _R1f, R2f, Rex, R1s, T2s, R2slT; grad_moment)[1])
gfd = jacobian(central_fdm(5,1; factor=1e6), f_R1f, R1f)[1]

_, grads = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list=(grad_R1f(),); grad_moment)

g = grads[:,1]
@test g ≈ gfd rtol = rtol

## R2f
f_R2f = _R2f -> real.(calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1f, _R2f, Rex, R1s, T2s, R2slT; grad_moment)[1])
gfd = jacobian(central_fdm(5,1; factor=1e6), f_R2f, R2f)[1]

_, grads = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list=(grad_R2f(),); grad_moment)

g = grads[:,1]
@test g ≈ gfd rtol = rtol

## Rex
f_Rex = _Rex -> real.(calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, _Rex, R1s, T2s, R2slT; grad_moment)[1])
gfd = jacobian(central_fdm(5,1; factor=1e6), f_Rex, Rex)[1]

_, grads = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list=(grad_Rex(),); grad_moment)
g = grads[:,1]
@test g ≈ gfd rtol = rtol

## R1s
f_R1s = _R1s -> real.(calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, Rex, _R1s, T2s, R2slT; grad_moment)[1])
gfd = jacobian(central_fdm(5,1; factor=1e6), f_R1s, R1s)[1]

_, grads = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list=(grad_R1s(),); grad_moment)

g = grads[:,1]
@test g ≈ gfd rtol = rtol

## T2s
f_T2s = _T2s -> real.(calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, Rex, R1s, _T2s, R2slT; grad_moment)[1])
gfd = jacobian(central_fdm(5,1; factor=1e6,max_range=1e-8), f_T2s, T2s)[1]

_, grads = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list=(grad_T2s(),); grad_moment)

g = grads[:,1]
@test g ≈ gfd rtol = 1e-4 # not as precise because of the Higham's Complex Step Approximation

## ω0
f_ω0 = _ω0 -> calculatesignal_linearapprox(α, TRF, TR, _ω0, B1, 1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT; grad_moment)[1]
gfd = reinterpret(ComplexF64, jacobian(central_fdm(5,1; factor=1e6), f_ω0, ω0)[1])

_, grads = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list=(grad_ω0(),); grad_moment)

g = grads[:,1]
@test g ≈ gfd rtol = rtol

## B1
f_B1 = _B1 -> real.(calculatesignal_linearapprox(α, TRF, TR, ω0, _B1, 1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT; grad_moment)[1])
gfd = jacobian(central_fdm(5,1; factor=1e6), f_B1, B1)[1]

_, grads = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list=(grad_B1(),); grad_moment)

g = grads[:,1]
@test g ≈ gfd rtol = 1e-4 # not as precise because of the Higham's Complex Step Approximation

## R1a
R1a = 1.
f_R1a = _R1a -> real.(calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, _R1a, R2f, Rex, _R1a, T2s, R2slT; grad_moment)[1])
gfd = jacobian(central_fdm(5,1; factor=1e6), f_R1a, R1a)[1]

_, grads = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1a, R2f, Rex, R1a, T2s, R2slT, grad_list=(grad_R1a(),); grad_moment)

g = grads[:,1]
@test g ≈ gfd rtol = rtol