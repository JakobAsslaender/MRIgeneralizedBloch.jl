##
using MRIgeneralizedBloch
using DifferentialEquations
using SpecialFunctions
using QuadGK
using Test

## Pulse parameters
α = 0.8π
TRF = 100e-6

# define other parameters
B1 = 0.87
m0s = 0.2
m0f = 1 - m0s
R1f = 0.5
R1s = 3
R2f = 13
T2s = 12e-6
R2s = 1 / T2s
Rx = 17

# random initial m0
ms = m0s * rand()
mf = (1 - m0s) * rand()
ϑ = rand() * π / 2
φ = rand() * 2π

## ######################################################################################
# Compare function implementation (using a constant amplitude) to ω1::Number implementation
#########################################################################################
f_ω1(t) = α/TRF
G = interpolate_greens_function(greens_superlorentzian, 0, TRF / T2s);

## isolated semi-solid spin pool
for ω0 ∈ [0, 100randn()]
    local m0 = [rand(), 1]
    mfun(p, t; idxs = nothing) = typeof(idxs) <: Number ? m0[idxs] : m0

    local p = (α/TRF, B1, ω0, R1s, T2s, G)
    u_ω1Number = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0.0, TRF), p))[end]

    local p = (f_ω1, B1, ω0, R1s, T2s, G)
    u_ω1Function = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0.0, TRF), p))[end]

    @test u_ω1Number ≈ u_ω1Function
end


## Bloch-McConnell for exchanging pools
for ω0 ∈ [0, 100randn()]
    local m0 = [mf * sin(ϑ) * cos(φ), mf * sin(ϑ) * sin(φ), mf * cos(ϑ), ms, 1]
    mfun(p, t; idxs = nothing) = typeof(idxs) <: Number ? m0[idxs] : m0

    local p = (α/TRF, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G)
    u_ω1Number = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0.0, TRF), p))[end]

    local p = (f_ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G)
    u_ω1Function = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0.0, TRF), p))[end]

    @test u_ω1Number ≈ u_ω1Function
end



## ######################################################################################
# Compare a truly shaped RF pulse w/ Lorentzian LS to Bloch simulation
#########################################################################################
G = greens_lorentzian

NSideLobes = 1
f_ω1(t) = sinc(2(NSideLobes+1) * t/TRF - (NSideLobes+1)) * α / (sinint((NSideLobes+1)π) * TRF/π / (NSideLobes+1))
@test quadgk(f_ω1, 0, TRF)[1] ≈ α

## Solve org. Bloch for isolated semi-solid spin pool
function apply_hamiltonian_bloch!(∂m∂t, m, p::NTuple{5,Any}, t)
    ω1, B1, ω0, R1s, T2s = p
    R2s = 1/T2s

    H = [
           -R2s  -ω0  B1*ω1(t) 0
             ω0 -R2s         0 0
      -B1*ω1(t)    0      -R1s 0
              0    0         0 0
    ]
    ∂m∂t .= H * m
    return ∂m∂t
end

for ω0 ∈ [0, 100randn()]
    local m0 = [0, 0, m0s, 1]
    local p = (f_ω1, B1, ω0, R1s, T2s)
    z_Bloch = solve(ODEProblem(apply_hamiltonian_bloch!, m0, (0.0, TRF), p))[end][3]

    # Solve gen. Bloch for isolated semi-solid spin pool
    local m0 = [m0s, 1]
    mfun(p, t; idxs = nothing) = typeof(idxs) <: Number ? m0[idxs] : m0

    local p = (f_ω1, B1, ω0, R1s, T2s, G)
    z_gBloch = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0.0, TRF), p))[end][1]

    @test z_gBloch ≈ z_Bloch atol = 5e-3
end


## Solve org. Bloch-McConnell for coupled pool system
function apply_hamiltonian_bloch!(∂m∂t, m, p::NTuple{9,Any}, t)
    f_ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s = p
    ω1 = f_ω1(t)
    R2s = 1/T2s

    H = [
            -R2f  -ω0       B1*ω1      0     0            0       0
              ω0 -R2f           0      0     0            0       0
          -B1*ω1    0 -R1f-Rx*m0s      0     0       Rx*m0f R1f*m0f
               0    0           0   -R2s   -ω0        B1*ω1       0
               0    0           0     ω0  -R2s            0       0
               0    0      Rx*m0s -B1*ω1     0  -R1s-Rx*m0f R1s*m0s
               0    0           0      0     0            0       0
    ]
    ∂m∂t .= H * m
    return ∂m∂t
end

for ω0 ∈ [0, 100randn()]
    local m0 = [mf * sin(ϑ) * cos(φ), mf * sin(ϑ) * sin(φ), mf * cos(ϑ), 0, 0, ms, 1]

    local p = (f_ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s)
    local u_Bloch = solve(ODEProblem(apply_hamiltonian_bloch!, m0, (0.0, TRF), p))[end][[1:3...,6]]

    # Solve gen. Bloch for isolated semi-solid spin pool
    local m0 = [mf * sin(ϑ) * cos(φ), mf * sin(ϑ) * sin(φ), mf * cos(ϑ), ms, 1]
    mfun(p, t; idxs = nothing) = typeof(idxs) <: Number ? m0[idxs] : m0

    local p = (f_ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G)
    local u_gBloch = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0.0, TRF), p))[end][1:4]

    @test u_gBloch ≈ u_Bloch rtol = 1e-2
end