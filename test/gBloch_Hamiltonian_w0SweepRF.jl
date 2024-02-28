##
using MRIgeneralizedBloch
using DifferentialEquations
using SpecialFunctions
using QuadGK
using LinearAlgebra
using Test

## Pulse parameters
α = 5π
TRF = 10.24e-3

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
# Compare function implementation (using a constant amplitude) to ω1::Real implementation
#########################################################################################
f_ω1(t) = α/TRF
ω0 = 0
f_φ(t)  = ω0 * t
G = interpolate_greens_function(greens_superlorentzian, 0, TRF / T2s)

## isolated semi-solid spin pool
m0 = [rand(), 1]
mfun(p, t; idxs = nothing) = typeof(idxs) <: Number ? m0[idxs] : m0

p = (α/TRF, B1, ω0, R1s, T2s, G)
u_ω1Number = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), p)).u[end]

p = (f_ω1, B1, f_φ, R1s, T2s, G)
u_ω1Function = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), p)).u[end]

@test u_ω1Number ≈ u_ω1Function


## Bloch-McConnell for exchanging pools
m0 = [mf * sin(ϑ) * cos(φ), mf * sin(ϑ) * sin(φ), mf * cos(ϑ), ms, 1]
mfun(p, t; idxs = nothing) = typeof(idxs) <: Number ? m0[idxs] : m0

p = (α/TRF, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G)
u_ω1Number = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), p)).u[end]

p = (f_ω1, B1, f_φ, m0s, R1f, R2f, Rx, R1s, T2s, G)
u_ω1Function = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), p)).u[end]

@test u_ω1Number ≈ u_ω1Function


## ######################################################################################
# Compare a truly shaped RF pulse w/ Lorentzian LS to Bloch simulation
#########################################################################################
G = greens_lorentzian

γ = 267.522e6 # rad/s/T
ω₁ᵐᵃˣ = 13e-6 * γ # rad/s
μ = 5 # rad
β = 674.1 # 1/s

f_ω1(t) = ω₁ᵐᵃˣ .* sech.(β * (t - TRF/2)) # rad/s
f_ω0(t) = -μ * β * tanh.(β * (t - TRF/2)) # rad/s
f_φ(t)  = -μ * log(cosh(β * t) - sinh(β*t) * tanh(β*TRF/2))

@test ω₁ᵐᵃˣ /(√μ * β) > 1 # adiabatic condition


## Cf. org. Bloch and gBloch for isolated free spin pool
function apply_hamiltonian_bloch!(∂m∂t, m, p::NTuple{5,Any}, t)
    ω1, B1, ω0, R1s, T2s = p
    R2s = 1/T2s

    H = [
           -R2s  -ω0(t)  B1*ω1(t)   0
          ω0(t)    -R2s         0   0
      -B1*ω1(t)       0      -R1s R1s
              0       0         0   0
    ]

    ∂m∂t .= H * m
    return ∂m∂t
end

m0 = [0, 0, 1, 1]
p = (f_ω1, B1, f_ω0, R1s, 1/R2f)
u_Bloch = solve(ODEProblem(apply_hamiltonian_bloch!, m0, (0, TRF), p))

# Solve gBloch
m0 = [1, 1]
mfun(p, t; idxs = nothing) = typeof(idxs) <: Number ? m0[idxs] : m0
p = (f_ω1, B1, f_φ, R1s, 1/R2f, G)
z_gBloch = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0.0, TRF), p))
x_gBloch(t) =  B1 * quadgk(x -> f_ω1(x) * cos(f_φ(x)) * G((t - x) * R2f) * z_gBloch(x)[1], 0, t, order=100)[1]
y_gBloch(t) = -B1 * quadgk(x -> f_ω1(x) * sin(f_φ(x)) * G((t - x) * R2f) * z_gBloch(x)[1], 0, t, order=100)[1]
mt_gBloch(t) = (x_gBloch(t) + 1im * y_gBloch(t)) * exp(1im * (f_φ(t)))

@test [real(mt_gBloch(TRF)), imag(mt_gBloch(TRF)), z_gBloch(TRF)[1]] ≈ u_Bloch(TRF)[1:3] rtol = 1e-3
@test z_gBloch(TRF)[1] ≈ -1 atol = 1e-1 # inversion efficiency should be close to 1

## Cf. org. Bloch and gBloch for isolated semi-solid spin pool
m0 = [0, 0, 1, 1]
p = (f_ω1, B1, f_ω0, R1s, T2s)
u_Bloch = solve(ODEProblem(apply_hamiltonian_bloch!, m0, (0, TRF), p))

# Solve gBloch
m0 = [1, 1]
mfun(p, t; idxs = nothing) = typeof(idxs) <: Number ? m0[idxs] : m0
p = (f_ω1, B1, f_φ, R1s, T2s, G)
z_gBloch = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), p))
x_gBloch(t) =  B1 * quadgk(x -> f_ω1(x) * cos(f_φ(x)) * G((t - x) / T2s) * z_gBloch(x)[1], 0, t, order=100)[1]
y_gBloch(t) = -B1 * quadgk(x -> f_ω1(x) * sin(f_φ(x)) * G((t - x) / T2s) * z_gBloch(x)[1], 0, t, order=100)[1]
mt_gBloch(t) = (x_gBloch(t) + 1im * y_gBloch(t)) * exp(1im * (f_φ(t)))

@test [real(mt_gBloch(TRF)), imag(mt_gBloch(TRF)), z_gBloch(TRF)[1]] ≈ u_Bloch(TRF)[1:3] rtol = 1e-3



## Solve org. Bloch-McConnell for coupled pool system
function apply_hamiltonian_bloch!(∂m∂t, m, p::NTuple{9,Any}, t)
    f_ω1, B1, f_ω0, m0s, R1f, R2f, Rx, R1s, T2s = p
    ω1 = f_ω1(t)
    ω0 = f_ω0(t)
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

m0 = [0, 0, 1-m0s, 0, 0, m0s, 1]
p = (f_ω1, B1, f_ω0, m0s, R1f, R2f, Rx, R1s, T2s)
u_Bloch = solve(ODEProblem(apply_hamiltonian_bloch!, m0, (0.0, TRF), p))

# Solve gen. Bloch for isolated semi-solid spin pool
m0 = [0, 0, 1-m0s, m0s, 1]
mfun(p, t; idxs = nothing) = typeof(idxs) <: Number ? m0[idxs] : m0
p = (f_ω1, B1, f_φ, m0s, R1f, R2f, Rx, R1s, T2s, G)
u_gBloch = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0.0, TRF), p))

xs_gBloch(t) =  B1 * quadgk(x -> f_ω1(x) * cos(f_φ(x)) * G((t - x) / T2s) * u_gBloch(x)[4], 0, t, order=100)[1]
ys_gBloch(t) = -B1 * quadgk(x -> f_ω1(x) * sin(f_φ(x)) * G((t - x) / T2s) * u_gBloch(x)[4], 0, t, order=100)[1]

mts_gBloch(t) = (xs_gBloch(t)   + 1im * ys_gBloch(t))   * exp(1im * (f_φ(t)))
mtf_gBloch(t) = (u_gBloch(t)[1] + 1im * u_gBloch(t)[2]) * exp(1im * (f_φ(t)))

@test [real(mtf_gBloch(TRF)), imag(mtf_gBloch(TRF)), u_gBloch(TRF)[3]] ≈ u_Bloch(TRF)[1:3] rtol = 5e-3
@test [real(mts_gBloch(TRF)), imag(mts_gBloch(TRF)), u_gBloch(TRF)[4]] ≈ u_Bloch(TRF)[4:6] rtol = 1e-3


##
# plot(u_Bloch, idxs=1:3)
# plot!(t -> real(mtf_gBloch(t)), 0, TRF)
# plot!(t -> imag(mtf_gBloch(t)), 0, TRF)
# plot!(u_gBloch, idxs=3)

# ##
# plot(u_Bloch, idxs=4:6)
# plot!(t -> real(mts_gBloch(t)), 0, TRF)
# plot!(t -> imag(mts_gBloch(t)), 0, TRF)
# plot!(u_gBloch, idxs=4)