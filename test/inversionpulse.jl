using MRIgeneralizedBloch
using MRIgeneralizedBloch:propagator_linear_crushed_pulse
using StaticArrays
using Test

##
T = 500e-6
T2s = 10e-6
M0 = 0.85

R2slT = precompute_R2sl()
R2sl = R2slT[1](T, π, 1, T2s)

function simulate_inversion_pulse(m0_v, ωz, B1, T, M0, m0s, R1f, R2f, Rex, R1s, R2sl)
    Δ = π * (-1 + .0001:.0001:1) # * 2.17 / 2

    M  = zeros(6)
    for i in eachindex(Δ)
        Rz = exp(hamiltonian_linear(0, 1, Δ[i], 1, M0, 0, 0, 0, 0, 0, 0)) # spoiler
        Ry = exp(hamiltonian_linear(π / T, B1, ωz, T, M0, m0s, R1f, R2f, Rex, R1s, R2sl))
        M += Rz * (Ry * (Rz * m0_v))
    end
    M /= length(Δ)
end

u_rot = @SMatrix [cos(π) -sin(π) 0 0 0 0;
                  sin(π)  cos(π) 0 0 0 0;
                      0       0  1 0 0 0;
                      0       0  0 1 0 0;
                      0       0  0 0 1 0;
                      0       0  0 0 0 1]

## Test w/o exchange: in this case we expect equivalence except numerical differences
ωz = 0 # off resonance is implemented in the surrounding free presession part
m0s = 0
R1f = 1e-6
R2f = 1e-6
Rex = 1e-6
R1s = 1e-6

for i = 1:10
    local B1 = 1 + randn() / 4
    u_fp = exp(hamiltonian_linear(0, B1, ωz, T / 2, M0, m0s, R1f, R2f, Rex, R1s, R2sl))
    UE = u_fp * propagator_linear_crushed_pulse(π / T, T, B1, R2sl, nothing, nothing, nothing) * u_rot * u_fp

    m0_v = SVector(M0 * (1 - m0s), 0, 0, 0, M0 * m0s, 1)
    MB = simulate_inversion_pulse(m0_v, ωz, B1, T, M0, m0s, R1f, R2f, Rex, R1s, R2sl)
    ME = UE * m0_v
    @test ME ≈ MB

    m0_v = SVector(0, M0 * (1 - m0s), 0, 0, M0 * m0s, 1)
    MB = simulate_inversion_pulse(m0_v, ωz, B1, T, M0, m0s, R1f, R2f, Rex, R1s, R2sl)
    ME = UE * m0_v
    @test ME ≈ MB

    m0_v = SVector(0, 0, M0 * (1 - m0s), 0, M0 * m0s, 1)
    MB = simulate_inversion_pulse(m0_v, ωz, B1, T, M0, m0s, R1f, R2f, Rex, R1s, R2sl)
    ME = UE * m0_v
    @test ME ≈ MB
end


## Test with exchange: in this case our approach is an approximation
m0s = 0.15
R1f = 0.3
R2f = 10
Rex = 30
R1s = 2

rtolmax = 1e-2
for i = 1:10
    local B1 = 1 + randn() / 4
    u_fp = exp(hamiltonian_linear(0, B1, ωz, T / 2, M0, m0s, R1f, R2f, Rex, R1s, R2sl))
    UE = u_fp * propagator_linear_crushed_pulse(π / T, T, B1, R2sl, nothing, nothing, nothing) * u_rot * u_fp

    m0_v = SVector(M0 * (1 - m0s), 0, 0, 0, M0 * m0s, 1)
    MB = simulate_inversion_pulse(m0_v, ωz, B1, T, M0, m0s, R1f, R2f, Rex, R1s, R2sl)
    ME = UE * m0_v
    @test ME[[1,2,3,5,6]] ≈ MB[[1,2,3,5,6]] rtol = rtolmax

    m0_v = SVector(0, M0 * (1 - m0s), 0, 0, M0 * m0s, 1)
    MB = simulate_inversion_pulse(m0_v, ωz, B1, T, M0, m0s, R1f, R2f, Rex, R1s, R2sl)
    ME = UE * m0_v
    @test ME[[1,2,3,5,6]] ≈ MB[[1,2,3,5,6]] rtol = rtolmax

    m0_v = SVector(0, 0, M0 * (1 - m0s), 0, M0 * m0s, 1)
    MB = simulate_inversion_pulse(m0_v, ωz, B1, T, M0, m0s, R1f, R2f, Rex, R1s, R2sl)
    ME = UE * m0_v
    @test ME[[1,2,3,5,6]] ≈ MB[[1,2,3,5,6]] rtol = rtolmax
end