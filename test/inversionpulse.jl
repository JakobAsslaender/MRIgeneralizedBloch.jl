using MRIgeneralizedBloch
using MRIgeneralizedBloch:propagator_linear_crushed_pulse
using StaticArrays
using Test

##
T = 500e-6
T2s = 10e-6

R2slT = precompute_R2sl()
R2sl = R2slT[1](T, π, 1, T2s)

function simulate_inversion_pulse(M0, ωz, B1, T, m0s, R1f, R2f, K, nTR, R1s, R2sl)
    Δ = π * (-1 + .0001:.0001:1) # * 2.17 / 2

    M  = zeros(6)
    for i in eachindex(Δ)
        Rz = exp(hamiltonian_linear(0, 1, Δ[i], 1, 0, 0, 0, 0, 0, 0, 0, 0)) # spoiler
        Ry = exp(hamiltonian_linear(π / T, B1, ωz, T, m0s, R1f, R2f, K, nTR, R1s, R2sl))
        M += Rz * (Ry * (Rz * M0))
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
K = 1e-6
nTR = 0.07
R1s = 1e-6

for i = 1:10
    local B1 = 1 + randn() / 4
    u_fp = exp(hamiltonian_linear(0, B1, ωz, T / 2, m0s, R1f, R2f, K, nTR, R1s, R2sl))
    UE = u_fp * propagator_linear_crushed_pulse(π / T, T, B1, R2sl, undef, undef, undef) * u_rot * u_fp

    M0 = SVector(1 - m0s, 0, 0, 0, m0s, 1)
    MB = simulate_inversion_pulse(M0, ωz, B1, T, m0s, R1f, R2f, K, nTR, R1s, R2sl)
    ME = UE * M0
    @test ME ≈ MB

    M0 = SVector(0, 1 - m0s, 0, 0, m0s, 1)
    MB = simulate_inversion_pulse(M0, ωz, B1, T, m0s, R1f, R2f, K, nTR, R1s, R2sl)
    ME = UE * M0
    @test ME ≈ MB

    M0 = SVector(0, 0, 1 - m0s, 0, m0s, 1)
    MB = simulate_inversion_pulse(M0, ωz, B1, T, m0s, R1f, R2f, K, nTR, R1s, R2sl)
    ME = UE * M0
    @test ME ≈ MB
end


## Test with exchange: in this case our approach is an approximation
m0s = 0.15
R1f = 0.3
R2f = 10
K = 30
nTR = 0.07
R1s = 2

rtolmax = 1e-2
for i = 1:10
    local B1 = 1 + randn() / 4
    u_fp = exp(hamiltonian_linear(0, B1, ωz, T / 2, m0s, R1f, R2f, K, nTR, R1s, R2sl))
    UE = u_fp * propagator_linear_crushed_pulse(π / T, T, B1, R2sl, undef, undef, undef) * u_rot * u_fp

    M0 = SVector(1 - m0s, 0, 0, 0, m0s, 1)
    MB = simulate_inversion_pulse(M0, ωz, B1, T, m0s, R1f, R2f, K, nTR, R1s, R2sl)
    ME = UE * M0
    @test ME[[1,2,3,5,6]] ≈ MB[[1,2,3,5,6]] rtol = rtolmax

    M0 = SVector(0, 1 - m0s, 0, 0, m0s, 1)
    MB = simulate_inversion_pulse(M0, ωz, B1, T, m0s, R1f, R2f, K, nTR, R1s, R2sl)
    ME = UE * M0
    @test ME[[1,2,3,5,6]] ≈ MB[[1,2,3,5,6]] rtol = rtolmax

    M0 = SVector(0, 0, 1 - m0s, 0, m0s, 1)
    MB = simulate_inversion_pulse(M0, ωz, B1, T, m0s, R1f, R2f, K, nTR, R1s, R2sl)
    ME = UE * M0
    @test ME[[1,2,3,5,6]] ≈ MB[[1,2,3,5,6]] rtol = rtolmax
end