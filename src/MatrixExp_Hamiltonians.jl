##
"""
    hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s[, dR2sdT2s, dR2sdB1, grad_type])

Calculate the hamiltonian of the linear approximation of the generalized Bloch model.

If no gradient is supplied, it returns a 6x6 (static) matrix with the dimensions (in this order) `[xf, yf, zf, xs, zs, 1]`; the attached 1 is a mathematical trick to allow for ``T_1`` relaxation to a non-zero thermal equilibrium.
If a gradient is supplied, it returns a 11x11 (static) matrix with the dimensions (in this order) `[xf, yf, zf, xs, zs, dxf/dθ, dyf/dθ, dzf/dθ, dxs/dθ, dzs/dθ,  1]`, where `θ` is the parameter specified by `grad_type`

# Arguments
- `ω1::Number`: Rabi frequency in rad/s (rotation about the y-axis)
- `B1::Number`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `ω0::Number`: Larmor (or off-resonance) frequency in rad/s (rotation about the z-axis)
- `T::Number`: Time in seconds; this can, e.g., be the RF-pulse duration, or the time of free precession with `ω1=0`
- `m0s::Number`: Fractional size of the semi-solid pool; should be in range of 0 to 1
- `R1f::Number`: Longitudinal relaxation rate of the free pool in 1/seconds
- `R2f::Number`: Transversal relaxation rate of the free pool in 1/seconds
- `Rx::Number`: Exchange rate between the two spin pools in 1/seconds
- `R1s::Number`: Longitudinal relaxation rate of the semi-solid pool in 1/seconds
- `R2s::Number`: Transversal relaxation rate of the semi-solid pool in 1/seconds; this number can be calculated with the first function returned by [`precompute_R2sl`](@ref) to implement the linear approximation described in the generalized Bloch paper

Optional:
- `dR2sdT2s::Number`: Derivative of linearized R2sl wrt. the actual T2s; only required if `grad_type = grad_T2s()`; this number can be calculated with the second function returned by [`precompute_R2sl`](@ref)
- `dR2sdB1::Number`: Derivative of linearized R2sl wrt. B1; only required if `grad_type = grad_B1()`; this number can be calculated with the third function returned by [`precompute_R2sl`](@ref)
- `grad_type::grad_param`: `grad_m0s()`, `grad_R1f()`, `grad_R1s()`, `grad_R2f()`, `grad_Rx()`, `grad_T2s()`, `grad_ω0()`, or `grad_B1()`; create one hamiltonian for each desired gradient

# Examples
```jldoctest
julia> α = π;

julia> T = 500e-6;

julia> ω1 = α/T;

julia> B1 = 1;

julia> ω0 = 0;

julia> m0s = 0.1;

julia> R1f = 1;

julia> R2f = 15;

julia> Rx = 30;

julia> R1s = 6.5;

julia> R2s = 1e5;

julia> m0 = [0, 0, 1-m0s, 0, m0s, 1];

julia> (xf, yf, zf, xs, zs, _) = exp(hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s)) * m0
6-element StaticArraysCore.SVector{6, Float64} with indices SOneTo(6):
  0.0010647535813058293
  0.0
 -0.8957848274535014
  0.005126529591877105
  0.08122007142111888
  1.0
```
"""
function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0         0;
               ω0  -R2f               0         0               0         0;
         -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f R1f * m0f;
                0     0               0      -R2s         B1 * ω1         0;
                0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f R1s * m0s;
                0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, _, _, _)
    return hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s)
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, _, _, grad_type::grad_m0s)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f        0     0               0         0               0 R1f * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f        0     0               0         0               0 R1s * m0s;
                0     0               0         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                0     0               0         0               0       ω0  -R2f               0         0               0         0;
                0     0             -Rx         0             -Rx -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f      -R1f;
                0     0               0         0               0        0     0               0      -R2s         B1 * ω1         0;
                0     0              Rx         0              Rx        0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f       R1s;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, _, _, grad_type::grad_R1f)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f        0     0               0         0               0 R1f * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f        0     0               0         0               0 R1s * m0s;
                0     0               0         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                0     0               0         0               0       ω0  -R2f               0         0               0         0;
                0     0              -1         0               0 -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f       m0f;
                0     0               0         0               0        0     0               0      -R2s         B1 * ω1         0;
                0     0               0         0               0        0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f         0;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, _, _, grad_type::grad_R1s)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f        0     0               0         0               0 R1f * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f        0     0               0         0               0 R1s * m0s;
                0     0               0         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                0     0               0         0               0       ω0  -R2f               0         0               0         0;
                0     0               0         0               0 -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f         0;
                0     0               0         0               0        0     0               0      -R2s         B1 * ω1         0;
                0     0               0         0              -1        0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f       m0s;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end
function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, _, _, grad_type::grad_R1a)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f        0     0               0         0               0 R1f * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f        0     0               0         0               0 R1s * m0s;
                0     0               0         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                0     0               0         0               0       ω0  -R2f               0         0               0         0;
                0     0              -1         0               0 -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f       m0f;
                0     0               0         0               0        0     0               0      -R2s         B1 * ω1         0;
                0     0               0         0              -1        0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f       m0s;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, _, _, grad_type::grad_R2f)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f        0     0               0         0               0 R1f * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f        0     0               0         0               0 R1s * m0s;
               -1     0               0         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                0    -1               0         0               0       ω0  -R2f               0         0               0         0;
                0     0               0         0               0 -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f         0;
                0     0               0         0               0        0     0               0      -R2s         B1 * ω1         0;
                0     0               0         0               0        0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f         0;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, _, _, grad_type::grad_Rx)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f        0     0               0         0               0 R1f * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f        0     0               0         0               0 R1s * m0s;
                0     0               0         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                0     0               0         0               0       ω0  -R2f               0         0               0         0;
                0     0            -m0s         0             m0f -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f         0;
                0     0               0         0               0        0     0               0      -R2s         B1 * ω1         0;
                0     0             m0s         0            -m0f        0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f         0;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, dR2sdT2s, _, grad_type::grad_T2s)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f        0     0               0         0               0 R1f * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f        0     0               0         0               0 R1s * m0s;
                0     0               0         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                0     0               0         0               0       ω0  -R2f               0         0               0         0;
                0     0               0         0               0 -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f         0;
                0     0               0 -dR2sdT2s               0        0     0               0      -R2s         B1 * ω1         0;
                0     0               0         0               0        0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f         0;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, _, _, grad_type::grad_ω0)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f        0     0               0         0               0 R1f * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f        0     0               0         0               0 R1s * m0s;
                0    -1               0         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                1     0               0         0               0       ω0  -R2f               0         0               0         0;
                0     0               0         0               0 -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f         0;
                0     0               0         0               0        0     0               0      -R2s         B1 * ω1         0;
                0     0               0         0               0        0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f         0;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, _, dR2sdB1, grad_type::grad_B1)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f        0     0               0         0               0 R1f * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f        0     0               0         0               0 R1s * m0s;
                0     0              ω1         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                0     0               0         0               0       ω0  -R2f               0         0               0         0;
              -ω1     0               0         0               0 -B1 * ω1     0 -R1f - Rx * m0s         0        Rx * m0f         0;
                0     0               0  -dR2sdB1              ω1        0     0               0      -R2s         B1 * ω1         0;
                0     0               0       -ω1               0        0     0        Rx * m0s  -B1 * ω1 -R1s - Rx * m0f         0;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function propagator_linear_inversion_pulse(ω1, T, B1, R2s, _, _, _)
    Hs = @SMatrix [    -R2s         B1 * ω1;
                   -B1 * ω1               0]
    Us = exp(Hs * T)

    U = @SMatrix [
        sin(B1 * ω1 * T / 2)^2  0 0 0 0 0;
        0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0;
        0 0 cos(B1 * ω1 * T)        0 0 0;
        0 0 0 Us[1,1]    Us[1,2]        0;
        0 0 0 Us[2,1]    Us[2,2]        0;
        0 0 0 0          0              1]
    return U
end

function propagator_linear_inversion_pulse(ω1, T, B1, R2s, _, _, grad_type::grad_param)
    Hs = @SMatrix [    -R2s         B1 * ω1;
                   -B1 * ω1               0]
    Us = exp(Hs * T)

    U = @SMatrix [
        sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0 0 0;
        0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0 0 0;
        0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0 0 0;
        0 0 0 Us[1,1]    Us[1,2]        0 0 0 0 0 0;
        0 0 0 Us[2,1]    Us[2,2]        0 0 0 0 0 0;
        0 0 0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0;
        0 0 0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0;
        0 0 0 0 0 0 0 cos(B1 * ω1 * T)        0 0 0;
        0 0 0 0 0 0 0 0 Us[1,1]    Us[1,2]        0;
        0 0 0 0 0 0 0 0 Us[2,1]    Us[2,2]        0;
        0 0 0 0 0 0 0 0 0          0              1]
    return U
end

function propagator_linear_inversion_pulse(ω1, T, B1, R2s, dR2sdT2s, _, grad_type::grad_T2s)
    Hs = @SMatrix [    -R2s  B1 * ω1;
                   -B1 * ω1        0]
    Us = exp(Hs * T)

    dHsdT2s = @SMatrix [-dR2sdT2s   0;
                                0   0]

    # Higham's Complex Step Approximation:
    h = 1im * eps()
    dU = real.(exp((Hs + h * dHsdT2s) * T) ./ h)

    U = @SMatrix [
        sin(B1 * ω1 * T / 2)^2    0 0 0 0 0 0 0 0 0 0;
        0 -sin(B1 * ω1 * T / 2)^2   0 0 0 0 0 0 0 0 0;
        0 0 cos(B1 * ω1 * T)          0 0 0 0 0 0 0 0;
        0 0 0 Us[1,1]    Us[1,2]          0 0 0 0 0 0;
        0 0 0 Us[2,1]    Us[2,2]          0 0 0 0 0 0;
        0 0 0 0 0 sin(B1 * ω1 * T / 2)^2    0 0 0 0 0;
        0 0 0 0 0 0 -sin(B1 * ω1 * T / 2)^2   0 0 0 0;
        0 0 0 0 0 0 0 cos(B1 * ω1 * T)          0 0 0;
        0 0 0 dU[1,1] dU[1,2] 0 0 0 Us[1,1] Us[1,2] 0;
        0 0 0 dU[2,1] dU[2,2] 0 0 0 Us[2,1] Us[2,2] 0;
        0 0 0 0       0       0 0 0 0       0       1]
    return U
end

function propagator_linear_inversion_pulse(ω1, T, B1, R2s, _, dR2sdB1, grad_type::grad_B1)
    Hs = @SMatrix [    -R2s  B1 * ω1;
                   -B1 * ω1        0]
    Us = exp(Hs * T)

    dHsdB1 = @SMatrix [-dR2sdB1  ω1;
                            -ω1   0]

    # Higham's Complex Step Approximation:
    h = 1im * eps()
    dU = real.(exp((Hs + h * dHsdB1) * T) ./ h)

    U = @SMatrix [
        sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0 0 0;
        0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0 0 0;
        0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0 0 0;
        0 0 0 Us[1,1]    Us[1,2]        0 0 0 0 0 0;
        0 0 0 Us[2,1]    Us[2,2]        0 0 0 0 0 0;
        sin(B1 * ω1 * T / 2) * cos(B1 * ω1 * T / 2) * ω1 * T  0 0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0;
        0 -sin(B1 * ω1 * T / 2) * cos(B1 * ω1 * T / 2) * ω1 * T 0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0;
        0 0 -sin(B1 * ω1 * T) * ω1 * T                            0 0 0 0 cos(B1 * ω1 * T)        0 0 0;
        0 0 0 dU[1,1] dU[1,2] 0 0 0 Us[1,1] Us[1,2] 0;
        0 0 0 dU[2,1] dU[2,2] 0 0 0 Us[2,1] Us[2,2] 0;
        0 0 0 0       0       0 0 0 0       0       1]
    return U
end

function z_rotation_propagator(rfphase_increment, _)
    u_rot = @SMatrix [cos(rfphase_increment) -sin(rfphase_increment) 0 0 0 0;
                      sin(rfphase_increment)  cos(rfphase_increment) 0 0 0 0;
                                     0                0  1 0 0 0;
                                     0                0  0 1 0 0;
                                     0                0  0 0 1 0;
                                     0                0  0 0 0 1]
    return u_rot
end

function z_rotation_propagator(rfphase_increment, grad::grad_param)
    u_rot = @SMatrix [cos(rfphase_increment) -sin(rfphase_increment) 0 0 0 0                 0                0 0 0 0;
                      sin(rfphase_increment)  cos(rfphase_increment) 0 0 0 0                 0                0 0 0 0;
                      0                 0                1 0 0 0                 0                0 0 0 0;
                      0                 0                0 1 0 0                 0                0 0 0 0;
                      0                 0                0 0 1 0                 0                0 0 0 0;
                      0                 0                0 0 0 cos(rfphase_increment) -sin(rfphase_increment) 0 0 0 0;
                      0                 0                0 0 0 sin(rfphase_increment)  cos(rfphase_increment) 0 0 0 0;
                      0                 0                0 0 0 0                 0                1 0 0 0;
                      0                 0                0 0 0 0                 0                0 1 0 0;
                      0                 0                0 0 0 0                 0                0 0 1 0;
                      0                 0                0 0 0 0                 0                0 0 0 1]
    return u_rot
end

function xs_destructor(_)
    @SMatrix [
        1 0 0 0 0 0;
        0 1 0 0 0 0;
        0 0 1 0 0 0;
        0 0 0 0 0 0;
        0 0 0 0 1 0;
        0 0 0 0 0 1]
end

function xs_destructor(grad::grad_param)
    @SMatrix [
        1 0 0 0 0 0 0 0 0 0 0;
        0 1 0 0 0 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 1 0 0 0 0 0 0;
        0 0 0 0 0 1 0 0 0 0 0;
        0 0 0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 0 0 1 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 1 0;
        0 0 0 0 0 0 0 0 0 0 1]
end

    function A0(_)
    @SMatrix [
        1 0 0 0 0 0;
        0 1 0 0 0 0;
        0 0 1 0 0 0;
        0 0 0 1 0 0;
        0 0 0 0 1 0;
        0 0 0 0 0 0]
end

function A0(grad::grad_param)
    @SMatrix [
        1 0 0 0 0 0 0 0 0 0 0;
        0 1 0 0 0 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0 0 0 0;
        0 0 0 1 0 0 0 0 0 0 0;
        0 0 0 0 1 0 0 0 0 0 0;
        0 0 0 0 0 1 0 0 0 0 0;
        0 0 0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 0 0 1 0 0 0;
        0 0 0 0 0 0 0 0 1 0 0;
        0 0 0 0 0 0 0 0 0 1 0;
        0 0 0 0 0 0 0 0 0 0 0]
end

function C(_)
    @SVector [0,0,0,0,0,1]
end

function C(grad::grad_param)
    @SVector [0,0,0,0,0,0,0,0,0,0,1]
end