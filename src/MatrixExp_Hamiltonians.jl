##
"""
    hamiltonian_linear(ω1, B1, ω0, T, M0, m0s, R1f, R2f, Rex, R1s, R2s[, dR2sdT2s, dR2sdB1, grad_type])

Calculate the hamiltonian of the linear approximation of the generalized Bloch model.

If no gradient is supplied, it returns a 6x6 (static) matrix with the dimensions (in this order) `[xf, yf, zf, xs, zs, 1]`; the attached 1 is a mathematical trick to allow for ``T_1`` relaxation to a non-zero thermal equilibrium.
If a gradient is supplied, it returns a 11x11 (static) matrix with the dimensions (in this order) `[xf, yf, zf, xs, zs, dxf/dθ, dyf/dθ, dzf/dθ, dxs/dθ, dzs/dθ,  1]`, where `θ` is the parameter specified by `grad_type`

# Arguments
- `ω1::Real`: Rabi frequency in rad/s (rotation about the y-axis)
- `B1::Real`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `ω0::Real`: Larmor (or off-resonance) frequency in rad/s (rotation about the z-axis)
- `T::Real`: Time in seconds; this can, e.g., be the RF-pulse duration, or the time of free precession with `ω1=0`
- `M0::Real`: Equilibrium magnetization (proton density) scaling factor
- `m0s::Real`: Fractional size of the semi-solid pool; should be in range of 0 to 1
- `R1f::Real`: Longitudinal relaxation rate of the free pool in 1/seconds
- `R2f::Real`: Transversal relaxation rate of the free pool in 1/seconds
- `Rex::Real`: Exchange rate between the two spin pools in 1/seconds
- `R1s::Real`: Longitudinal relaxation rate of the semi-solid pool in 1/seconds
- `R2s::Real`: Transversal relaxation rate of the semi-solid pool in 1/seconds; this number can be calculated with the first function returned by [`precompute_R2sl`](@ref) to implement the linear approximation described in the generalized Bloch paper

# Optional Arguments:
- `dR2sdT2s::Real`: Derivative of linearized R2sl wrt. the actual T2s; only required if `grad_type = grad_T2s()`; this number can be calculated with the second function returned by [`precompute_R2sl`](@ref)
- `dR2sdB1::Real`: Derivative of linearized R2sl wrt. B1; only required if `grad_type = grad_B1()`; this number can be calculated with the third function returned by [`precompute_R2sl`](@ref)
- `grad_type::grad_param`: `grad_m0s()`, `grad_R1f()`, `grad_R1s()`, `grad_R2f()`, `grad_Rex()`, `grad_T2s()`, `grad_ω0()`, or `grad_B1()`; create one hamiltonian for each desired gradient

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

julia> Rex = 30;

julia> R1s = 6.5;

julia> R2s = 1e5;

julia> M0 = 1;

julia> m0 = [0, 0, M0*(1-m0s), 0, M0*m0s, 1];

julia> (xf, yf, zf, xs, zs, _) = exp(hamiltonian_linear(ω1, B1, ω0, T, M0, m0s, R1f, R2f, Rex, R1s, R2s)) * m0
6-element StaticArraysCore.SVector{6, Float64} with indices SOneTo(6):
  0.0010647535813058293
  0.0
 -0.8957848274535014
  0.005126529591877105
  0.08122007142111888
  1.0
```
"""
function hamiltonian_linear(ω1, B1, ω0, T, M0, m0s, R1f, R2f, Rex, R1s, R2s)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0          B1 * ω1         0                0              0;
               ω0  -R2f                0         0                0              0;
         -B1 * ω1     0 -R1f - Rex * m0s         0        Rex * m0f R1f * M0 * m0f;
                0     0                0      -R2s          B1 * ω1              0;
                0     0        Rex * m0s  -B1 * ω1 -R1s - Rex * m0f R1s * M0 * m0s;
                0     0                0         0                0              0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, M0, m0s, R1f, R2f, Rex, R1s, R2s, _, _, _)
    return hamiltonian_linear(ω1, B1, ω0, T, M0, m0s, R1f, R2f, Rex, R1s, R2s)
end

function propagator_linear_crushed_pulse(ω1, T, B1, R2s, _, _, _)
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

function propagator_linear_crushed_pulse(ω1, T, B1, R2s, _, _, grad_type::grad_param)
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

function propagator_linear_crushed_pulse(ω1, T, B1, R2s, dR2sdT2s, _, grad_type::grad_T2s)
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

function propagator_linear_crushed_pulse(ω1, T, B1, R2s, _, dR2sdB1, grad_type::grad_B1)
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
    sϕ, cϕ = sincos(rfphase_increment)
    u_rot = @SMatrix [cϕ -sϕ 0 0 0 0;
                      sϕ  cϕ 0 0 0 0;
                       0   0 1 0 0 0;
                       0   0 0 1 0 0;
                       0   0 0 0 1 0;
                       0   0 0 0 0 1]
    return u_rot
end

function z_rotation_propagator(rfphase_increment, ::grad_param)
    sϕ, cϕ = sincos(rfphase_increment)
    u_rot = @SMatrix [cϕ -sϕ 0 0 0  0   0 0 0 0 0;
                      sϕ  cϕ 0 0 0  0   0 0 0 0 0;
                      0    0 1 0 0  0   0 0 0 0 0;
                      0    0 0 1 0  0   0 0 0 0 0;
                      0    0 0 0 1  0   0 0 0 0 0;
                      0    0 0 0 0 cϕ -sϕ 0 0 0 0;
                      0    0 0 0 0 sϕ  cϕ 0 0 0 0;
                      0    0 0 0 0  0   0 1 0 0 0;
                      0    0 0 0 0  0   0 0 1 0 0;
                      0    0 0 0 0  0   0 0 0 1 0;
                      0    0 0 0 0  0   0 0 0 0 1]
    return u_rot
end

function xs_destructor(_::SMatrix{6, 6})
    Diagonal(SVector{6}(1,1,1,0,1,1))
end

function xs_destructor(_::SMatrix{11, 11})
    Diagonal(SVector{11}(1,1,1,0,1,1,1,1,0,1,1))
end

function xy_destructor(_::SMatrix{6, 6})
    Diagonal(SVector{6}(0,0,1,0,1,1))
end

function xy_destructor(_::SMatrix{11, 11})
    Diagonal(SVector{11}(0,0,1,0,1,0,0,1,0,1,1))
end

function A0(_::SMatrix{N, N}) where N
    Diagonal(SVector{N}(ntuple(_ -> 1, N-1)..., 0))
end

function C(_::SMatrix{N, N}) where N
    SVector(ntuple(_ -> 0, N-1)..., 1)
end


##############################################################################
# 3-pool (free water, restricted water, macromolecular) Hamiltonians
##############################################################################
"""
    hamiltonian_linear(ω1, B1, ω0, T, M0, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, R2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw)

Calculate the hamiltonian of the linear approximation of the 3-pool generalized Bloch model.

Returns a 9x9 (static) matrix with the dimensions (in this order)
`[xf_fw, yf_fw, zf_fw, xf_rw, yf_rw, zf_rw, xs_mm, zs_mm, 1]`

# Arguments
- `ω1::Real`: Rabi frequency in rad/s
- `B1::Real`: Normalized transmit B1 field
- `ω0::Real`: Off-resonance frequency in rad/s
- `T::Real`: Time in seconds
- `M0::Real`: Equilibrium magnetization scaling factor
- `m0_rw::Real`: Fractional size of the restricted water pool
- `m0_mm::Real`: Fractional size of the macromolecular pool
- `R1_fw::Real`: Longitudinal relaxation rate of free water in 1/s
- `R1_rw::Real`: Longitudinal relaxation rate of restricted water in 1/s
- `R1_mm::Real`: Longitudinal relaxation rate of macromolecular pool in 1/s
- `R2_fw::Real`: Transversal relaxation rate of free water in 1/s
- `R2_rw::Real`: Transversal relaxation rate of restricted water in 1/s
- `R2_mm::Real`: Transversal relaxation rate of macromolecular pool in 1/s
- `Rx_fw_mm::Real`: Exchange rate parameter between free water and macromolecular pools
- `Rx_rw_fw::Real`: Exchange rate parameter between restricted water and free water pools
- `Rx_mm_rw::Real`: Exchange rate parameter between macromolecular and restricted water pools
"""
function hamiltonian_linear(ω1, B1, ω0, T, M0, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, R2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw)
    m0_fw = 1 - m0_rw - m0_mm

    k_fw_mm = Rx_fw_mm * m0_mm * (m0_fw + m0_mm)
    k_mm_fw = Rx_fw_mm * m0_fw * (m0_mm + m0_fw)
    k_rw_fw = Rx_rw_fw * m0_fw * (m0_rw + m0_fw)
    k_fw_rw = Rx_rw_fw * m0_rw * (m0_fw + m0_rw)
    k_mm_rw = Rx_mm_rw * m0_rw * (m0_mm + m0_rw)
    k_rw_mm = Rx_mm_rw * m0_mm * (m0_rw + m0_mm)

    H = @SMatrix [
        -R2_fw-k_fw_rw        -ω0         B1*ω1      k_rw_fw              0          0           0         0       0;
         ω0          -R2_fw-k_fw_rw         0          0              k_rw_fw          0           0         0       0;
        -B1*ω1             0      -R1_fw-k_fw_mm-k_fw_rw  0               0        k_rw_fw         0      k_mm_fw  M0*R1_fw*m0_fw;
         k_fw_rw           0               0     -R2_rw-k_rw_fw         -ω0        B1*ω1          0         0       0;
         0             k_fw_rw             0          ω0         -R2_rw-k_rw_fw       0           0         0       0;
         0               0            k_fw_rw      -B1*ω1             0      -R1_rw-k_rw_fw-k_rw_mm  0      k_mm_rw  M0*R1_rw*m0_rw;
         0               0               0          0               0          0        -R2_mm    B1*ω1     0;
         0               0            k_fw_mm        0               0        k_rw_mm   -B1*ω1  -R1_mm-k_mm_fw-k_mm_rw  M0*R1_mm*m0_mm;
         0               0               0          0               0          0           0         0       0]

    return H * T
end

# Convenience method that ignores dR2 arguments (for free precession periods)
function hamiltonian_linear(ω1, B1, ω0, T, M0, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, R2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw, _, _, _)
    return hamiltonian_linear(ω1, B1, ω0, T, M0, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, R2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw)
end

# 3-pool z-rotation propagator (9x9, no gradient)
function z_rotation_propagator_3pool(rfphase_increment)
    sϕ, cϕ = sincos(rfphase_increment)
    u_rot = @SMatrix [
        cϕ -sϕ  0  0   0  0  0  0  0;
        sϕ  cϕ  0  0   0  0  0  0  0;
         0   0  1  0   0  0  0  0  0;
         0   0  0  cϕ -sϕ  0  0  0  0;
         0   0  0  sϕ  cϕ  0  0  0  0;
         0   0  0  0   0  1  0  0  0;
         0   0  0  0   0  0  1  0  0;
         0   0  0  0   0  0  0  1  0;
         0   0  0  0   0  0  0  0  1]
    return u_rot
end

# 3-pool z-rotation propagator (17x17, with gradient)
function z_rotation_propagator_3pool(rfphase_increment, ::grad_param)
    sϕ, cϕ = sincos(rfphase_increment)
    u_rot = @SMatrix [
        cϕ -sϕ  0  0   0  0  0  0  0  0   0  0  0  0  0  0  0;
        sϕ  cϕ  0  0   0  0  0  0  0  0   0  0  0  0  0  0  0;
         0   0  1  0   0  0  0  0  0  0   0  0  0  0  0  0  0;
         0   0  0  cϕ -sϕ  0  0  0  0  0   0  0  0  0  0  0  0;
         0   0  0  sϕ  cϕ  0  0  0  0  0   0  0  0  0  0  0  0;
         0   0  0  0   0  1  0  0  0  0   0  0  0  0  0  0  0;
         0   0  0  0   0  0  1  0  0  0   0  0  0  0  0  0  0;
         0   0  0  0   0  0  0  1  0  0   0  0  0  0  0  0  0;
         0   0  0  0   0  0  0  0  cϕ -sϕ  0  0   0  0  0  0  0;
         0   0  0  0   0  0  0  0  sϕ  cϕ  0  0   0  0  0  0  0;
         0   0  0  0   0  0  0  0  0   0  1  0   0  0  0  0  0;
         0   0  0  0   0  0  0  0  0   0  0  cϕ -sϕ  0  0  0  0;
         0   0  0  0   0  0  0  0  0   0  0  sϕ  cϕ  0  0  0  0;
         0   0  0  0   0  0  0  0  0   0  0  0   0  1  0  0  0;
         0   0  0  0   0  0  0  0  0   0  0  0   0  0  1  0  0;
         0   0  0  0   0  0  0  0  0   0  0  0   0  0  0  1  0;
         0   0  0  0   0  0  0  0  0   0  0  0   0  0  0  0  1]
    return u_rot
end

# 3-pool x_mm destructor (zeros out the xs_mm transverse component)
function x_mm_destructor(_::SMatrix{9, 9})
    Diagonal(SVector{9}(1,1,1,1,1,1,0,1,1))
end

function x_mm_destructor(_::SMatrix{17, 17})
    Diagonal(SVector{17}(1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1))
end

# 3-pool xy destructor (zeros out all transverse components)
function xy_destructor(_::SMatrix{9, 9})
    Diagonal(SVector{9}(0,0,1,0,0,1,0,1,1))
end

function xy_destructor(_::SMatrix{17, 17})
    Diagonal(SVector{17}(0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,1))
end

# 3-pool A0 and C for Matrix types (used in antiperiodic boundary conditions)
function A0(x::Matrix)
    N, M = size(x)
    @assert N == M
    Diagonal([i == N ? 0 : 1 for i = 1:N])
end

function C(x::Matrix)
    N, M = size(x)
    @assert N == M
    [i == N ? 1 : 0 for i = 1:N]
end