##
"""
    hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rex, R1s, R2s[, dR2sdT2s, dR2sdB1, grad_type])

Calculate the hamiltonian of the linear approximation of the generalized Bloch model.

If no gradient is supplied, it returns a 6x6 (static) matrix with the dimensions (in this order) `[xf, yf, zf, xs, zs, 1]`; the attached 1 is a mathematical trick to allow for ``T_1`` relaxation to a non-zero thermal equilibrium.
If a gradient is supplied, it returns a 11x11 (static) matrix with the dimensions (in this order) `[xf, yf, zf, xs, zs, dxf/dθ, dyf/dθ, dzf/dθ, dxs/dθ, dzs/dθ,  1]`, where `θ` is the parameter specified by `grad_type`

# Arguments
- `ω1::Real`: Rabi frequency in rad/s (rotation about the y-axis)
- `B1::Real`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `ω0::Real`: Larmor (or off-resonance) frequency in rad/s (rotation about the z-axis)
- `T::Real`: Time in seconds; this can, e.g., be the RF-pulse duration, or the time of free precession with `ω1=0`
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

julia> m0 = [0, 0, 1-m0s, 0, m0s, 1];

julia> (xf, yf, zf, xs, zs, _) = exp(hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rex, R1s, R2s)) * m0
6-element StaticArraysCore.SVector{6, Float64} with indices SOneTo(6):
  0.0010647535813058293
  0.0
 -0.8957848274535014
  0.005126529591877105
  0.08122007142111888
  1.0
```
"""

function hamiltonian_linear(ω1, B1, ω0, T, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, R2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw)
    m0_fw = 1 - m0_rw - m0_mm
    
    k_fw_mm = Rx_fw_mm * m0_mm * (m0_fw + m0_mm)
    k_mm_fw = Rx_fw_mm * m0_fw * (m0_mm + m0_fw)

    k_rw_fw = Rx_rw_fw * m0_fw * (m0_rw + m0_fw)
    k_fw_rw = Rx_rw_fw * m0_rw * (m0_fw + m0_rw)

    k_mm_rw = Rx_mm_rw * m0_rw * (m0_mm + m0_rw)
    k_rw_mm = Rx_mm_rw * m0_mm * (m0_rw + m0_mm)


H = @SMatrix [
-R2_fw - k_fw_rw                 -ω0                     B1 * ω1            k_rw_fw                 0                           0                  0                            0                  0;  
               ω0   -R2_fw - k_fw_rw                           0                  0           k_rw_fw                           0                  0                            0                  0;  
         -B1 * ω1                  0  -R1_fw - k_fw_mm - k_fw_rw                  0                 0                     k_rw_fw                  0                      k_mm_fw      R1_fw * m0_fw;  
          k_fw_rw                  0                           0   -R2_rw - k_rw_fw               -ω0                     B1 * ω1                  0                            0                  0;  
                0            k_fw_rw                           0                 ω0  -R2_rw - k_rw_fw                           0                  0                            0                  0;  
                0                  0                     k_fw_rw           -B1 * ω1                 0  -R1_rw - k_rw_fw - k_rw_mm                  0                      k_mm_rw      R1_rw * m0_rw;  
                0                  0                           0                  0                 0                           0             -R2_mm                      B1 * ω1                  0;  
                0                  0                     k_fw_mm                  0                 0                     k_rw_mm           -B1 * ω1   -R1_mm - k_mm_fw - k_mm_rw      R1_mm * m0_mm;  
                0                  0                           0                  0                 0                           0                  0                            0                  0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, R2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw, _, _, _)
    return hamiltonian_linear(ω1, B1, ω0, T, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, R2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw)
end

# function propagator_linear_crushed_pulse(ω1, T, B1, R2_mm, _, _, _)
#     H_mm = @SMatrix [ -R2_mm         B1 * ω1;
#                        -B1 * ω1            0]
#     U_mm = exp(H_mm * T)

# U = @SMatrix [
#         sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0;
#         0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0;
#         0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0;
#         0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0;
#         0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0;
#         0 0 0 0 0 cos(B1 * ω1 * T)        0 0 0;
#         0 0 0 0 0 0 U_mm[1,1]  U_mm[1,2]      0;
#         0 0 0 0 0 0 U_mm[2,1]  U_mm[2,2]      0;
#         0 0 0 0 0 0 0          0              1]
#     return U
# end

# function propagator_linear_crushed_pulse(ω1, T, B1, R2_mm, _, _, grad_type::grad_param)
#     H_mm = @SMatrix [-R2_mm         B1 * ω1;
#                      -B1 * ω1             0]
#     U_mm = exp(H_mm * T)

#     U = @SMatrix [
#         sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0 0                       0 0 0 0 0 0 0 0;
#         0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0 0                       0 0 0 0 0 0 0 0;
#         0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0 0                       0 0 0 0 0 0 0 0;
#         0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0                       0 0 0 0 0 0 0 0;
#         0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0                       0 0 0 0 0 0 0 0;
#         0 0 0 0 0 cos(B1 * ω1 * T)        0 0 0 0                       0 0 0 0 0 0 0 0;
#         0 0 0 0 0 0 U_mm[1,1]  U_mm[1,2]      0 0                       0 0 0 0 0 0 0 0;
#         0 0 0 0 0 0 U_mm[2,1]  U_mm[2,2]      0 0                       0 0 0 0 0 0 0 0;
#         0 0 0 0 0 0 0          0              0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0;
#         0 0 0 0 0 0 0          0              0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0;
#         0 0 0 0 0 0 0          0              0 0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0;
#         0 0 0 0 0 0 0          0              0 0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0;
#         0 0 0 0 0 0 0          0              0 0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0;
#         0 0 0 0 0 0 0          0              0 0 0 0 0 0 cos(B1 * ω1 * T)        0 0 0;
#         0 0 0 0 0 0 0          0              0 0 0 0 0 0 0 U_mm[1,1]  U_mm[1,2]      0;
#         0 0 0 0 0 0 0          0              0 0 0 0 0 0 0 U_mm[2,1]  U_mm[2,2]      0;
#         0 0 0 0 0 0 0          0              0 0 0 0 0 0 0 0 0 0          0          1]
#     return U
# end

# function propagator_linear_crushed_pulse(ω1, T, B1, R2s, dR2sdT2s, _, grad_type::grad_T2s)
#     Hs = @SMatrix [    -R2s  B1 * ω1;
#                    -B1 * ω1        0]
#     Us = exp(Hs * T)

#     dHsdT2s = @SMatrix [-dR2sdT2s   0;
#                                 0   0]

#     # Higham's Complex Step Approximation:
#     h = 1im * eps()
#     dU = real.(exp((Hs + h * dHsdT2s) * T) ./ h)

#     U = @SMatrix [
#         sin(B1 * ω1 * T / 2)^2    0 0 0 0 0 0 0 0 0 0;
#         0 -sin(B1 * ω1 * T / 2)^2   0 0 0 0 0 0 0 0 0;
#         0 0 cos(B1 * ω1 * T)          0 0 0 0 0 0 0 0;
#         0 0 0 Us[1,1]    Us[1,2]          0 0 0 0 0 0;
#         0 0 0 Us[2,1]    Us[2,2]          0 0 0 0 0 0;
#         0 0 0 0 0 sin(B1 * ω1 * T / 2)^2    0 0 0 0 0;
#         0 0 0 0 0 0 -sin(B1 * ω1 * T / 2)^2   0 0 0 0;
#         0 0 0 0 0 0 0 cos(B1 * ω1 * T)          0 0 0;
#         0 0 0 dU[1,1] dU[1,2] 0 0 0 Us[1,1] Us[1,2] 0;
#         0 0 0 dU[2,1] dU[2,2] 0 0 0 Us[2,1] Us[2,2] 0;
#         0 0 0 0       0       0 0 0 0       0       1]
#     return U
# end

# function propagator_linear_crushed_pulse(ω1, T, B1, R2s, _, dR2sdB1, grad_type::grad_B1)
#     Hs = @SMatrix [    -R2s  B1 * ω1;
#                    -B1 * ω1        0]
#     Us = exp(Hs * T)

#     dHsdB1 = @SMatrix [-dR2sdB1  ω1;
#                             -ω1   0]

#     # Higham's Complex Step Approximation:
#     h = 1im * eps()
#     dU = real.(exp((Hs + h * dHsdB1) * T) ./ h)

#     U = @SMatrix [
#         sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0 0 0;
#         0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0 0 0;
#         0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0 0 0;
#         0 0 0 Us[1,1]    Us[1,2]        0 0 0 0 0 0;
#         0 0 0 Us[2,1]    Us[2,2]        0 0 0 0 0 0;
#         sin(B1 * ω1 * T / 2) * cos(B1 * ω1 * T / 2) * ω1 * T  0 0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0;
#         0 -sin(B1 * ω1 * T / 2) * cos(B1 * ω1 * T / 2) * ω1 * T 0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0;
#         0 0 -sin(B1 * ω1 * T) * ω1 * T                            0 0 0 0 cos(B1 * ω1 * T)        0 0 0;
#         0 0 0 dU[1,1] dU[1,2] 0 0 0 Us[1,1] Us[1,2] 0;
#         0 0 0 dU[2,1] dU[2,2] 0 0 0 Us[2,1] Us[2,2] 0;
#         0 0 0 0       0       0 0 0 0       0       1]
#     return U
# end

function z_rotation_propagator(rfphase_increment, _)
    sϕ, cϕ = sincos(rfphase_increment)
    u_rot = @SMatrix [cϕ -sϕ 0  0   0 0 0 0 0;
                      sϕ  cϕ 0  0   0 0 0 0 0;
                       0   0 1  0   0 0 0 0 0;
                       0   0 0 cϕ -sϕ 0 0 0 0;
                       0   0 0 sϕ  cϕ 0 0 0 0;
                       0   0 0  0   0 1 0 0 0;
                       0   0 0  0   0 0 1 0 0;
                       0   0 0  0   0 0 0 1 0
                       0   0 0  0   0 0 0 0 1]
    return u_rot
end

function z_rotation_propagator(rfphase_increment, ::grad_param)
    sϕ, cϕ = sincos(rfphase_increment)
    u_rot = @SMatrix [cϕ -sϕ 0  0   0 0 0 0  0   0 0  0   0 0 0 0 0;
                      sϕ  cϕ 0  0   0 0 0 0  0   0 0  0   0 0 0 0 0;
                       0   0 1  0   0 0 0 0  0   0 0  0   0 0 0 0 0;
                       0   0 0 cϕ -sϕ 0 0 0  0   0 0  0   0 0 0 0 0;
                       0   0 0 sϕ  cϕ 0 0 0  0   0 0  0   0 0 0 0 0;
                       0   0 0  0   0 1 0 0  0   0 0  0   0 0 0 0 0;
                       0   0 0  0   0 0 1 0  0   0 0  0   0 0 0 0 0;
                       0   0 0  0   0 0 0 1  0   0 0  0   0 0 0 0 0;
                       0   0 0  0   0 0 0 0 cϕ -sϕ 0  0   0 0 0 0 0;
                       0   0 0  0   0 0 0 0 sϕ  cϕ 0  0   0 0 0 0 0;
                       0   0 0  0   0 0 0 0  0   0 1  0   0 0 0 0 0;
                       0   0 0  0   0 0 0 0  0   0 0 cϕ -sϕ 0 0 0 0;
                       0   0 0  0   0 0 0 0  0   0 0 sϕ  cϕ 0 0 0 0;
                       0   0 0  0   0 0 0 0  0   0 0  0   0 1 0 0 0;
                       0   0 0  0   0 0 0 0  0   0 0  0   0 0 1 0 0;
                       0   0 0  0   0 0 0 0  0   0 0  0   0 0 0 1 0
                       0   0 0  0   0 0 0 0  0   0 0  0   0 0 0 0 1]
    return u_rot
end


function x_mm_destructor(x)
    if size(x) == (9, 9)
        Diagonal([1,1,1,1,1,1,0,1,1])
    elseif size(x) == (17, 17)
        Diagonal([1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1])
    else
        error("`x_mm_destructor` is only implemented for sizes `(9, 9)`, `(17, 17)`")
    end
end


function xy_destructor(x)
    if size(x) == (9, 9)
        Diagonal([0,0,1,0,0,1,0,1,1])
    elseif size(x) == (17, 17)
        Diagonal([0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,1])
    else
        error("`xy_destructor` is only implemented for sizes `(9, 9)`, `(17, 17)`")
    end
end


# function A0(_::SMatrix{N, N}) where N
#     Diagonal(SVector{N}(ntuple(_ -> 1, N-1)..., 0))
# end

function A0(x::Matrix)
    N, M = size(x)
    @assert N == M
    Diagonal([i == N ? 0 : 1 for i = 1:N])
end

# function C(_::SMatrix{N, N}) where N
#     SVector(ntuple(_ -> 0, N-1)..., 1)
# end

function C(x::Matrix)
    N, M = size(x)
    @assert N == M
    [i == N ? 1 : 0 for i = 1:N]
end