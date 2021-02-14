##
function Linear_Hamiltonian_Matrix(ωy, B1, ωz, T, m0s, R1, R2f, Rx, Rrf, _, _, grad_type::Array)
    u = @SMatrix [
        -R2f * T  -ωz * T           B1 * ωy * T                      0.0            0.0;
          ωz * T -R2f * T            0.0                      0.0            0.0;
         -B1 * ωy * T    0.0 (-R1 - Rx * m0s) * T           Rx * (1.0 - m0s) * T R1 * (1.0 - m0s) * T;
           0.0    0.0       Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T       R1 * m0s * T;
           0.0    0.0            0.0                      0.0            0.0]
end

function Linear_Hamiltonian_Matrix(ωy, B1, ωz, T, m0s, R1, R2f, Rx, Rrf, _, _, grad_type::grad_m0s)
    u = @SMatrix [
        -R2f * T  -ωz * T           B1 * ωy * T                      0.0      0.0    0.0            0.0                      0.0                 0.0;
          ωz * T -R2f * T            0.0                      0.0      0.0    0.0            0.0                      0.0                 0.0;
         -B1 * ωy * T    0.0 (-R1 - Rx * m0s) * T           Rx * (1.0 - m0s) * T      0.0    0.0            0.0                      0.0      R1 * (1.0 - m0s) * T;
           0.0    0.0       Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T      0.0    0.0            0.0                      0.0            R1 * m0s * T;
           0.0    0.0            0.0                      0.0   -R2f * T  -ωz * T           B1 * ωy * T                      0.0                 0.0;
           0.0    0.0            0.0                      0.0     ωz * T -R2f * T            0.0                      0.0                 0.0;
           0.0    0.0          -Rx * T                    -Rx * T    -B1 * ωy * T    0.0 (-R1 - Rx * m0s) * T           Rx * (1.0 - m0s) * T               -R1 * T;
           0.0    0.0           Rx * T                     Rx * T      0.0    0.0       Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T                R1 * T;
           0.0    0.0            0.0                      0.0      0.0    0.0            0.0                      0.0                 0.0]
end

function Linear_Hamiltonian_Matrix(ωy, B1, ωz, T, m0s, R1, R2f, Rx, Rrf, _, _, grad_type::grad_R1)
    u = @SMatrix [
        -R2f * T  -ωz * T               B1 * ωy * T                                0.0        0.0      0.0                  0.0                                0.0                       0.0;
          ωz * T -R2f * T                  0.0                                0.0        0.0      0.0                  0.0                                0.0                       0.0;
         -B1 * ωy * T      0.0 (-R1 - Rx * m0s) * T               Rx * (1.0 - m0s) * T        0.0      0.0                  0.0                                0.0      R1 * (1.0 - m0s) * T;
             0.0      0.0         Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T        0.0      0.0                  0.0                                0.0              R1 * m0s * T;
             0.0      0.0                  0.0                                0.0   -R2f * T  -ωz * T               B1 * ωy * T                                0.0                       0.0;
             0.0      0.0                  0.0                                0.0     ωz * T -R2f * T                  0.0                                0.0                       0.0;
             0.0      0.0                   -T                                0.0    -B1 * ωy * T      0.0 (-R1 - Rx * m0s) * T               Rx * (1.0 - m0s) * T           (1.0 - m0s) * T;
             0.0      0.0                  0.0                                 -T        0.0      0.0         Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T                   m0s * T;
             0.0      0.0                  0.0                                0.0        0.0      0.0                  0.0                                0.0                       0.0]
end

function Linear_Hamiltonian_Matrix(ωy, B1, ωz, T, m0s, R1, R2f, Rx, Rrf, _, _, grad_type::grad_R2f)
    u = @SMatrix [
        -R2f * T  -ωz * T           B1 * ωy * T                      0.0      0.0    0.0            0.0                      0.0                 0.0;
          ωz * T -R2f * T            0.0                      0.0      0.0    0.0            0.0                      0.0                 0.0;
         -B1 * ωy * T    0.0 (-R1 - Rx * m0s) * T           Rx * (1.0 - m0s) * T      0.0    0.0            0.0                      0.0      R1 * (1.0 - m0s) * T;
           0.0    0.0       Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T      0.0    0.0            0.0                      0.0            R1 * m0s * T;
            -T    0.0            0.0                      0.0   -R2f * T  -ωz * T           B1 * ωy * T                      0.0                 0.0;
           0.0     -T            0.0                      0.0     ωz * T -R2f * T            0.0                      0.0                 0.0;
           0.0    0.0            0.0                      0.0    -B1 * ωy * T    0.0 (-R1 - Rx * m0s) * T           Rx * (1.0 - m0s) * T                 0.0;
           0.0    0.0            0.0                      0.0      0.0    0.0       Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T                 0.0;
           0.0    0.0            0.0                      0.0      0.0    0.0            0.0                      0.0                 0.0]
end

function Linear_Hamiltonian_Matrix(ωy, B1, ωz, T, m0s, R1, R2f, Rx, Rrf, _, _, grad_type::grad_Rx)
    u = @SMatrix [
        -R2f * T  -ωz * T           B1 * ωy * T                      0.0      0.0    0.0            0.0                      0.0                 0.0;
          ωz * T -R2f * T            0.0                      0.0      0.0    0.0            0.0                      0.0                 0.0;
         -B1 * ωy * T    0.0 (-R1 - Rx * m0s) * T           Rx * (1.0 - m0s) * T      0.0    0.0            0.0                      0.0      R1 * (1.0 - m0s) * T;
           0.0    0.0       Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T      0.0    0.0            0.0                      0.0            R1 * m0s * T;
           0.0    0.0            0.0                      0.0   -R2f * T  -ωz * T           B1 * ωy * T                      0.0                 0.0;
           0.0    0.0            0.0                      0.0     ωz * T -R2f * T            0.0                      0.0                 0.0;
           0.0    0.0         -m0s * T              (1.0 - m0s) * T    -B1 * ωy * T    0.0 (-R1 - Rx * m0s) * T           Rx * (1.0 - m0s) * T                 0.0;
           0.0    0.0          m0s * T             -(1.0 - m0s) * T      0.0    0.0       Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T                 0.0;
           0.0    0.0            0.0                      0.0      0.0    0.0            0.0                      0.0                 0.0]
end

function Linear_Hamiltonian_Matrix(ωy, B1, ωz, T, m0s, R1, R2f, Rx, Rrf, dRrfdT2s, _, grad_type::grad_T2s)
    u = @SMatrix [
            -R2f * T  -ωz * T          B1 * ωy * T                                0.0          0.0      0.0                  0.0                                0.0                  0.0;
              ωz * T -R2f * T                  0.0                                0.0          0.0      0.0                  0.0                                0.0                  0.0;
        -B1 * ωy * T      0.0 (-R1 - Rx * m0s) * T               Rx * (1.0 - m0s) * T          0.0      0.0                  0.0                                0.0 R1 * (1.0 - m0s) * T;
                 0.0      0.0         Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T          0.0      0.0                  0.0                                0.0         R1 * m0s * T;
                 0.0      0.0                  0.0                                0.0     -R2f * T  -ωz * T          B1 * ωy * T                                0.0                  0.0;
                 0.0      0.0                  0.0                                0.0       ωz * T -R2f * T                  0.0                                0.0                  0.0;
                 0.0      0.0                  0.0                                0.0 -B1 * ωy * T      0.0 (-R1 - Rx * m0s) * T               Rx * (1.0 - m0s) * T                  0.0;
                 0.0      0.0                  0.0                      -dRrfdT2s * T          0.0      0.0         Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T                  0.0;
                 0.0      0.0                  0.0                                0.0          0.0      0.0                  0.0                                0.0                  0.0]
end

function Linear_Hamiltonian_Matrix(ωy, B1, ωz, T, m0s, R1, R2f, Rx, Rrf, _, _, grad_type::grad_ω0)
    u = @SMatrix [
        -R2f * T  -ωz * T           B1 * ωy * T                      0.0      0.0    0.0            0.0                      0.0                 0.0;
          ωz * T -R2f * T            0.0                      0.0      0.0    0.0            0.0                      0.0                 0.0;
        -B1 * ωy * T    0.0 (-R1 - Rx * m0s) * T           Rx * (1.0 - m0s) * T      0.0    0.0            0.0                      0.0      R1 * (1.0 - m0s) * T;
           0.0    0.0       Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T      0.0    0.0            0.0                      0.0            R1 * m0s * T;
           0.0     -T            0.0                      0.0   -R2f * T  -ωz * T           B1 * ωy * T                      0.0                 0.0;
             T    0.0            0.0                      0.0     ωz * T -R2f * T            0.0                      0.0                 0.0;
           0.0    0.0            0.0                      0.0    -B1 * ωy * T    0.0 (-R1 - Rx * m0s) * T           Rx * (1.0 - m0s) * T                 0.0;
           0.0    0.0            0.0                      0.0      0.0    0.0       Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T                 0.0;
           0.0    0.0            0.0                      0.0      0.0    0.0            0.0                      0.0                 0.0]
end

function Linear_Hamiltonian_Matrix(ωy, B1, ωz, T, m0s, R1, R2f, Rx, Rrf, _, dRrfdB1, grad_type::grad_B1)
    u = @SMatrix [
            -R2f * T  -ωz * T          B1 * ωy * T                                0.0          0.0      0.0                  0.0                                0.0                  0.0;
              ωz * T -R2f * T                  0.0                                0.0          0.0      0.0                  0.0                                0.0                  0.0;
        -B1 * ωy * T      0.0 (-R1 - Rx * m0s) * T               Rx * (1.0 - m0s) * T          0.0      0.0                  0.0                                0.0 R1 * (1.0 - m0s) * T;
                 0.0      0.0         Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T          0.0      0.0                  0.0                                0.0         R1 * m0s * T;
                 0.0      0.0               ωy * T                                0.0     -R2f * T  -ωz * T          B1 * ωy * T                                0.0                  0.0;
                 0.0      0.0                  0.0                                0.0       ωz * T -R2f * T                  0.0                                0.0                  0.0;
             -ωy * T      0.0                  0.0                                0.0 -B1 * ωy * T      0.0 (-R1 - Rx * m0s) * T               Rx * (1.0 - m0s) * T                  0.0;
                 0.0      0.0                  0.0                       -dRrfdB1 * T          0.0      0.0         Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T                  0.0;
                 0.0      0.0                  0.0                                0.0          0.0      0.0                  0.0                                0.0                  0.0]
end

function Inversion_Pulse_Propagator(ωy, T, B1, Rrf, _, _, grad_type::Array)
    u = zeros(Float64, 5, 5)
    u[1,1] = sin(B1 * ωy * T / 2)^2
    u[2,2] = -sin(B1 * ωy * T / 2)^2
    u[3,3] = cos(B1 * ωy * T)
    u[4,4] = real(exp(- Rrf * T))
    u[end,end] = 1.0
    return u
end

function Inversion_Pulse_Propagator(ωy, T, B1, Rrf, _, _, grad_type::Any)
    u = zeros(Float64, 9, 9)
    u[1,1] = sin(B1 * ωy * T / 2)^2
    u[2,2] = -sin(B1 * ωy * T / 2)^2
    u[3,3] = cos(B1 * ωy * T)
    u[4,4] = real(exp(- Rrf * T))
    u[5,5] = u[1,1]
    u[6,6] = u[2,2]
    u[7,7] = u[3,3]
    u[8,8] = u[4,4]
    u[end,end] = 1.0
    return u
end

function Inversion_Pulse_Propagator(ωy, T, B1, Rrf, dRrfdT2s, _, grad_type::grad_T2s)
    u = Inversion_Pulse_Propagator(ωy, T, B1, Rrf, 0, 0, 0)
    u[8,4] = - dRrfdT2s * T * real(exp(-Rrf * T))
    return u
end

function Inversion_Pulse_Propagator(ωy, T, B1, Rrf, _, dRrfdB1, grad_type::grad_B1)
    u = Inversion_Pulse_Propagator(ωy, T, B1, Rrf, 0, 0, 0)
    
    u[5,1] =  sin(B1 * ωy * T / 2) * cos(B1 * ωy * T / 2) * ωy * T
    u[6,2] = -sin(B1 * ωy * T / 2) * cos(B1 * ωy * T / 2) * ωy * T
    u[7,3] = -sin(B1 * ωy * T) * ωy * T
    u[8,4] = - dRrfdB1 * T * real(exp(-Rrf * T))
    return u
end

function MatrixApprox_calculate_magnetization!(M, ω1, TRF, TR, sweep_phase, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, grad_list)
    
    (_Rrf, _dRrfdT2s, _dRrfdB1) = Rrf_vec

    # initialization and memory allocation
    u_fp = Vector{Matrix{Float64}}(undef, length(ω1))
    u_pl = similar(u_fp)

    sweep_phase += π
    if isa(grad_list[1], Array)
        u_rot = @SMatrix [cos(sweep_phase) -sin(sweep_phase) 0 0 0;
                          sin(sweep_phase)  cos(sweep_phase) 0 0 0;
                                0         0  1 0 0;
                                0         0  0 1 0;
                                0         0  0 0 1]
    else
        u_rot = @SMatrix [cos(sweep_phase) -sin(sweep_phase) 0 0 0 0 0 0 0;
                          sin(sweep_phase)  cos(sweep_phase) 0 0 0 0 0 0 0;
                                0         0  1 0 0 0 0 0 0;
                                0         0  0 1 0 0 0 0 0;
                                0         0  0 0 cos(sweep_phase) -sin(sweep_phase) 0 0 0;
                                0         0  0 0 sin(sweep_phase)  cos(sweep_phase) 0 0 0;
                                0         0  0 0 0               0  1 0 0;
                                0         0  0 0 0               0  0 1 0;
                                0         0  0 0 0               0  0 0 1]
    end

    nM = size(M, 1) ÷ (length(grad_list) + 1) # 1 component for signal, 4 components for magnetization (xf, yf, zf, zs)
    for ig in eachindex(grad_list)
        Calculate_Propagators!(u_pl, u_fp, ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, _Rrf, _dRrfdT2s, _dRrfdB1, grad_list[ig])

        y0 = Anti_periodic_boundary_conditions(u_pl, u_fp, u_rot)

        # pick first set of rows for the main magnetization and second one for the derivatives
        if length(grad_list) == 1
            Propagate_magnetization!(M, u_pl, u_fp, u_rot, y0)
        else
            Mig = @view M[[1:nM;ig * nM + 1:(ig + 1) * nM],:] 
            Propagate_magnetization!(Mig, u_pl, u_fp, u_rot, y0)
        end
    end
    return nothing
end

function Calculate_Saturation_rate(TRF, ω1, B1, T2s, Rrf_T, grad_list)
    # calculate saturation rate
    _Rrf = similar(ω1, Complex)
    _dRrfdT2s = similar(ω1)
    _dRrfdB1 = similar(ω1)

    if any(isa.(grad_list, grad_T2s))
        for ip = 1:length(ω1)
            _Rrf[ip], _dRrfdB1[ip], _dRrfdT2s[ip] = Rrf_T[3](TRF[ip], ω1[ip], B1, T2s)
        end
    elseif any(isa.(grad_list, grad_B1))
        for ip = 1:length(ω1)
            _Rrf[ip], _dRrfdB1[ip] = Rrf_T[2](TRF[ip], ω1[ip], B1, T2s)
        end
    else
        for ip = 1:length(ω1)
            _Rrf[ip] = Rrf_T[1](TRF[ip], ω1[ip], B1, T2s)
        end
    end
    return (_Rrf, _dRrfdT2s, _dRrfdB1)
end

function Calculate_Propagators!(u_pl, u_fp, ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, _Rrf, _dRrfdT2s, _dRrfdB1, grad)
    u_pl[1] = Inversion_Pulse_Propagator(ω1[1], TRF[1], B1, _Rrf[1], _dRrfdT2s[1], _dRrfdB1[1], grad)
    u_fp[1] = exp(Linear_Hamiltonian_Matrix(0.0, B1, ω0, TR / 2, m0s, R1, R2f, Rx, 0.0, 0.0, 0.0, grad))

    for ip = 2:length(u_pl)
        if abs(imag(_Rrf[ip])) > eps()
            H = Linear_Hamiltonian_Matrix(ω1[ip], B1, ω0, TRF[ip], m0s, R1, R2f, Rx, _Rrf[ip], _dRrfdT2s[ip], _dRrfdB1[ip], grad)
        else
            H = Linear_Hamiltonian_Matrix(ω1[ip], B1, ω0, TRF[ip], m0s, R1, R2f, Rx, real(_Rrf[ip]), _dRrfdT2s[ip], _dRrfdB1[ip], grad)
        end
        u_pl[ip] = real.(exp(H))
        H = Linear_Hamiltonian_Matrix(0.0, B1, ω0, (TR - TRF[ip]) / 2, m0s, R1, R2f, Rx, 0.0, 0.0, 0.0, grad)
        u_fp[ip] = exp(H)
    end
    return nothing
end

function Anti_periodic_boundary_conditions(u_pl, u_fp, u_rot)
    A = u_fp[1] * u_pl[1] * u_rot * u_fp[1]
    for i = 2:length(u_pl)
        A = u_fp[i] * u_pl[i] * u_rot * u_fp[i] * A
    end

    F = eigen(A)
    y0 = F.vectors[:,end]
    y0 /= y0[end]
    return y0
end

function Propagate_magnetization!(M::AbstractArray{T,2}, u_pl, u_fp, u_rot, y0) where T <: Real
    for i = 1:length(u_pl)
        y0 = u_fp[i] * y0
        y0 = u_rot   * y0
        y0 = u_pl[i] * y0
        y0 = u_fp[i] * y0
        M[:,i] = y0[1:end - 1]
    end
    return M
end

function Propagate_magnetization!(S::AbstractArray{T,1}, u_pl, u_fp, u_rot, y0) where T <: Complex
    for i = 1:length(u_pl)
        y0 = u_fp[i] * y0
        y0 = u_rot   * y0
        y0 = u_pl[i] * y0
        y0 = u_fp[i] * y0
        S[i] = y0[1] + 1im * y0[2]
    end
    return nothing
end

function Propagate_magnetization!(S::AbstractArray{T,2}, u_pl, u_fp, u_rot, y0) where T <: Complex
    for i = 1:length(u_pl)
        y0 = u_fp[i] * y0
        y0 = u_rot   * y0
        y0 = u_pl[i] * y0
        y0 = u_fp[i] * y0
        S[1,i] = y0[1] + 1im * y0[2]
        S[2,i] = y0[5] + 1im * y0[6]
    end
    return nothing
end

# Version w/o gradients
function MatrixApprox_calculate_magnetization(ω1, TRF, TR::Number, sweep_phase::Number, ω0::Number, B1::Number, m0s::Number, R1::Number, R2f::Number, Rx::Number, Rrf_vec::NTuple{3, AbstractArray})
    M = similar(ω1, 4, length(ω1))
    MatrixApprox_calculate_magnetization!(M, ω1, TRF, TR, sweep_phase, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, [[]])
    return M
end
function MatrixApprox_calculate_magnetization(ω1, TRF, TR::Number, sweep_phase::AbstractArray{T,1}, ω0::Number, B1::Number, m0s::Number, R1::Number, R2f::Number, Rx::Number, Rrf_vec::NTuple{3, AbstractArray}) where T <: Number
    M = similar(ω1, 4, length(sweep_phase)*length(ω1))
    for i in eachindex(sweep_phase)
        Mi = @view M[:, (i-1)*length(ω1)+1 : i*length(ω1)]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, [[]])
    end
    return M
end
function MatrixApprox_calculate_magnetization(ω1, TRF, TR::Number, ω0::Number, B1::Number, m0s::Number, R1::Number, R2f::Number, Rx::Number, Rrf_vec::NTuple{3, AbstractArray})
    return MatrixApprox_calculate_magnetization(ω1, TRF, TR, 0, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec)
end


# Version with gradients
function MatrixApprox_calculate_magnetization(ω1, TRF, TR::Number, sweep_phase::Number, ω0::Number, B1::Number, m0s::Number, R1::Number, R2f::Number, Rx::Number, Rrf_vec::NTuple{3, AbstractArray}, grad_list::Array{grad_param,1})
    M = similar(ω1, 4 + 4 * length(grad_list), length(ω1))
    MatrixApprox_calculate_magnetization!(M, ω1, TRF, TR, sweep_phase, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, grad_list)
    return M
end
function MatrixApprox_calculate_magnetization(ω1, TRF, TR::Number, sweep_phase::AbstractArray{T,1}, ω0::Number, B1::Number, m0s::Number, R1::Number, R2f::Number, Rx::Number, T2s::Number, Rrf_vec::NTuple{3, AbstractArray}, grad_list::Array{grad_param,1}) where T <: Number
    M = similar(ω1, 4 + 4 * length(grad_list), length(sweep_phase)*length(ω1))
    for i in eachindex(sweep_phase)
        Mi = @view M[:, (i-1)*length(ω1)+1 : i*length(ω1)]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, grad_list)
    end
    return M
end
function MatrixApprox_calculate_magnetization(ω1, TRF, TR::Number, ω0::Number, B1::Number, m0s::Number, R1::Number, R2f::Number, Rx::Number, Rrf_vec::NTuple{3, AbstractArray}, grad_list::Array{grad_param,1})
    return MatrixApprox_calculate_magnetization(ω1, TRF, TR, 0, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, grad_list)
end

####### returns the complex valued signal #######
# Version w/o gradients
function MatrixApprox_calculate_signal(ω1, TRF, TR::Number, sweep_phase::Number, ω0::Number, B1::Number, m0s::Number, R1::Number, R2f::Number, Rx::Number, Rrf_vec::NTuple{3, AbstractArray})
    M = similar(ω1, ComplexF64)
    MatrixApprox_calculate_magnetization!(M, ω1, TRF, TR, sweep_phase, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, [[]])
    return M
end
function MatrixApprox_calculate_signal(ω1, TRF, TR::Number, sweep_phase::AbstractArray{T,1}, ω0::Number, B1::Number, m0s::Number, R1::Number, R2f::Number, Rx::Number, Rrf_vec::NTuple{3, AbstractArray}) where T <: Number
    M = similar(ω1, ComplexF64, length(sweep_phase)*length(ω1))
    for i in eachindex(sweep_phase)
        Mi = @view M[(i-1)*length(ω1)+1 : i*length(ω1)]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, [[]])
    end
    return M
end
function MatrixApprox_calculate_signal(ω1, TRF, TR::Number, ω0::Number, B1::Number, m0s::Number, R1::Number, R2f::Number, Rx::Number, Rrf_vec::NTuple{3, AbstractArray})
    return MatrixApprox_calculate_signal(ω1, TRF, TR, 0, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec)
end

# Version with gradients
function MatrixApprox_calculate_signal(ω1, TRF, TR::Number, sweep_phase::Number, ω0::Number, B1::Number, m0s::Number, R1::Number, R2f::Number, Rx::Number, Rrf_vec::NTuple{3, AbstractArray}, grad_list::Array{grad_param,1})
    M = similar(ω1, ComplexF64, 1 + length(grad_list), length(ω1))
    MatrixApprox_calculate_magnetization!(M, ω1, TRF, TR, sweep_phase, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, grad_list)
    return M
end
function MatrixApprox_calculate_signal(ω1, TRF, TR::Number, sweep_phase::AbstractArray{T,1}, ω0::Number, B1::Number, m0s::Number, R1::Number, R2f::Number, Rx::Number, Rrf_vec::NTuple{3, AbstractArray}, grad_list::Array{grad_param,1}) where T <: Number
    M = similar(ω1, ComplexF64, 1 + length(grad_list), length(sweep_phase)*length(ω1))
    for i in eachindex(sweep_phase)
        Mi = @view M[:, (i-1)*length(ω1)+1 : i*length(ω1)]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, grad_list)
    end
    return M
end
function MatrixApprox_calculate_signal(ω1, TRF, TR::Number, ω0::Number, B1::Number, m0s::Number, R1::Number, R2f::Number, Rx::Number, Rrf_vec::NTuple{3, AbstractArray}, grad_list::Array{grad_param,1})
    return MatrixApprox_calculate_signal(ω1, TRF, TR, 0, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, grad_list)
end