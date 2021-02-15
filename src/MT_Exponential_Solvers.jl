##
function Linear_Hamiltonian_Matrix(ωy, B1, ωz, T, m0s, R1, R2f, Rx, Rrf, _, _, grad_type::Array)
    u = @SMatrix [
        -R2f * T     -ωz * T           B1 * ωy * T                                0.0                  0.0;
          ωz * T    -R2f * T                   0.0                                0.0                  0.0;
         -B1 * ωy * T    0.0  (-R1 - Rx * m0s) * T               Rx * (1.0 - m0s) * T R1 * (1.0 - m0s) * T;
           0.0           0.0          Rx * m0s * T (-R1 - Rrf - Rx * (1.0 - m0s)) * T         R1 * m0s * T;
           0.0           0.0                   0.0                                0.0                  0.0]
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
    u = @SMatrix [
        sin(B1 * ωy * T / 2)^2  0 0 0 0;
        0 -sin(B1 * ωy * T / 2)^2 0 0 0;
        0 0 cos(B1 * ωy * T)        0 0;
        0 0 0 exp(- Rrf * T)          0;
        0 0 0 0 1
    ];
    return u
end

function Inversion_Pulse_Propagator(ωy, T, B1, Rrf, _, _, grad_type::Any)
    u = @SMatrix [
        sin(B1 * ωy * T / 2)^2  0 0 0 0 0 0 0 0;
        0 -sin(B1 * ωy * T / 2)^2 0 0 0 0 0 0 0;
        0 0 cos(B1 * ωy * T)        0 0 0 0 0 0;
        0 0 0 exp(- Rrf * T)          0 0 0 0 0;
        0 0 0 0 sin(B1 * ωy * T / 2)^2  0 0 0 0;
        0 0 0 0 0 -sin(B1 * ωy * T / 2)^2 0 0 0;
        0 0 0 0 0 0 cos(B1 * ωy * T)        0 0;
        0 0 0 0 0 0 0 exp(- Rrf * T)          0;
        0 0 0 0 0 0 0 0 1
    ];
    return u
end

function Inversion_Pulse_Propagator(ωy, T, B1, Rrf, dRrfdT2s, _, grad_type::grad_T2s)
    u = @SMatrix [
        sin(B1 * ωy * T / 2)^2  0 0 0 0 0 0 0 0;
        0 -sin(B1 * ωy * T / 2)^2 0 0 0 0 0 0 0;
        0 0 cos(B1 * ωy * T)        0 0 0 0 0 0;
        0 0 0 exp(- Rrf * T)          0 0 0 0 0;
        0 0 0 0 sin(B1 * ωy * T / 2)^2  0 0 0 0;
        0 0 0 0 0 -sin(B1 * ωy * T / 2)^2 0 0 0;
        0 0 0 0 0 0 cos(B1 * ωy * T)        0 0;
        0 0 0 -dRrfdT2s * T * exp(-Rrf * T) 0 0 0 exp(- Rrf * T) 0;
        0 0 0 0 0 0 0 0 1
    ];
    return u
end

function Inversion_Pulse_Propagator(ωy, T, B1, Rrf, _, dRrfdB1, grad_type::grad_B1)
    u = @SMatrix [
        sin(B1 * ωy * T / 2)^2  0 0 0 0 0 0 0 0;
        0 -sin(B1 * ωy * T / 2)^2 0 0 0 0 0 0 0;
        0 0 cos(B1 * ωy * T)        0 0 0 0 0 0;
        0 0 0 exp(- Rrf * T)          0 0 0 0 0;
        sin(B1 * ωy * T / 2) * cos(B1 * ωy * T / 2) * ωy * T 0 0 0 sin(B1 * ωy * T / 2)^2  0 0 0 0;
        0 -sin(B1 * ωy * T / 2) * cos(B1 * ωy * T / 2) * ωy * T 0 0 0 -sin(B1 * ωy * T / 2)^2 0 0 0;
        0 0 -sin(B1 * ωy * T) * ωy * T 0 0 0 cos(B1 * ωy * T)        0 0;
        0 0 0 -dRrfdB1 * T * exp(-Rrf * T) 0 0 0 exp(- Rrf * T)          0;
        0 0 0 0 0 0 0 0 1
    ];
    return u
end

function MatrixApprox_calculate_magnetization!(M, ω1, TRF, TR, sweep_phase, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec)
    
    (_Rrf, _dRrfdT2s, _dRrfdB1) = Rrf_vec

    sweep_phase += π
    u_rot = @SMatrix [cos(sweep_phase) -sin(sweep_phase) 0 0 0;
                      sin(sweep_phase)  cos(sweep_phase) 0 0 0;
                                     0                0  1 0 0;
                                     0                0  0 1 0;
                                     0                0  0 0 1]
    
    y0 = Anti_periodic_boundary_conditions(ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _Rrf, _dRrfdT2s, _dRrfdB1, [], u_rot)

    Propagate_magnetization!(M, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _Rrf, _dRrfdT2s, _dRrfdB1, u_rot, y0, [])
        
    return nothing
end

function MatrixApprox_calculate_magnetization!(M, ω1, TRF, TR, sweep_phase, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, grad_list)
    
    (_Rrf, _dRrfdT2s, _dRrfdB1) = Rrf_vec

    sweep_phase += π
    
    u_rot = @SMatrix [cos(sweep_phase) -sin(sweep_phase) 0 0 0 0 0 0 0;
                      sin(sweep_phase)  cos(sweep_phase) 0 0 0 0 0 0 0;
                      0 0 1 0 0 0 0 0 0;
                      0 0 0 1 0 0 0 0 0;
                      0 0 0 0 cos(sweep_phase) -sin(sweep_phase) 0 0 0;
                      0 0 0 0 sin(sweep_phase)  cos(sweep_phase) 0 0 0;
                      0 0 0 0 0 0 1 0 0;
                      0 0 0 0 0 0 0 1 0;
                      0 0 0 0 0 0 0 0 1]

    nM = size(M, 1) ÷ (length(grad_list) + 1) # 1 component for signal, 4 components for magnetization (xf, yf, zf, zs)
    for i in eachindex(grad_list)
        y0 = Anti_periodic_boundary_conditions(ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _Rrf, _dRrfdT2s, _dRrfdB1, grad_list[i], u_rot)

        # pick first set of rows for the main magnetization and second one for the derivatives
        Mig = @view M[[1:nM;i * nM + 1:(i + 1) * nM],:] 
        Propagate_magnetization!(Mig, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _Rrf, _dRrfdT2s, _dRrfdB1, u_rot, y0, grad_list[i])
    end
    return nothing
end

function Calculate_Saturation_rate(ω1, TRF, B1, T2s, Rrf_T, grad_list)
    # calculate saturation rate
    _Rrf = similar(ω1)
    _dRrfdT2s = similar(ω1)
    _dRrfdB1 = similar(ω1)

    if any(isa.(grad_list, grad_T2s))
        for i = 1:length(ω1)
            _Rrf[i], _dRrfdB1[i], _dRrfdT2s[i] = Rrf_T[3](TRF[i], ω1[i], B1, T2s)
        end
    elseif any(isa.(grad_list, grad_B1))
        for i = 1:length(ω1)
            _Rrf[i], _dRrfdB1[i] = Rrf_T[2](TRF[i], ω1[i], B1, T2s)
        end
    else
        for i = 1:length(ω1)
            _Rrf[i] = Rrf_T[1](TRF[i], ω1[i], B1, T2s)
        end
    end
    return (_Rrf, _dRrfdT2s, _dRrfdB1)
end

function Anti_periodic_boundary_conditions(ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _Rrf, _dRrfdT2s, _dRrfdB1, grad, u_rot)
    
    # put inversion pulse at the end (this defines y0 as the magnetization at the first TE after the inversion pulse)
    u_fp = exp(Linear_Hamiltonian_Matrix(0.0, B1, ω0, TR / 2, m0s, R1, R2f, Rx, 0.0, 0.0, 0.0, grad))
    u_pl = Inversion_Pulse_Propagator(ω1[1], TRF[1], B1, _Rrf[1], _dRrfdT2s[1], _dRrfdB1[1], grad)
    A = u_fp * u_pl * u_rot * u_fp
    
    for i = length(ω1):-1:2
        u_fp = exp(Linear_Hamiltonian_Matrix(0.0, B1, ω0, (TR - TRF[i]) / 2, m0s, R1, R2f, Rx, 0.0, 0.0, 0.0, grad))
        u_pl = exp(Linear_Hamiltonian_Matrix(ω1[i], B1, ω0, TRF[i], m0s, R1, R2f, Rx, _Rrf[i], _dRrfdT2s[i], _dRrfdB1[i], grad))
        A = A * u_fp * u_pl * u_rot * u_fp
    end

    F = eigen(A)
    y0 = F.vectors[:,end]
    y0 /= y0[end]
    return y0
end

function Propagate_magnetization!(M::AbstractArray{T,2}, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _Rrf, _dRrfdT2s, _dRrfdB1, u_rot, y0, grad) where T <: Real
    M[:,1] = @view y0[1:end - 1]
    for i = 2:length(ω1)
        u_fp = exp(Linear_Hamiltonian_Matrix(0.0, B1, ω0, (TR - TRF[i]) / 2, m0s, R1, R2f, Rx, 0.0, 0.0, 0.0, grad))
        u_pl = exp(Linear_Hamiltonian_Matrix(ω1[i], B1, ω0, TRF[i], m0s, R1, R2f, Rx, _Rrf[i], _dRrfdT2s[i], _dRrfdB1[i], grad))

        y0 = u_fp * (u_pl * (u_rot * (u_fp * y0)))
        M[:,i] = @view y0[1:end - 1]
    end
    return nothing
end

function Propagate_magnetization!(S::AbstractArray{T,1}, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _Rrf, _dRrfdT2s, _dRrfdB1, u_rot, y0, grad) where T <: Complex
    S[1] = y0[1] + 1im * y0[2]
    for i = 2:length(ω1)
        u_fp = exp(Linear_Hamiltonian_Matrix(0.0, B1, ω0, (TR - TRF[i]) / 2, m0s, R1, R2f, Rx, 0.0, 0.0, 0.0, grad))
        u_pl = exp(Linear_Hamiltonian_Matrix(ω1[i], B1, ω0, TRF[i], m0s, R1, R2f, Rx, _Rrf[i], _dRrfdT2s[i], _dRrfdB1[i], grad))

        y0 = u_fp * (u_pl * (u_rot * (u_fp * y0)))
        S[i] = y0[1] + 1im * y0[2]
    end
    return nothing
end

function Propagate_magnetization!(S::AbstractArray{T,2}, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _Rrf, _dRrfdT2s, _dRrfdB1, u_rot, y0, grad) where T <: Complex
    S[1,1] = y0[1] + 1im * y0[2]
    S[2,1] = y0[5] + 1im * y0[6]
    for i = 2:length(ω1)
        u_fp = exp(Linear_Hamiltonian_Matrix(0.0, B1, ω0, (TR - TRF[i]) / 2, m0s, R1, R2f, Rx, 0.0, 0.0, 0.0, grad))
        u_pl = exp(Linear_Hamiltonian_Matrix(ω1[i], B1, ω0, TRF[i], m0s, R1, R2f, Rx, _Rrf[i], _dRrfdT2s[i], _dRrfdB1[i], grad))

        y0 = u_fp * (u_pl * (u_rot * (u_fp * y0)))
        S[1,i] = y0[1] + 1im * y0[2]
        S[2,i] = y0[5] + 1im * y0[6]
    end
    return nothing
end

####### returns all components of the magnetization #######
# Version w/o gradients
function MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s::Number, Rrf_T, sweep_phase::Vector{T}=[0]) where {T <: Number}
    Rrf_vec = Calculate_Saturation_rate(ω1, TRF, B1, T2s, Rrf_T, [[]])
    return MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, sweep_phase)
end

function MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, sweep_phase::Vector{T}=[0]) where {T <: Number}
    M = similar(ω1, 4, length(sweep_phase) * length(ω1))

    for i in eachindex(sweep_phase)
        Mi = @view M[:, (i - 1) * length(ω1) + 1:i * length(ω1)]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, Rrf_vec)
    end
    return M
end

# Version with gradients
function MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s::Number, Rrf_T, grad_list::Array{T,1}, sweep_phase=[0]) where T <: grad_param
    Rrf_vec = Calculate_Saturation_rate(ω1, TRF, B1, T2s, Rrf_T, grad_list)
    return MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, grad_list, sweep_phase)
end

function MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, grad_list::Array{T,1}, sweep_phase=[0]) where {T <: grad_param}
    M = similar(ω1, 4 + 4 * length(grad_list), length(sweep_phase) * length(ω1))

    for i in eachindex(sweep_phase)
        Mi = @view M[:, (i - 1) * length(ω1) + 1:i * length(ω1)]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, grad_list)
    end
    return M
end

####### returns the complex valued signal #######
# Version w/o gradients
function MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s::Number, Rrf_T, sweep_phase::Vector{T}=[0]) where {T <: Number}
    Rrf_vec = Calculate_Saturation_rate(ω1, TRF, B1, T2s, Rrf_T, [[]])
    return MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, sweep_phase)
end

function MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, sweep_phase::Vector{T}=[0]) where {T <: Number}
    M = similar(ω1, ComplexF64, length(sweep_phase) * length(ω1))
    for i in eachindex(sweep_phase)
        Mi = @view M[(i - 1) * length(ω1) + 1:i * length(ω1)]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, Rrf_vec)
    end
    return M
end

# Version with gradients
function MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s::Number, Rrf_T, grad_list::Array{T,1}, sweep_phase=[0]) where T <: grad_param
    Rrf_vec = Calculate_Saturation_rate(ω1, TRF, B1, T2s, Rrf_T, grad_list)
    return MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, grad_list, sweep_phase)
end

function MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, grad_list::Array{T,1}, sweep_phase=[0]) where T <: grad_param
    M = similar(ω1, ComplexF64, 1 + length(grad_list), length(sweep_phase) * length(ω1))
    for i in eachindex(sweep_phase)
        Mi = @view M[:, (i - 1) * length(ω1) + 1:i * length(ω1)]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, Rrf_vec, grad_list)
    end
    return M
end