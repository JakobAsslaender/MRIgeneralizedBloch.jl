#########################################################
# calcuate real valued magnetization components
#########################################################
# Version w/o gradients
function MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s::Number, R2s_T, sweep_phase::Vector{T}=[0]) where {T <: Number}
    R2s_vec = Calculate_Saturation_rate(ω1, TRF, B1, T2s, R2s_T, [undef])
    return MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, sweep_phase)
end

function MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, sweep_phase::Vector{T}=[0]) where {T <: Number}
    M = similar(ω1, 5, length(sweep_phase) * length(ω1))

    for i in eachindex(sweep_phase)
        Mi = @view M[:, (i - 1) * length(ω1) + 1:i * length(ω1)]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, R2s_vec)
    end
    return M
end

# Version with gradients
function MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s::Number, R2s_T, grad_list::Array{T,1}, sweep_phase=[0]) where T <: grad_param
    R2s_vec = Calculate_Saturation_rate(ω1, TRF, B1, T2s, R2s_T, grad_list)
    return MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, grad_list, sweep_phase)
end

function MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, grad_list::Array{T,1}, sweep_phase=[0]) where {T <: grad_param}
    M = similar(ω1, 5 * (1 + length(grad_list)), length(sweep_phase) * length(ω1))

    for i in eachindex(sweep_phase)
        Mi = @view M[:, (i - 1) * length(ω1) + 1:i * length(ω1)]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, R2s_vec, grad_list)
    end
    return M
end

#########################################################
# calcuate complex valued signal (Mxf + 1im * Myf)
#########################################################
# Version w/o gradients
function MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s::Number, R2s_T, sweep_phase::Vector{T}=[0]) where {T <: Number}
    R2s_vec = Calculate_Saturation_rate(ω1, TRF, B1, T2s, R2s_T, [undef])
    return MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, sweep_phase)
end

function MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, sweep_phase::Vector{T}=[0]) where {T <: Number}
    M = similar(ω1, ComplexF64, length(sweep_phase) * length(ω1))
    for i in eachindex(sweep_phase)
        Mi = @view M[(i - 1) * length(ω1) + 1:i * length(ω1)]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, R2s_vec)
    end
    return M
end

# Version with gradients
function MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s::Number, R2s_T, grad_list::Array{T,1}, sweep_phase=[0]) where T <: grad_param
    R2s_vec = Calculate_Saturation_rate(ω1, TRF, B1, T2s, R2s_T, grad_list)
    return MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, grad_list, sweep_phase)
end

function MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, grad_list::Array{T,1}, sweep_phase=[0]) where T <: grad_param
    M = similar(ω1, ComplexF64, 1 + length(grad_list), length(sweep_phase) * length(ω1))
    for i in eachindex(sweep_phase)
        Mi = @view M[:, (i - 1) * length(ω1) + 1:i * length(ω1)]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, R2s_vec, grad_list)
    end
    return M
end

#########################################################
# helper functions
#########################################################
function MatrixApprox_calculate_magnetization!(M, ω1, TRF, TR, sweep_phase, ω0, B1, m0s, R1, R2f, Rx, R2s_vec)
    
    (_R2s, _dR2sdT2s, _dR2sdB1) = R2s_vec

    sweep_phase += π
    u_rot = @SMatrix [cos(sweep_phase) -sin(sweep_phase) 0 0 0 0;
                      sin(sweep_phase)  cos(sweep_phase) 0 0 0 0;
                                     0                0  1 0 0 0;
                                     0                0  0 1 0 0;
                                     0                0  0 0 1 0;
                                     0                0  0 0 0 1]
    
    y0 = Anti_periodic_boundary_conditions(ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, [], u_rot)

    Propagate_magnetization!(M, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, u_rot, y0, [])
        
    return nothing
end

function MatrixApprox_calculate_magnetization!(M, ω1, TRF, TR, sweep_phase, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, grad_list)
    
    (_R2s, _dR2sdT2s, _dR2sdB1) = R2s_vec

    sweep_phase += π
    
    u_rot = @SMatrix [cos(sweep_phase) -sin(sweep_phase) 0 0 0 0                 0                0 0 0 0;
                      sin(sweep_phase)  cos(sweep_phase) 0 0 0 0                 0                0 0 0 0;
                      0                 0                1 0 0 0                 0                0 0 0 0;
                      0                 0                0 1 0 0                 0                0 0 0 0;
                      0                 0                0 0 1 0                 0                0 0 0 0;
                      0                 0                0 0 0 cos(sweep_phase) -sin(sweep_phase) 0 0 0 0;
                      0                 0                0 0 0 sin(sweep_phase)  cos(sweep_phase) 0 0 0 0;
                      0                 0                0 0 0 0                 0                1 0 0 0;
                      0                 0                0 0 0 0                 0                0 1 0 0;
                      0                 0                0 0 0 0                 0                0 0 1 0;
                      0                 0                0 0 0 0                 0                0 0 0 1]

    nM = size(M, 1) ÷ (length(grad_list) + 1) # 1 component for signal, 5 components for magnetization (xf, yf, zf, xs, zs)
    for i in eachindex(grad_list)
        y0 = Anti_periodic_boundary_conditions(ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, grad_list[i], u_rot)

        # pick first set of rows for the main magnetization and second one for the derivatives
        Mig = @view M[[1:nM;i * nM + 1:(i + 1) * nM],:] 
        Propagate_magnetization!(Mig, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, u_rot, y0, grad_list[i])
    end
    return nothing
end

function Anti_periodic_boundary_conditions(ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, grad, u_rot)
    
    # put inversion pulse at the end (this defines y0 as the magnetization at the first TE after the inversion pulse)
    u_fp = exp(Linear_Hamiltonian_Matrix(0, B1, ω0, TR / 2, m0s, R1, R2f, Rx, _R2s[1], _dR2sdT2s[1], _dR2sdB1[1], grad))
    u_pl = Inversion_Pulse_Propagator(ω1[1], TRF[1], B1, _R2s[1], _dR2sdT2s[1], _dR2sdB1[1], grad)
    A = u_fp * u_pl * u_rot * u_fp
    
    for i = length(ω1):-1:2
        u_fp = exp(Linear_Hamiltonian_Matrix(0, B1, ω0, (TR - TRF[i]) / 2, m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))
        u_pl = exp(Linear_Hamiltonian_Matrix(ω1[i], B1, ω0, TRF[i], m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))
        A = A * u_fp * u_pl * u_rot * u_fp
    end

    F = eigen(A)
    y0 = F.vectors[:,end]
    y0 /= y0[end]
    return y0
end

function Propagate_magnetization!(M::AbstractArray{T,2}, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, u_rot, y0, grad) where T <: Real
    M[:,1] = @view y0[1:end - 1]
    for i = 2:length(ω1)
        u_fp = exp(Linear_Hamiltonian_Matrix(0, B1, ω0, (TR - TRF[i]) / 2, m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))
        u_pl = exp(Linear_Hamiltonian_Matrix(ω1[i], B1, ω0, TRF[i], m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))

        y0 = u_fp * (u_pl * (u_rot * (u_fp * y0)))
        M[:,i] = @view y0[1:end - 1]
    end
    return nothing
end

function Propagate_magnetization!(S::AbstractArray{T,1}, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, u_rot, y0, grad) where T <: Complex
    S[1] = y0[1] + 1im * y0[2]
    for i = 2:length(ω1)
        u_fp = exp(Linear_Hamiltonian_Matrix(0, B1, ω0, (TR - TRF[i]) / 2, m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))
        u_pl = exp(Linear_Hamiltonian_Matrix(ω1[i], B1, ω0, TRF[i], m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))

        y0 = u_fp * (u_pl * (u_rot * (u_fp * y0)))
        S[i] = y0[1] + 1im * y0[2]
    end
    return nothing
end

function Propagate_magnetization!(S::AbstractArray{T,2}, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, u_rot, y0, grad) where T <: Complex
    S[1,1] = y0[1] + 1im * y0[2]
    S[2,1] = y0[6] + 1im * y0[7]
    for i = 2:length(ω1)
        u_fp = exp(Linear_Hamiltonian_Matrix(0, B1, ω0, (TR - TRF[i]) / 2, m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))
        u_pl = exp(Linear_Hamiltonian_Matrix(ω1[i], B1, ω0, TRF[i], m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))

        y0 = u_fp * (u_pl * (u_rot * (u_fp * y0)))
        S[1,i] = y0[1] + 1im * y0[2]
        S[2,i] = y0[6] + 1im * y0[7]
    end
    return nothing
end
