############################################################################
# convenience wrapper to calcuate real valued magnetization components
############################################################################
# Version w/o gradients
# TODO: remove T, change , to ;
function MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s::Number, R2s_T, sweep_phase::Vector{T}=[0]) where {T <: Number}
    R2s_vec = evaluate_R2sl_vector(ω1, TRF, B1, T2s, R2s_T, [undef])
    return MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, sweep_phase)
end

function MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, sweep_phase::Vector{T}=[0]) where {T <: Number}
    M = similar(ω1, length(sweep_phase) * length(ω1), 5)

    for i in eachindex(sweep_phase)
        Mi = @view M[(i - 1) * length(ω1) + 1:i * length(ω1), :]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, R2s_vec)
    end
    return M
end

# Version with gradients
function MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s::Number, R2s_T, grad_list::Array{T,1}, sweep_phase=[0]) where T <: grad_param
    R2s_vec = evaluate_R2sl_vector(ω1, TRF, B1, T2s, R2s_T, grad_list)
    return MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, grad_list, sweep_phase)
end

function MatrixApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, grad_list::Array{T,1}, sweep_phase=[0]) where {T <: grad_param}
    M = similar(ω1, length(sweep_phase) * length(ω1), 5 * (1 + length(grad_list)))

    for i in eachindex(sweep_phase)
        Mi = @view M[:, (i - 1) * length(ω1) + 1:i * length(ω1)]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, R2s_vec, grad_list)
    end
    return M
end

############################################################################
# convenience wrapper to calcuate complex valued signal (Mxf + 1im * Myf)
############################################################################
# Version w/o gradients
function MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s::Number, R2s_T, sweep_phase::Vector{T}=[0]) where {T <: Number}
    R2s_vec = evaluate_R2sl_vector(ω1, TRF, B1, T2s, R2s_T, [undef])
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
    R2s_vec = evaluate_R2sl_vector(ω1, TRF, B1, T2s, R2s_T, grad_list)
    return MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, grad_list, sweep_phase)
end

function MatrixApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, grad_list::Array{T,1}, sweep_phase=[0]) where T <: grad_param
    M = similar(ω1, ComplexF64, length(sweep_phase) * length(ω1), 1 + length(grad_list))
    for i in eachindex(sweep_phase)
        Mi = @view M[(i - 1) * length(ω1) + 1:i * length(ω1), :]
        MatrixApprox_calculate_magnetization!(Mi, ω1, TRF, TR, sweep_phase[i], ω0, B1, m0s, R1, R2f, Rx, R2s_vec, grad_list)
    end
    return M
end


############################################################################
# actual solvers, which are implemented with in-place calcluation of M / S
############################################################################
function MatrixApprox_calculate_magnetization!(M, ω1, TRF, TR, sweep_phase, ω0, B1, m0s, R1, R2f, Rx, R2s_vec, grad_list=undef)

    y0 = antiperiodic_boundary_conditions_linear(ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, R2s_vec[1], R2s_vec[2], R2s_vec[3], sweep_phase, grad_list)
    # ms_setindex!(M, y0, 1)

    propagate_magnetization_linear!(M, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, R2s_vec[1], R2s_vec[2], R2s_vec[3], sweep_phase, y0, grad_list)
    return M
end


############################################################################
# helper functions
############################################################################
function antiperiodic_boundary_conditions_linear(ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, sweep_phase=0, grad=undef)
    # put inversion pulse at the end (this defines y0 as the magnetization at the first TE after the inversion pulse)
    u_rot = z_rotation_propagator(sweep_phase, grad)
    u_fp = exp(hamiltonian_linear(0, B1, ω0, TR / 2, m0s, R1, R2f, Rx, _R2s[1], _dR2sdT2s[1], _dR2sdB1[1], grad))
    u_pl = propagator_linear_inversion_pulse(ω1[1], TRF[1], B1, _R2s[1], _dR2sdT2s[1], _dR2sdB1[1], grad)
    A = u_fp * u_pl * u_rot * u_fp
    
    for i = length(ω1):-1:2
        u_fp = exp(hamiltonian_linear(0, B1, ω0, (TR - TRF[i]) / 2, m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))
        u_pl = exp(hamiltonian_linear(ω1[i], B1, ω0, TRF[i], m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))
        A = A * u_fp * u_pl * u_rot * u_fp
    end

    F = eigen(A)
    y0 = F.vectors[:,end]
    y0 /= y0[end]
    return y0
end

function antiperiodic_boundary_conditions_linear(ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, sweep_phase, grad_list::Vector{<:grad_param})
    y0 = similar(grad_list, Vector)
    # y0 = Vector{Vector{Float64}}(undef, length(grad_list))
    for j in eachindex(grad_list)
        y0[j] = antiperiodic_boundary_conditions_linear(ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, sweep_phase, grad_list[j])
    end
    return y0
end


function propagate_magnetization_linear!(M, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, sweep_phase, y0, grad=undef)

    u_rot = z_rotation_propagator(sweep_phase, grad)

    ms_setindex!(M, y0, 1)
    for i = 2:length(ω1)
        u_fp = exp(hamiltonian_linear(0, B1, ω0, (TR - TRF[i]) / 2, m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))
        u_pl = exp(hamiltonian_linear(ω1[i], B1, ω0, TRF[i], m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))

        y0 = u_fp * (u_pl * (u_rot * (u_fp * y0)))
        ms_setindex!(M, y0, i)
    end
    return M
end

function propagate_magnetization_linear!(M, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, u_rot, y0, grad_list::Vector{<:grad_param})
    nM = size(M, 2) ÷ (length(grad_list) + 1) # 1 component for signal, 5 components for magnetization (xf, yf, zf, xs, zs)

    for j in eachindex(grad_list)
        Mj = @view M[:, [1:nM; j * nM + 1:(j + 1) * nM]] # pick first set of rows for the main magnetization and second one for the derivatives
        propagate_magnetization_linear!(Mj, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, u_rot, y0[j], grad_list[j])
    end
    return M
end

function ms_setindex!(M::AbstractArray{<:Real,2}, y, i)
    M[i,:] = @view y[1:end - 1]
    return M
end
function ms_setindex!(S::AbstractArray{<:Complex,1}, y, i)
    S[i] = y[1] + 1im * y[2]
    return S
end
function ms_setindex!(S::AbstractArray{<:Complex,2}, y, i)
    S[i,1] = y[1] + 1im * y[2]
    S[i,2] = y[6] + 1im * y[7]
    return S
end

function z_rotation_propagator(sweep_phase, _)
    sweep_phase += π
    u_rot = @SMatrix [cos(sweep_phase) -sin(sweep_phase) 0 0 0 0;
                      sin(sweep_phase)  cos(sweep_phase) 0 0 0 0;
                                     0                0  1 0 0 0;
                                     0                0  0 1 0 0;
                                     0                0  0 0 1 0;
                                     0                0  0 0 0 1]
    return u_rot
end

function z_rotation_propagator(sweep_phase, grad::grad_param)
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
    return u_rot
end