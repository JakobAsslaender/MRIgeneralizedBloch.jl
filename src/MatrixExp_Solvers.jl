############################################################################
# main call function
############################################################################
function calculatesignal_linearapprox(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec; grad_list=[undef], sweep_phase=[0], m0=:antiperiodic, output=:complexsignal)

    if output == :complexsignal && grad_list == [undef]
        S = Array{ComplexF64}(undef, length(ω1), length(sweep_phase), 1)
        jj = 1
    elseif output == :complexsignal
        S = Array{ComplexF64}(undef, length(ω1), length(sweep_phase), 1 + length(grad_list))
        jj = [[1,j+1] for j=1:length(grad_list)]
    elseif output == :realmagnetization && grad_list == [undef]
        S = Array{SVector{6,Float64}}(undef, length(ω1), length(sweep_phase), length(grad_list))
        jj = 1
    elseif output == :realmagnetization
        S = Array{SVector{11,Float64}}(undef, length(ω1), length(sweep_phase), length(grad_list))
        jj = 1:length(grad_list)
    end

    if m0 == :thermal || m0 == :IR
        if grad_list == [undef]
            _m0 = @SVector [0, 0, 1-m0s, 0, m0s, 1]
        else
            _m0 = @SVector [0, 0, 1-m0s, 0, m0s, 0, 0, 0, 0, 0, 1]
        end
    elseif isa(m0, AbstractVector)
        if grad_list == [undef] && length(m0) < 6
            _m0 = SVector{6}([m0; zeros(5-length(m0)); 1])
        elseif length(m0) < 11
            _m0 = SVector{11}([m0; zeros(10-length(m0)); 1])
        else
            error("the m0 vector of length <= 5 w/o gradients or <= 10 with gradients")
        end
    elseif m0 != :antiperiodic
        error("m0 must either be :antiperiodic, :IR, :thermal, or a vector")
    end

    for j in eachindex(grad_list), i in eachindex(sweep_phase)
        if m0 == :antiperiodic
            m = antiperiodic_boundary_conditions_linear(ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, R2s_vec[1], R2s_vec[2], R2s_vec[3], grad_list[j], sweep_phase[i])
        elseif m0 == :thermal || isa(m0, AbstractVector)
            # this implements the α[1]/2 - TR/2 preparation (TR/2 is implemented in propagate_magnetization_linear!)
            u_pr = exp(hamiltonian_linear(ω1[1]/2, B1, ω0, TRF[1], m0s, R1, R2f, Rx, R2s_vec[1][1], R2s_vec[2][1], R2s_vec[3][1], grad)) # R2sl is actually wrong for the prep pulse
            m = u_pr * _m0
        elseif m0 == :IR
            # this implements the π - spoiler - TR - α[2]/2 - TR/2 preparation (TR/2 is implemented in propagate_magnetization_linear!)
            u_ip = propagator_linear_inversion_pulse(ω1[1], TRF[1], B1, R2s_vec[1][1], R2s_vec[2][1], R2s_vec[3][1], grad)
            m = u_ip * _m0
            println(m)
            u_fp = exp(hamiltonian_linear(0, B1, ω0, TR, m0s, R1, R2f, Rx, R2s_vec[1][1], R2s_vec[2][1], R2s_vec[3][1], grad))
            m = u_fp * m
            println(m)
            u_pr = exp(hamiltonian_linear(ω1[2]/2, B1, ω0, TRF[2], m0s, R1, R2f, Rx, R2s_vec[1][2], R2s_vec[2][2], R2s_vec[3][2], grad)) # R2sl is actually wrong for the prep pulse
            m = u_pr * m
            println(m)
        end

        Mij = @view S[:, i, jj[j]]
        propagate_magnetization_linear!(Mij, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, R2s_vec[1], R2s_vec[2], R2s_vec[3], sweep_phase[i], m, grad_list[j])
    end
    return S
end

function calculatesignal_linearapprox(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2s_T; grad_list=[undef], sweep_phase=[0], m0=:antiperiodic, output=:complexsignal)

    R2s_vec = evaluate_R2sl_vector(ω1, TRF, B1, T2s, R2s_T, grad_list)

    return calculatesignal_linearapprox(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec; grad_list=grad_list, sweep_phase=sweep_phase, m0=m0, output=output)
end


############################################################################
# helper functions
############################################################################
function antiperiodic_boundary_conditions_linear(ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, grad=undef, sweep_phase=0)
    # put inversion pulse at the end (this defines m as the magnetization at the first TE after the inversion pulse)
    u_rot = z_rotation_propagator(sweep_phase, grad)
    u_fp = exp(hamiltonian_linear(0, B1, ω0, TR / 2, m0s, R1, R2f, Rx, _R2s[1], _dR2sdT2s[1], _dR2sdB1[1], grad))
    u_pl = propagator_linear_inversion_pulse(ω1[1], TRF[1], B1, _R2s[1], _dR2sdT2s[1], _dR2sdB1[1], grad)
    A = u_fp * u_pl * u_rot * u_fp
    
    for i = length(ω1):-1:2
        u_fp = exp(hamiltonian_linear(0, B1, ω0, (TR - TRF[i]) / 2, m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))
        u_pl = exp(hamiltonian_linear(ω1[i], B1, ω0, TRF[i], m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))
        A = A * u_fp * u_pl * u_rot * u_fp
    end

    m = eigvecs(A)[:,end]
    m /= m[end]
    return m
end

function propagate_magnetization_linear!(S, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, sweep_phase, m, grad=undef)

    u_rot = z_rotation_propagator(sweep_phase, grad)
    println(m)
    ms_setindex!(S, m, 1, grad)
    for i = 2:length(ω1)
        u_fp = exp(hamiltonian_linear(0, B1, ω0, (TR - TRF[i]) / 2, m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))
        u_pl = exp(hamiltonian_linear(ω1[i], B1, ω0, TRF[i], m0s, R1, R2f, Rx, _R2s[i], _dR2sdT2s[i], _dR2sdB1[i], grad))

        m = u_fp * (u_pl * (u_rot * (u_fp * m)))

        ms_setindex!(S, m, i, grad)
    end
    return S
end

function ms_setindex!(S::AbstractArray{<:AbstractVector}, y, i, grad)
    S[i] = y
    return S
end
function ms_setindex!(S::AbstractArray{<:Complex}, y, i, grad)
    S[i] = y[1] + 1im * y[2]
    return S
end
function ms_setindex!(S::AbstractArray{<:Complex}, y, i, grad::grad_param)
    S[i,1] = y[1] + 1im * y[2]
    S[i,2] = y[6] + 1im * y[7]
    return S
end