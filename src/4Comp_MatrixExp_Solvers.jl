############################################################################
# main call function
############################################################################
function calculatesignal_linearapprox(α, TRF, TR, ω0, B1,
    m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, T2_C, T2_PG, R2slT_C, R2slT_PG; grad_list=(undef,), rfphase_increment=[π], m0=:periodic, preppulse=false, output=:complexsignal, isInversionPulse=[true; falses(length(α) - 1)])

    R2_C, dR2dT2_C, dR2dB1_C = evaluate_R2sl_vector(abs.(α), TRF, B1, T2_C, R2slT_C, grad_list)
    R2_PG, dR2dT2_PG, dR2dB1_PG = evaluate_R2sl_vector(abs.(α), TRF, B1, T2_PG, R2slT_PG, grad_list)

    return calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG, dR2dT2_C, dR2dB1_C, dR2dT2_PG, dR2dB1_PG; grad_list, rfphase_increment, m0, preppulse, output, isInversionPulse)
end

function calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG, dR2dT2_C, dR2dB1_C, dR2dT2_PG, dR2dB1_PG; grad_list=(undef,), rfphase_increment=[π], m0=:periodic, preppulse=false, output=:complexsignal, isInversionPulse=[true; falses(length(α) - 1)])
    if isempty(grad_list)
        grad_list = (undef,)
    end

    ω1 = α ./ TRF

    if grad_list == (undef,)
        jj = 1
        if output == :complexsignal
            S = Array{ComplexF64}(undef, length(ω1), length(rfphase_increment), 1)
        elseif output == :realmagnetization
            S = Array{SVector{11,Float64}}(undef, length(ω1), length(rfphase_increment), 1)
        end
    else
        if output == :complexsignal
            S = Array{ComplexF64}(undef, length(ω1), length(rfphase_increment), 1 + length(grad_list))
            jj = [[1, j + 1] for j = 1:length(grad_list)]
        elseif output == :realmagnetization
            S = Array{SVector{21,Float64}}(undef, length(ω1), length(rfphase_increment), length(grad_list))
            jj = 1:length(grad_list)
        end
    end

    if m0 != :periodic
        error("m0 must either be :periodic")
    end

    for j in eachindex(grad_list), i in eachindex(rfphase_increment)
        m = antiperiodic_boundary_conditions_linear(ω1, B1, ω0, TRF, TR, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG, dR2dT2_C, dR2dB1_C, dR2dT2_PG, dR2dB1_PG, grad_list[j], rfphase_increment[i], isInversionPulse)

        # this implements the α[1]/2 - TR/2 preparation (TR/2 is implemented in propagate_magnetization_linear!)
        if preppulse
            k = (m0 == :IR) ? 2 : 1
            u_pr = exp(hamiltonian_linear(ω1[k] / 2, B1, ω0, TRF[k], m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C[k], R2_PG[k], dR2dT2_C[k], dR2dB1_C[k], dR2dT2_PG[k], dR2dB1_PG[k], grad_list[j])) # R2sl is actually wrong for the prep pulse
            m = u_pr * m
        end

        Mij = @view S[:, i, jj[j]]
        propagate_magnetization_linear!(Mij, m, ω1, B1, ω0, TRF, TR, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG, dR2dT2_C, dR2dB1_C, dR2dT2_PG, dR2dB1_PG, grad_list[j], rfphase_increment[i], isInversionPulse)
    end
    return S
end

############################################################################
# helper functions
############################################################################
function evolution_matrix_linear(ω1, B1, ω0, TRF, TR, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG, dR2dT2_C, dR2dB1_C, dR2dT2_PG, dR2dB1_PG, grad, rfphase_increment, isInversionPulse)

    # put inversion pulse at the end (this defines m as the magnetization at the first TE after the inversion pulse)
    u_fp, u_pl = pulse_propagators(ω1[1], B1, ω0, TRF[1], TR, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C[1], R2_PG[1], dR2dT2_C[1], dR2dB1_C[1], dR2dT2_PG[1], dR2dB1_PG[1], grad, isInversionPulse[1])
    u_rot = z_rotation_propagator(rfphase_increment, u_fp)

    A = u_fp * u_pl * u_rot * u_fp

    for i = length(ω1):-1:2
        u_fp, u_pl = pulse_propagators(ω1[i], B1, ω0, TRF[i], TR, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C[i], R2_PG[i], dR2dT2_C[i], dR2dB1_C[i], dR2dT2_PG[i], dR2dB1_PG[i], grad, isInversionPulse[i])

        A = A * u_fp * u_pl * u_rot * u_fp
    end
    return A
end

function antiperiodic_boundary_conditions_linear(ω1, B1, ω0, TRF, TR, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG, dR2dT2_C, dR2dB1_C, dR2dT2_PG, dR2dB1_PG, grad, rfphase_increment, isInversionPulse)
    A = evolution_matrix_linear(ω1, B1, ω0, TRF, TR, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG, dR2dT2_C, dR2dB1_C, dR2dT2_PG, dR2dB1_PG, grad, rfphase_increment, isInversionPulse)

    Q = A - A0(A)
    m = Q \ C(A)
    return m
end

function propagate_magnetization_linear!(S, m, ω1, B1, ω0, TRF, TR, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG, dR2dT2_C, dR2dB1_C, dR2dT2_PG, dR2dB1_PG, grad, rfphase_increment, isInversionPulse)


    function ms_setindex!(S::AbstractArray{<:AbstractVector}, y, i, grad)
        S[i] = y
        return S
    end
    function ms_setindex!(S::AbstractArray{<:Complex}, y, i, grad)
        S[i] = y[1] + 1im * y[2] + y[6] + 1im * y[7]
        return S
    end
    function ms_setindex!(S::AbstractArray{<:Complex}, y, i, grad::grad_param)
        S[i, 1] = y[1] + 1im * y[2] + y[6] + 1im * y[7]
        S[i, 2] = y[11] + 1im * y[12] + y[16] + 1im * y[17]
        return S
    end

    ms_setindex!(S, m, 1, grad)
    for i = 2:length(ω1)
        u_fp, u_pl = pulse_propagators(ω1[i], B1, ω0, TRF[i], TR, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C[i], R2_PG[i], dR2dT2_C[i], dR2dB1_C[i], dR2dT2_PG[i], dR2dB1_PG[i], grad, isInversionPulse[i])
        u_rot = z_rotation_propagator(rfphase_increment, u_fp)

        m = u_fp * (u_pl * (u_rot * (u_fp * m)))
        ms_setindex!(S, m, i, grad)
    end
    return S
end

function pulse_propagators(ω1, B1, ω0, TRF, TR, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG, dR2dT2_C, dR2dB1_C, dR2dT2_PG, dR2dB1_PG, grad, isInversionPulse)
    if isInversionPulse # inversion pulses are accompanied by crusher gradients
        u_fp = exp(hamiltonian_linear(0, B1, ω0, TR / 2, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, 0, 0, 0, 0, 0, 0, grad))
        u_fp = xs_destructor(u_fp) * u_fp

        u_pl = propagator_linear_inversion_pulse(ω1, TRF, B1, R2_C, R2_PG, dR2dT2_C, dR2dB1_C, dR2dT2_PG, dR2dB1_PG, grad)
    else
        u_fp = exp(hamiltonian_linear(0, B1, ω0, (TR - TRF) / 2, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, 0, 0, 0, 0, 0, 0, grad))
        u_fp = xs_destructor(u_fp) * u_fp
        u_pl = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG, dR2dT2_C, dR2dB1_C, dR2dT2_PG, dR2dB1_PG, grad))
    end
    return u_fp, u_pl
end