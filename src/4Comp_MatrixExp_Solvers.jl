############################################################################
# main call function
############################################################################
function calculatesignal_linearapprox(α, TRF, TR, ω0, B1,
    m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, T2_M, T2_NM, R2slT_M, R2slT_NM; grad_list=(undef,), rfphase_increment=[π], m0=:periodic, preppulse=false, output=:complexsignal, isInversionPulse=[true; falses(length(α) - 1)])

    R2_M, dR2dT2_M, dR2dB1_M = evaluate_R2sl_vector(abs.(α), TRF, B1, T2_M, R2slT_M, grad_list)
    R2_NM, dR2dT2_NM, dR2dB1_NM = evaluate_R2sl_vector(abs.(α), TRF, B1, T2_NM, R2slT_NM, grad_list)

    return calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM, dR2dT2_M, dR2dB1_M, dR2dT2_NM, dR2dB1_NM; grad_list, rfphase_increment, m0, preppulse, output, isInversionPulse)
end

function calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM, dR2dT2_M, dR2dB1_M, dR2dT2_NM, dR2dB1_NM; grad_list=(undef,), rfphase_increment=[π], m0=:periodic, preppulse=false, output=:complexsignal, isInversionPulse=[true; falses(length(α) - 1)])
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
        m = antiperiodic_boundary_Monditions_linear(ω1, B1, ω0, TRF, TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM, dR2dT2_M, dR2dB1_M, dR2dT2_NM, dR2dB1_NM, grad_list[j], rfphase_increment[i], isInversionPulse)

        # this implements the α[1]/2 - TR/2 preparation (TR/2 is implemented in propagate_magnetization_linear!)
        if preppulse
            k = (m0 == :IR) ? 2 : 1
            u_pr = exp(hamiltonian_linear(ω1[k] / 2, B1, ω0, TRF[k], m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M[k], R2_NM[k], dR2dT2_M[k], dR2dB1_M[k], dR2dT2_NM[k], dR2dB1_NM[k], grad_list[j])) # R2sl is actually wrong for the prep pulse
            m = u_pr * m
        end

        Mij = @view S[:, i, jj[j]]
        propagate_magnetization_linear!(Mij, m, ω1, B1, ω0, TRF, TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM, dR2dT2_M, dR2dB1_M, dR2dT2_NM, dR2dB1_NM, grad_list[j], rfphase_increment[i], isInversionPulse)
    end
    return S
end

############################################################################
# helper functions
############################################################################
function evolution_matrix_linear(ω1, B1, ω0, TRF, TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM, dR2dT2_M, dR2dB1_M, dR2dT2_NM, dR2dB1_NM, grad, rfphase_increment, isInversionPulse)

    # put inversion pulse at the end (this defines m as the magnetization at the first TE after the inversion pulse)
    u_fp, u_pl = pulse_propagators(ω1[1], B1, ω0, TRF[1], TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M[1], R2_NM[1], dR2dT2_M[1], dR2dB1_M[1], dR2dT2_NM[1], dR2dB1_NM[1], grad, isInversionPulse[1])
    u_rot = z_rotation_propagator(rfphase_increment, u_fp)

    A = u_fp * u_pl * u_rot * u_fp

    for i = length(ω1):-1:2
        u_fp, u_pl = pulse_propagators(ω1[i], B1, ω0, TRF[i], TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M[i], R2_NM[i], dR2dT2_M[i], dR2dB1_M[i], dR2dT2_NM[i], dR2dB1_NM[i], grad, isInversionPulse[i])

        A = A * u_fp * u_pl * u_rot * u_fp
    end
    return A
end

function antiperiodic_boundary_Monditions_linear(ω1, B1, ω0, TRF, TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM, dR2dT2_M, dR2dB1_M, dR2dT2_NM, dR2dB1_NM, grad, rfphase_increment, isInversionPulse)
    A = evolution_matrix_linear(ω1, B1, ω0, TRF, TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM, dR2dT2_M, dR2dB1_M, dR2dT2_NM, dR2dB1_NM, grad, rfphase_increment, isInversionPulse)

    Q = A - A0(A)
    m = Q \ C(A)
    return m
end

function propagate_magnetization_linear!(S, m, ω1, B1, ω0, TRF, TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM, dR2dT2_M, dR2dB1_M, dR2dT2_NM, dR2dB1_NM, grad, rfphase_increment, isInversionPulse)


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
        u_fp, u_pl = pulse_propagators(ω1[i], B1, ω0, TRF[i], TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M[i], R2_NM[i], dR2dT2_M[i], dR2dB1_M[i], dR2dT2_NM[i], dR2dB1_NM[i], grad, isInversionPulse[i])
        u_rot = z_rotation_propagator(rfphase_increment, u_fp)

        m = u_fp * (u_pl * (u_rot * (u_fp * m)))
        ms_setindex!(S, m, i, grad)
    end
    return S
end

function pulse_propagators(ω1, B1, ω0, TRF, TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM, dR2dT2_M, dR2dB1_M, dR2dT2_NM, dR2dB1_NM, grad, isInversionPulse)
    if isInversionPulse # inversion pulses are accompanied by crusher gradients
        u_fp = exp(hamiltonian_linear(0, B1, ω0, TR / 2, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, 0, 0, 0, 0, 0, 0, grad))
        u_fp = xs_destructor(u_fp) * u_fp

        u_pl = propagator_linear_inversion_pulse(ω1, TRF, B1, R2_M, R2_NM, dR2dT2_M, dR2dB1_M, dR2dT2_NM, dR2dB1_NM, grad)
    else
        u_fp = exp(hamiltonian_linear(0, B1, ω0, (TR - TRF) / 2, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, 0, 0, 0, 0, 0, 0, grad))
        u_fp = xs_destructor(u_fp) * u_fp
        u_pl = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM, dR2dT2_M, dR2dB1_M, dR2dT2_NM, dR2dB1_NM, grad))
    end
    return u_fp, u_pl
end