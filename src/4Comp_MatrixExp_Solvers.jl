############################################################################
# main call function
############################################################################
function calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, T2_M, T2_NM, R2slT; rfphase_increment=[π], m0=:periodic, output=:complexsignal)

    _R2s_M  = evaluate_R2sl_vector(abs.(α), TRF, B1, T2_M,  R2slT, undef)[1]
    _R2s_NM = evaluate_R2sl_vector(abs.(α), TRF, B1, T2_NM, R2slT, undef)[1]

    return calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, _R2s_M, _R2s_NM; rfphase_increment=rfphase_increment, m0=m0, output=output)
end

function calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, _R2s_M, _R2s_NM; rfphase_increment=[π], m0=:periodic, output=:complexsignal)

    ω1 = α ./ TRF

    if output == :complexsignal
        S = Array{ComplexF64}(undef, length(ω1), length(rfphase_increment))
    elseif output == :realmagnetization
        S = Array{SVector{6,Float64}}(undef, length(ω1), length(rfphase_increment))
    end

    for i in eachindex(rfphase_increment)
        if m0 == :periodic
            m = antiperiodic_boundary_conditions_linear(ω1, B1, ω0, TRF, TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, _R2s_M, _R2s_NM, rfphase_increment[i])
        end

        Mij = @view S[:, i]
        propagate_magnetization_linear!(Mij, m, ω1, B1, ω0, TRF, TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, _R2s_M, _R2s_NM, rfphase_increment[i])
    end
    return S
end

############################################################################
# helper functions
############################################################################
function evolution_matrix_linear(ω1, B1, ω0, TRF, TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, _R2s_M, _R2s_NM, rfphase_increment=π)
    # put inversion pulse at the end (this defines m as the magnetization at the first TE after the inversion pulse)
    u_rot = z_rotation_propagator(rfphase_increment)
    u_fp = xs_destructor() * exp(hamiltonian_linear(0, B1, ω0, TR / 2, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, 0, 0))
    u_pl = propagator_linear_inversion_pulse(ω1[1], TRF[1], B1, _R2s_M[1], _R2s_NM[1])
    A = u_fp * u_pl * u_rot * u_fp

    for i = length(ω1):-1:2
        u_fp = xs_destructor() * exp(hamiltonian_linear(0, B1, ω0, (TR - TRF[i]) / 2, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW,0, 0))
        u_pl = exp(hamiltonian_linear(ω1[i], B1, ω0, TRF[i], m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, _R2s_M[i], _R2s_NM[i]))
        A = A * u_fp * u_pl * u_rot * u_fp
    end
    return A
end

function antiperiodic_boundary_conditions_linear(ω1, B1, ω0, TRF, TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, _R2s_M, _R2s_NM, rfphase_increment=π)
    A = evolution_matrix_linear(ω1, B1, ω0, TRF, TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, _R2s_M, _R2s_NM, rfphase_increment)

    Q = A - A0()
    m = Q \ C()
    return m
end

function propagate_magnetization_linear!(S, m, ω1, B1, ω0, TRF, TR, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, _R2s_M, _R2s_NM, rfphase_increment)

    u_rot = z_rotation_propagator(rfphase_increment)
    ms_setindex!(S, m, 1)
    for i = 2:length(ω1)
        u_fp = xs_destructor() * exp(hamiltonian_linear(0, B1, ω0, (TR - TRF[i]) / 2, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, 0, 0))
        u_pl = exp(hamiltonian_linear(ω1[i], B1, ω0, TRF[i], m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, _R2s_M[i], _R2s_NM[i]))

        m = u_fp * (u_pl * (u_rot * (u_fp * m)))

        ms_setindex!(S, m, i)
    end
    return S
end

function ms_setindex!(S::AbstractArray{<:AbstractVector}, y, i)
    S[i] = y
    return S
end
function ms_setindex!(S::AbstractArray{<:Complex}, y, i)
    S[i] = y[1] + 1im * y[2] + y[6] + 1im * y[7]
    return S
end