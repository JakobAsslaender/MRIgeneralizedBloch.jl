############################################################################
# main call function
############################################################################
"""
    calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT[; grad_list=(undef,), rfphase_increment=[π], m0=:periodic, output=:complexsignal, grad_moment = [i == 1 ? :spoiler_dual : :balanced for i ∈ eachindex(α)])

Calculate the signal or magnetization evolution with the linear approximation of the generalized Bloch model assuming a super-Lorentzian lineshape.

The simulation assumes a sequence of rectangular RF-pulses with varying flip angles α and RF-pulse durations TRF, but a fixed repetition time TR. Further, it assumes balanced gradient moments.

# Arguments
- `α::Vector{Real}`: Array of flip angles in radians
- `TRF::Vector{Real}`: Array of the RF-pulse durations in seconds
- `TR::Real`: Repetition time in seconds
- `ω0::Real`: Off-resonance frequency in rad/s
- `B1::Real`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `m0s::Real`: Fractional size of the semi-solid pool; should be in range of 0 to 1
- `R1f::Real`: Longitudinal relaxation rate of the free pool in 1/seconds
- `R2f::Real`: Transversal relaxation rate of the free pool in 1/seconds
- `Rex::Real`: Exchange rate between the two spin pools in 1/seconds
- `R1f::Real`: Longitudinal relaxation rate of the semi-solid pool in 1/seconds
- `T2s::Real`: Transversal relaxation time of the semi-solid pool in seconds
- `R2slT::NTuple{3, Function}`: Tuple of three functions: R2sl(TRF, ω1, B1, T2s), dR2sldB1(TRF, ω1, B1, T2s), and R2sldT2s(TRF, ω1, B1, T2s). Can be generated with [`precompute_R2sl`](@ref)

Optional:
- `grad_list=(undef,)`: Tuple that specifies the gradients that are calculated; the vector elements can either be `undef` for no gradient, or any subset/order of `grad_list=(grad_m0s(), grad_R1f(), grad_R2f(), grad_Rex(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1())`; the derivative wrt. to apparent `R1a = R1f = R1s` can be calculated with `grad_R1a()`
- `rfphase_increment=[π]::Vector{Real}`: Increment of the RF phase between consecutive pulses. The default value `π`, together with ``ω0=0`` corresponds to the on-resonance condition. When more than one value is supplied, their resulting signal is stored along the second dimension of the output array.
- `m0=:periodic`: With the default keyword `:periodic`, the signal and their derivatives are calculated assuming ``m(0) = -m(T)``, where `T` is the duration of the RF-train. With the keyword :thermal, the magnetization ``m(0)`` is initialized with thermal equilibrium `[xf, yf, zf, xs, zs] = [0, 0, 1-m0s, 0, m0s]`, followed by a α[1]/2 - TR/2 prep pulse; and with the keyword `:IR`, this initialization is followed an inversion pulse of duration `TRF[1]`, (set `α[1]=π`) and a α[2]/2 - TR/2 prep pulse.
- `preppulse=false`: if `true`, a `α/2 - TR/2` preparation is applied. In the case of `m0=:IR`, it is applied after the inversion pulse based on `α[2]`, otherwise it is based on `α[1]`
- `output=:complexsignal`: The default keywords triggers the function to output a complex-valued signal (`xf + 1im yf`); the keyword `output=:realmagnetization` triggers an output of the entire (real valued) vector `[xf, yf, zf, xs, zs, 1]`
- `grad_moment = [i == 1 ? :spoiler_dual : :balanced for i ∈ eachindex(α)]`: Different types of gradient moments of each TR are possible (`:balanced`, `:crusher`, `:spoiler_dual`, `:spoiler_prepulse`). `:balanced` simulates a TR with all gradient moments nulled. `:crusher` assumes equivalent (non-zero) gradient moments before and simulates the refocussing path of the extended phase graph. `:spoiler_prepulse` nulls all transverse magnetization before the RF pulse, emulating an idealized FLASH. `:spoiler_dual` nulls all transverse magnetization before and after the RF pulse.

# Examples
```jldoctest
julia> R2slT = precompute_R2sl();

julia> calculatesignal_linearapprox(range(0, π/2, 100), fill(5e-4, 100), 4e-3, 0, 1, 0.1, 1, 15, 30, 6.5, 10e-6, R2slT)
100×1×1 Array{ComplexF64, 3}:
[:, :, 1] =
                  -0.0 - 0.0im
 0.0019738333472518535 + 0.0im
   0.00218756542837078 + 0.0im
  0.004164448431837324 + 0.0im
  0.004592307591163216 + 0.0im
  0.006593922316239837 + 0.0im
  0.007231933411392494 + 0.0im
   0.00927379103757742 + 0.0im
  0.010115002530621257 + 0.0im
  0.012207846285906883 + 0.0im
                       ⋮
   0.13200818588649146 + 0.0im
   0.13125741802520272 + 0.0im
    0.1304655306901449 + 0.0im
   0.12961406988250931 + 0.0im
    0.1287244900784642 + 0.0im
   0.12778027457741703 + 0.0im
   0.12680111371298095 + 0.0im
   0.12577221756507706 + 0.0im
   0.12471167167688701 + 0.0im
```
"""
function calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT; grad_list=(undef,), rfphase_increment=[π], m0=:periodic, preppulse=false, output=:complexsignal, grad_moment = [i == 1 ? :spoiler_dual : :balanced for i ∈ eachindex(α)])

    R2s, dR2sdT2s, dR2sdB1 = evaluate_R2sl_vector(abs.(α), TRF, B1, T2s, R2slT, grad_list)

    return calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, R2s, dR2sdT2s, dR2sdB1; grad_list, rfphase_increment, m0, preppulse, output, grad_moment)
end

function calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, R2s, dR2sdT2s, dR2sdB1; grad_list=(undef,), rfphase_increment=[π], m0=:periodic, preppulse=false, output=:complexsignal, grad_moment = [i == 1 ? :spoiler_dual : :balanced for i ∈ eachindex(α)])
    if isempty(grad_list)
        grad_list = (undef,)
    end

    ω1 = α ./ TRF

    if grad_list == (undef,)
        jj = 1
        if output == :complexsignal
            S = Array{ComplexF64}(undef, length(ω1), length(rfphase_increment), 1)
        elseif output == :realmagnetization
            S = Array{SVector{6,Float64}}(undef, length(ω1), length(rfphase_increment), 1)
        end
    else
        if output == :complexsignal
            S = Array{ComplexF64}(undef, length(ω1), length(rfphase_increment), 1 + length(grad_list))
            jj = [[1,j + 1] for j = 1:length(grad_list)]
        elseif output == :realmagnetization
            S = Array{SVector{11,Float64}}(undef, length(ω1), length(rfphase_increment), length(grad_list))
            jj = 1:length(grad_list)
        end
    end

    if m0 == :thermal || m0 == :IR
        if grad_list == (undef,)
            _m0 = @SVector [0, 0, 1 - m0s, 0, m0s, 1]
        else
            _m0 = @SVector [0, 0, 1 - m0s, 0, m0s, 0, 0, 0, 0, 0, 1]
        end
    elseif isa(m0, AbstractVector)
        if grad_list == (undef,)
            _m0 = length(m0) == 6 ? SVector{6}(m0) : SVector{6}([m0; zeros(5 - length(m0)); 1])
        else
            _m0 = length(m0) == 11 ? SVector{11}(m0) : SVector{11}([m0; zeros(10 - length(m0)); 1])
        end
    elseif m0 != :periodic
        error("m0 must either be :periodic, :IR, :thermal, or a vector")
    end

    for j in eachindex(grad_list), i in eachindex(rfphase_increment)
        if m0 == :periodic
            m = antiperiodic_boundary_conditions_linear(ω1, B1, ω0, TRF, TR, m0s, R1f, R2f, Rex, R1s, R2s, dR2sdT2s, dR2sdB1, grad_list[j], rfphase_increment[i], grad_moment)
        elseif m0 == :thermal || isa(m0, AbstractVector)
            m = _m0
        elseif m0 == :IR
            # this implements the π - spoiler - TR preparation
            u_ip = propagator_linear_crushed_pulse(ω1[1], TRF[1], B1, R2s[1], dR2sdT2s[1], dR2sdB1[1], grad_list[j])
            m = u_ip * _m0
            u_fp = exp(hamiltonian_linear(0, B1, ω0, TR, m0s, R1f, R2f, Rex, R1s, 0, 0, 0, grad_list[j]))
            u_fp = xs_destructor(u_fp) * u_fp

            m = u_fp * m
        end

        # this implements the α[1]/2 - TR/2 preparation (TR/2 is implemented in propagate_magnetization_linear!)
        if preppulse
            k = (m0 == :IR) ? 2 : 1
            u_pr = exp(hamiltonian_linear(ω1[k] / 2, B1, ω0, TRF[k], m0s, R1f, R2f, Rex, R1s, R2s[k], dR2sdT2s[k], dR2sdB1[k], grad_list[j])) # R2sl is actually wrong for the prep pulse
            m = u_pr * m
        end

        Mij = @view S[:, i, jj[j]]
        propagate_magnetization_linear!(Mij, m, ω1, B1, ω0, TRF, TR, m0s, R1f, R2f, Rex, R1s, R2s, dR2sdT2s, dR2sdB1, grad_list[j], rfphase_increment[i], grad_moment)
    end
    return S
end

############################################################################
# helper functions
############################################################################
function evolution_matrix_linear(ω1, B1, ω0, TRF, TR, m0s, R1f, R2f, Rex, R1s, R2s, dR2sdT2s, dR2sdB1, grad, rfphase_increment, grad_moment)
    u_rot = z_rotation_propagator(rfphase_increment, grad)

    # put prep pulse at the end (this defines m as the magnetization at the first TE after the inversion pulse)
    u_fp, u_pl = pulse_propagators(ω1[1], B1, ω0, TRF[1], TR, m0s, R1f, R2f, Rex, R1s, R2s[1], dR2sdT2s[1], dR2sdB1[1], grad, grad_moment[1])
    A = u_fp * u_pl * u_rot * u_fp

    for i = length(ω1):-1:2
        u_fp, u_pl = pulse_propagators(ω1[i], B1, ω0, TRF[i], TR, m0s, R1f, R2f, Rex, R1s, R2s[i], dR2sdT2s[i], dR2sdB1[i], grad, grad_moment[i])

        A = A * u_fp * u_pl * u_rot * u_fp
    end
    return A
end

function antiperiodic_boundary_conditions_linear(ω1, B1, ω0, TRF, TR, m0s, R1f, R2f, Rex, R1s, R2s, dR2sdT2s, dR2sdB1, grad, rfphase_increment, grad_moment)
    A = evolution_matrix_linear(ω1, B1, ω0, TRF, TR, m0s, R1f, R2f, Rex, R1s, R2s, dR2sdT2s, dR2sdB1, grad, rfphase_increment, grad_moment)

    Q = A - A0(A)
    m = Q \ C(A)
    return m
end

function propagate_magnetization_linear!(S, m, ω1, B1, ω0, TRF, TR, m0s, R1f, R2f, Rex, R1s, R2s, dR2sdT2s, dR2sdB1, grad, rfphase_increment, grad_moment)

    u_rot = z_rotation_propagator(rfphase_increment, grad)
    ms_setindex!(S, m, 1, grad)
    for i = 2:length(ω1)
        u_fp, u_pl = pulse_propagators(ω1[i], B1, ω0, TRF[i], TR, m0s, R1f, R2f, Rex, R1s, R2s[i], dR2sdT2s[i], dR2sdB1[i], grad, grad_moment[i])

        m = u_fp * (u_pl * (u_rot * (u_fp * m)))
        ms_setindex!(S, m, i, grad)
    end
    return S
end

function pulse_propagators(ω1, B1, ω0, TRF, TR, m0s, R1f, R2f, Rex, R1s, R2s, dR2sdT2s, dR2sdB1, grad, grad_moment)
    if grad_moment == :crusher # refocus the transverse magnetization that existed before the RF pulse, but crush the FID path of this pulse
        u_fp = exp(hamiltonian_linear(0, B1, ω0, TR / 2, m0s, R1f, R2f, Rex, R1s, 0, 0, 0, grad))
        u_fp = xs_destructor(u_fp) * u_fp
        u_pl = propagator_linear_crushed_pulse(ω1, TRF, B1, R2s, dR2sdT2s, dR2sdB1, grad)
    elseif grad_moment == :spoiler_dual # destroy transverse magnetization before and after RF pulse
        u_fp = exp(hamiltonian_linear(0, B1, ω0, (TR - TRF) / 2, m0s, R1f, R2f, Rex, R1s, 0, 0, 0, grad))
        u_fp = xs_destructor(u_fp) * u_fp
        u_pl = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, Rex, R1s, R2s, dR2sdT2s, dR2sdB1, grad))
        u_pl = xy_destructor(u_pl) * u_pl * xy_destructor(u_pl)
    elseif grad_moment == :balanced # balance all moments
        u_fp = exp(hamiltonian_linear(0, B1, ω0, (TR - TRF) / 2, m0s, R1f, R2f, Rex, R1s, 0, 0, 0, grad))
        u_fp = xs_destructor(u_fp) * u_fp
        u_pl = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, Rex, R1s, R2s, dR2sdT2s, dR2sdB1, grad))
    elseif grad_moment == :spoiler_prepulse # destroy transverse magnetization before the RF pulse
        u_fp = exp(hamiltonian_linear(0, B1, ω0, (TR - TRF) / 2, m0s, R1f, R2f, Rex, R1s, 0, 0, 0, grad))
        u_fp = xs_destructor(u_fp) * u_fp
        u_pl = exp(hamiltonian_linear(ω1, B1, ω0, TRF, m0s, R1f, R2f, Rex, R1s, R2s, dR2sdT2s, dR2sdB1, grad))
        u_pl = u_pl * xy_destructor(u_pl)
    else
        error("Unknown gradient moment type.")
    end
    return u_fp, u_pl
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