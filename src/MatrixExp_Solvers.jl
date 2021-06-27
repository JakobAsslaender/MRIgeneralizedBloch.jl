############################################################################
# main call function
############################################################################
"""
    calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT[; grad_list=[undef], rfphase_increment=[π], m0=:antiperiodic, output=:complexsignal])

Calculate the signal or magnetization evolution with the linear approximation of the generalized Bloch model assuming a super-Loretzian lineshape. 

The simulation assumes a sequence of rectangluar RF-pulses with varying flip angles α and RF-pulse durations TRF, but a fixed repetition time TR. Further, it assumes balanced gradient moments. 

# Arguemnts
- `α::Vector{<:Number}`: Array of flip angles in radians
- `TRF::Vector{<:Number}`: Array of the RF-pulse durations in seconds
- `TR::Number`: Repetition time in seconds
- `ω0::Number`: Off-resonance frequency in rad/s 
- `B1::Number`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `m0s::Number`: Fractional size of the semi-solid pool; should be in range of 0 to 1
- `R1::Number`: Apparent longitudinal relaxation rate of the free and semi-solid pool in 1/seconds
- `R2f::Number`: Transversal relaxation rate of the free pool in 1/seconds
- `Rx::Number`: Exchange rate between the two spin pools in 1/seconds
- `T2s::Number`: Transversal relaxationt time of the semi-solid pool in seconds
- `R2slT::NTuple{3, Function}`: Tuple of three functions: R2sl(TRF, ω1, B1, T2s), dR2sldB1(TRF, ω1, B1, T2s), and R2sldT2s(TRF, ω1, B1, T2s). Can be generated with [`precompute_R2sl`](@ref)

Optional:
- `grad_list=[undef]`: Vector to indicate which gradients should be calculated; the vector elements can either be `undef` for no gradient, or any subset/order of `grad_list=[grad_m0s(), grad_R1(), grad_R2f(), grad_Rx(), grad_T2s(), grad_ω0(), grad_B1()]`
- `rfphase_increment=[π]::Vector{<:Number}`: Increment of the RF phase between consequtive pulses. The default value `π`, together with ``ω0=0`` corresponds to the on-resonance condition. When more than one value is supplied, their resulting signal is stored along the second dimension of the output array. 
- `m0=:antiperiodic`: With the default keyword `:antiperiodic`, the signal and their derivatives are calcualted assuming ``m(0) = -m(T)``, where `T` is the duration of the RF-train. With the keyword :thermal, the magnetization ``m(0)`` is initialized with thermal equilibrium `[xf, yf, zf, xs, zs] = [0, 0, 1-m0s, 0, m0s]`, followed by a α[1]/2 - TR/2 prep pulse; and with the keyword `:IR`, this initalization is followed an inversion pulse of duration `TRF[1]`, (set `α[1]=π`) and a α[2]/2 - TR/2 prep pulse.
- `output=:complexsignal`: The defaul keywords triggers the function to output a complex-valued signal (`xf + 1im yf`); the keyword `output=:realmagnetization` triggers an output of the entire (real valued) vector `[xf, yf, zf, xs, zs]`

# Examples
```jldoctest
julia> R2slT = precompute_R2sl(4e-4, 6e-4, 5e-6, 15e-6, 0, π, 0.9, 1.1);


julia> calculatesignal_linearapprox(ones(100)*π/2, ones(100)*5e-4, 4e-3, 0, 1, 0.1, 1, 15, 30, 10e-6, R2slT)
100×1×1 Array{ComplexF64, 3}:
[:, :, 1] =
  -0.02534278046134154 - 6.748418787451518e-18im
  0.003747524854957436 + 3.4325653058017187e-18im
   -0.0196833907083505 - 2.8004553842504324e-18im
  0.007162011566093652 + 3.672271808540125e-19im
 -0.014489517679344808 + 4.801739729077009e-19im
  0.010305804983291571 - 2.1233287976034503e-18im
 -0.009686634863202483 + 3.1882740494855564e-18im
   0.01322536234967787 - 4.1197910372278614e-18im
 -0.005228086823125368 + 5.405192374297665e-18im
  0.015948493046190188 - 5.693388849090533e-18im
                       ⋮
   0.05320472714326833 + 1.3125707519115202e-18im
   0.05258500575198769 + 4.900122476070182e-18im
  0.053374187025433915 + 1.4500192394684467e-18im
   0.05280280912207337 + 4.790222663609598e-18im
   0.05353272287002102 + 1.5786388776856578e-18im
   0.05300592940528556 + 4.687377647481522e-18im
   0.05368103548081201 + 1.6989211160502906e-18im
   0.05319535891146354 + 4.591205423715975e-18im
   0.05381978096985631 + 1.811340158316642e-18im
```
"""
function calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT; grad_list=[undef], rfphase_increment=[π], m0=:antiperiodic, output=:complexsignal)

    R2s_vec = evaluate_R2sl_vector(α, TRF, B1, T2s, R2slT, grad_list)

    return calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec; grad_list=grad_list, rfphase_increment=rfphase_increment, m0=m0, output=output)
end

function calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, R2s_vec; grad_list=[undef], rfphase_increment=[π], m0=:antiperiodic, output=:complexsignal)

    ω1 = α ./ TRF

    if output == :complexsignal && grad_list == [undef]
        S = Array{ComplexF64}(undef, length(ω1), length(rfphase_increment), 1)
        jj = 1
    elseif output == :complexsignal
        S = Array{ComplexF64}(undef, length(ω1), length(rfphase_increment), 1 + length(grad_list))
        jj = [[1,j+1] for j=1:length(grad_list)]
    elseif output == :realmagnetization && grad_list == [undef]
        S = Array{SVector{6,Float64}}(undef, length(ω1), length(rfphase_increment), length(grad_list))
        jj = 1
    elseif output == :realmagnetization
        S = Array{SVector{11,Float64}}(undef, length(ω1), length(rfphase_increment), length(grad_list))
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

    for j in eachindex(grad_list), i in eachindex(rfphase_increment)
        if m0 == :antiperiodic
            m = antiperiodic_boundary_conditions_linear(ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, R2s_vec[1], R2s_vec[2], R2s_vec[3], grad_list[j], rfphase_increment[i])
        elseif m0 == :thermal || isa(m0, AbstractVector)
            # this implements the α[1]/2 - TR/2 preparation (TR/2 is implemented in propagate_magnetization_linear!)
            u_pr = exp(hamiltonian_linear(ω1[1]/2, B1, ω0, TRF[1], m0s, R1, R2f, Rx, R2s_vec[1][1], R2s_vec[2][1], R2s_vec[3][1], grad)) # R2sl is actually wrong for the prep pulse
            m = u_pr * _m0
        elseif m0 == :IR
            # this implements the π - spoiler - TR - α[2]/2 - TR/2 preparation (TR/2 is implemented in propagate_magnetization_linear!)
            u_ip = propagator_linear_inversion_pulse(ω1[1], TRF[1], B1, R2s_vec[1][1], R2s_vec[2][1], R2s_vec[3][1], grad)
            m = u_ip * _m0
            u_fp = exp(hamiltonian_linear(0, B1, ω0, TR, m0s, R1, R2f, Rx, R2s_vec[1][1], R2s_vec[2][1], R2s_vec[3][1], grad))
            m = u_fp * m
            u_pr = exp(hamiltonian_linear(ω1[2]/2, B1, ω0, TRF[2], m0s, R1, R2f, Rx, R2s_vec[1][2], R2s_vec[2][2], R2s_vec[3][2], grad)) # R2sl is actually wrong for the prep pulse
            m = u_pr * m
        end

        Mij = @view S[:, i, jj[j]]
        propagate_magnetization_linear!(Mij, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, R2s_vec[1], R2s_vec[2], R2s_vec[3], rfphase_increment[i], m, grad_list[j])
    end
    return S
end

############################################################################
# helper functions
############################################################################
function antiperiodic_boundary_conditions_linear(ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, grad=undef, rfphase_increment=π)
    # put inversion pulse at the end (this defines m as the magnetization at the first TE after the inversion pulse)
    u_rot = z_rotation_propagator(rfphase_increment, grad)
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

function propagate_magnetization_linear!(S, ω1, B1, ω0, TRF, TR, m0s, R1, R2f, Rx, _R2s, _dR2sdT2s, _dR2sdB1, rfphase_increment, m, grad=undef)

    u_rot = z_rotation_propagator(rfphase_increment, grad)
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