
"""
    qM = fit_gBloch(data, α, TRF, TR;
        reM0 = (-Inf,   1,  Inf),
        imM0 = (-Inf,   0,  Inf),
        m0s  = (   0, 0.2,    1),
        R1f  = (   0, 0.3,  Inf),
        R2f  = (   0,  15,  Inf),
        Rx   = (   0,  20,  Inf),
        R1s  = (   0,   3,  Inf),
        T2s  = (8e-6,1e-5,12e-6),
        ω0   = (-Inf,   0,  Inf),
        B1   = (   0,   1,  1.5),
        R1a  = (   0, 0.7,  Inf),
        u=1,
        fit_apparentR1=false,
        show_trace=false,
        maxIter=100,
        R2slT = precompute_R2sl(TRF_min=minimum(TRF), TRF_max=maximum(TRF), T2s_min=minimum(T2s), T2s_max=maximum(T2s), ω1_max=maximum(α ./ TRF), B1_max=maximum(B1)),
        )

Fit the generalized Bloch model for a train of RF pulses and balanced gradient moments to `data`.

# Arguments
- `data::Vector{Number}`: Array of measured data points, either in the time or a compressed domain (cf. `u`)
- `α::Vector{Real}`: Array of flip angles in radians
- `TRF::Vector{Real}`: Array of the RF-pulse durations in seconds
- `TR::Real`: Repetition time in seconds
- `ω0::Real`: Off-resonance frequency in rad/s

# Optional Keyword Arguments:
- `reM0::Union{Real, Tuple{Real, Real, Real}}`: Real part of `M0`; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `imM0::Union{Real, Tuple{Real, Real, Real}}`: Imaginary part of `M0`; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `m0s::Union{Real, Tuple{Real, Real, Real}}`: Fractional size of the semi-solid pool (should be in range of 0 to 1); either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `R1f::Union{Real, Tuple{Real, Real, Real}}`: Longitudinal relaxation rate of the free pool in 1/s; only used in combination with `fit_apparentR1=false`; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `R2f::Union{Real, Tuple{Real, Real, Real}}`: Transversal relaxation rate of the free pool in 1/s; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `Rx::Union{Real, Tuple{Real, Real, Real}}`: Exchange rate between the two spin pools in 1/s; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `R1s::Union{Real, Tuple{Real, Real, Real}}`: Longitudinal relaxation rate of the semi-solid pool in 1/s; only used in combination with `fit_apparentR1=false`; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `T2s::Union{Real, Tuple{Real, Real, Real}}`: Transversal relaxation time of the semi-solid pool in s; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `ω0::Union{Real, Tuple{Real, Real, Real}}`: Off-resonance frequency in rad/s; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `B1::Union{Real, Tuple{Real, Real, Real}}`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `R1a::Union{Real, Tuple{Real, Real, Real}}`: Apparent longitudinal relaxation rate in 1/s; only used in combination with `fit_apparentR1=true`; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `u::Union{Number, Matrix}`: Compression matrix that transform the simulated time series to a series of coefficients. Set to `1` by default to enable the fitting in the time domain
- `fit_apparentR1::Bool`: Switch between fitting `R1f` and `R1s` separately (`false`; default) and an apparent `R1a = R1f = R1s` (`true`)
- `show_trace::Bool`: print output during the optimization; `default=false`
- `maxIter::Int`: Maximum number of iteration; `default=100`
- `R2slT::NTuple{3, Function}`: Tuple of three functions: R2sl(TRF, ω1, B1, T2s), dR2sldB1(TRF, ω1, B1, T2s), and R2sldT2s(TRF, ω1, B1, T2s). By default generated with [`precompute_R2sl`](@ref)

# Examples
c.f. [Non-Linear Least Square Fitting](@ref)
"""
function fit_gBloch(data, α, TRF, TR;
    reM0 = (-Inf,   1,  Inf),
    imM0 = (-Inf,   0,  Inf),
    m0s  = (   0, 0.2,    1),
    R1f  = (   0, 0.3,  Inf),
    R2f  = (   0,  15,  Inf),
    Rx   = (   0,  20,  Inf),
    R1s  = (   0,   3,  Inf),
    T2s  = (8e-6,1e-5,12e-6),
    ω0   = (-Inf,   0,  Inf),
    B1   = (   0,   1,  1.5),
    R1a  = (   0, 0.7,  Inf),
    u=1,
    fit_apparentR1=false,
    show_trace=false,
    maxIter=100,
    R2slT = precompute_R2sl(TRF_min=minimum(TRF), TRF_max=maximum(TRF), T2s_min=minimum(T2s), T2s_max=maximum(T2s), ω1_max=maximum(α ./ TRF), B1_max=maximum(B1)),
    )

    grad_list = MRIgeneralizedBloch.grad_param[]
    pmin = Float64[reM0[1], imM0[1]]
    p0   = Float64[reM0[2], imM0[2]]
    pmax = Float64[reM0[3], imM0[3]]

    idx = Vector{Int}(undef, 8)
    if fit_apparentR1
        param    = [m0s, R1a, R2f, Rx, R1a, T2s, ω0, B1]
        grad_all = [grad_m0s(), grad_R1a(), grad_R2f(), grad_Rx(), nothing, grad_T2s(), grad_ω0(), grad_B1()]
    else
        param    = [m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1]
        grad_all = [grad_m0s(), grad_R1f(), grad_R2f(), grad_Rx(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1()]
    end
    for i ∈ eachindex(param)
        idx[i] = if isa(param[i], Number)
            0
        elseif grad_all[i] === nothing
            idx[2] # copy R1a
        else
            push!(grad_list, grad_all[i])
            push!(pmin, param[i][1])
            push!(p0  , param[i][2])
            push!(pmax, param[i][3])
            length(p0)
        end
    end

    getparameters(p) = ((p[1]+1im*p[2]), ntuple(i-> idx[i] == 0 ? param[i] : p[idx[i]], length(idx))...)

    function model!(F, _, p)
        M0, m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1 = getparameters(p)
        m = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT)
        m = vec(m)
        m .*= M0
        m = u' * m
        F[1:end÷2]     .= real.(m)
        F[end÷2+1:end] .= imag.(m)
        return F
    end

    function jacobian!(J, _, p)
        M0, m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1 = getparameters(p)
        M = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=grad_list)
        M = dropdims(M, dims=2)
        M[:,2:end] .*= M0
        M = u' * M

        J[:,1]     = [real(M[:,1]); imag(M[:,1])]
        J[:,2]     = [-imag(M[:,1]); real(M[:,1])]
        J[:,3:end] = [real(M[:,2:end]); imag(M[:,2:end])]
        return J
    end

    result = curve_fit(model!, jacobian!, 1:(2*size(u,2)), [real(data); imag(data)], p0, lower=pmin, upper=pmax, show_trace=show_trace, maxIter=maxIter, inplace=true)

    M0, m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1 = getparameters(result.param)

    return qMTparam(M0, m0s, R1f, R2f, Rx, R1s, T2s, ω0, B1, norm(result.resid), result)
end

#########################################################################
# Type definitions etc.
#########################################################################
struct qMTparam
    M0
    m0s
    R1f
    R2f
    Rx
    R1s
    T2s
    ω0
    B1
    resid
    lsqfit_result
end

function qMTmap(N...)
    qMTparam(
        zeros(ComplexF64, N...),
        zeros(Float64, N...),
        zeros(Float64, N...),
        zeros(Float64, N...),
        zeros(Float64, N...),
        zeros(Float64, N...),
        zeros(Float64, N...),
        zeros(Float64, N...),
        zeros(Float64, N...),
        zeros(Float64, N...),
        Array{LsqFit.LsqFitResult}(undef, N...)
        )
end

function Base.getindex(A::qMTparam, i...)
    return qMTparam(
        A.M0[i...]           ,
        A.m0s[i...]          ,
        A.R1f[i...]          ,
        A.R2f[i...]          ,
        A.Rx[i...]           ,
        A.R1s[i...]          ,
        A.T2s[i...]          ,
        A.ω0[i...]           ,
        A.B1[i...]           ,
        A.resid[i...]        ,
        A.lsqfit_result[i...])
end

function Base.setindex!(A::qMTparam, v::qMTparam, i...)
    A.M0[i...]            = v.M0
    A.m0s[i...]           = v.m0s
    A.R1f[i...]           = v.R1f
    A.R2f[i...]           = v.R2f
    A.Rx[i...]            = v.Rx
    A.R1s[i...]           = v.R1s
    A.T2s[i...]           = v.T2s
    A.ω0[i...]            = v.ω0
    A.B1[i...]            = v.B1
    A.resid[i...]         = v.resid
    A.lsqfit_result[i...] = v.lsqfit_result
    return A
end

function Base.length(A::qMTparam)
    return length(A.m0s)
end

function Base.iterate(A::qMTparam, state=(eachindex(A.M0),))
    y = iterate(state...)
    y === nothing && return nothing
    A[y[1]], (state[1], Base.tail(y)...)
end

function Base.ndims(A::qMTparam)
    return Base.ndims(A.M0)
end

function Base.ndims(::Type{qMTparam})
    return 0
end