
"""
    qM = fit_gBloch(data, α, TRF, TR;
        reM0 = (-Inf,   1,  Inf),
        imM0 = (-Inf,   0,  Inf),
        m0s  = (   0, 0.2,    1),
        R1f  = (   0, 0.3,  Inf),
        R2f  = (   0,  15,  Inf),
        Rex  = (   0,  20,  Inf),
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
- `α::Vector{Real}`: Array of flip angles in radians; can also be a `Vector{Vector{Real}}` which simulates each RF pattern and concatenates the signals of each simulation
- `TRF::Vector{Real}`: Array of the RF-pulse durations in seconds (or `Vector{Vector{Real}}` if `α::Vector{Vector{Real}}``)
- `TR::Real`: Repetition time in seconds
- `ω0::Real`: Off-resonance frequency in rad/s

# Optional Keyword Arguments:
- `reM0::Union{Real, Tuple{Real, Real, Real}}`: Real part of `M0`; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `imM0::Union{Real, Tuple{Real, Real, Real}}`: Imaginary part of `M0`; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `m0s::Union{Real, Tuple{Real, Real, Real}}`: Fractional size of the semi-solid pool (should be in range of 0 to 1); either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `R1f::Union{Real, Tuple{Real, Real, Real}}`: Longitudinal relaxation rate of the free pool in 1/s; only used in combination with `fit_apparentR1=false`; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `R2f::Union{Real, Tuple{Real, Real, Real}}`: Transversal relaxation rate of the free pool in 1/s; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
- `Rex::Union{Real, Tuple{Real, Real, Real}}`: Exchange rate between the two spin pools in 1/s; either fixed value as a `Real` or fit limits thereof as a `Tuple` with the elements `(min, start, max)`
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
function fit_gBloch(data, α::Vector{T}, TRF::Vector{T}, TR; grad_moment = [i == 1 ? :spoiler_dual : :balanced for i ∈ eachindex(α)],
    reM0    = (-Inf,   1,  Inf),
    imM0    = (-Inf,   0,  Inf),
    m0_M    = (   0, 0.2,    1),
    m0_MW   = (   0, 0.2,    1),
    m0_NM   = (   0, 0.2,    1),
    Rx_IEW_MW = (   0,  20,  Inf),
    Rx_M_MW   = (   0,  20,  Inf),
    Rx_IEW_NM = (   0,  20,  Inf),
    R1_M   = (   0, 0.3,  Inf),
    R1_NM  = (   0, 0.3,  Inf),
    R1_IEW = (   0, 0.3,  Inf),
    R1_MW  = (   0, 0.3,  Inf),
    R2_IEW = (   0,  15,  Inf),
    R2_MW  = (   0,  15,  Inf),
    T2_M  = (8e-6,1e-5,12e-6),
    T2_NM = (8e-6,1e-5,12e-6),
    ω0   = (-Inf,   0,  Inf),
    B1   = (   0,   1,  1.5),
    u=1,
    show_trace=false,
    maxIter=100,
    R2slT = precompute_R2sl(TRF_min=minimum(TRF), TRF_max=maximum(TRF), T2s_min=minimum(T2s), T2s_max=maximum(T2s), ω1_max=maximum(α ./ TRF), B1_max=maximum(B1)),
    ) where T <: Real

    fit_gBloch(data, [α], [TRF], TR; grad_moment=[grad_moment], reM0, imM0, m0_M, m0_MW, m0_NM, Rx_IEW_MW, Rx_M_MW, Rx_IEW_NM, R1_M, R1_IEW, R2_IEW, T2_M, ω0, B1, u, show_trace, maxIter, R2slT)
end

function fit_gBloch(data, α::Vector{Vector{T}}, TRF::Vector{Vector{T}}, TR; grad_moment = fill([i == 1 ? :spoiler_dual : :balanced for i ∈ eachindex(α[1])], length(α)),
    reM0    = (-Inf,   1,  Inf),
    imM0    = (-Inf,   0,  Inf),
    m0_M    = (   0, 0.2,    1),
    m0_MW   = (   0, 0.2,    1),
    m0_NM   = (   0, 0.2,    1),
    Rx_IEW_MW = (   0,  20,  Inf),
    Rx_M_MW   = (   0,  20,  Inf),
    Rx_IEW_NM = (   0,  20,  Inf),
    R1_M   = (   0, 0.3,  Inf),
    R1_NM  = (   0, 0.3,  Inf),
    R1_IEW = (   0, 0.3,  Inf),
    R1_MW  = (   0, 0.3,  Inf),
    R2_IEW = (   0,  15,  Inf),
    R2_MW  = (   0,  15,  Inf),
    T2_M  = (8e-6,1e-5,12e-6),
    T2_NM = (8e-6,1e-5,12e-6),
    ω0   = (-Inf,   0,  Inf),
    B1   = (   0,   1,  1.5),
    u=1,
    show_trace=false,
    maxIter=100,
    R2slT = precompute_R2sl(TRF_min=minimum(minimum.(TRF)), TRF_max=maximum(maximum.(TRF)), T2s_min=minimum(T2s), T2s_max=maximum(T2s), ω1_max=maximum(maximum.(α ./ TRF)), B1_max=maximum(B1)),
    ) where T <: Real

    grad_list = MRIgeneralizedBloch.grad_param[]
    pmin = Float64[reM0[1], imM0[1]]
    p0   = Float64[reM0[2], imM0[2]]
    pmax = Float64[reM0[3], imM0[3]]

    idx = Vector{Int}(undef, 16)
    param    = [m0_M, m0_MW, m0_NM, Rx_IEW_MW, Rx_M_MW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_IEW, R2_MW, T2_M, T2_NM, ω0, B1]
    grad_all = [grad_m0_M(), grad_m0_MW(),grad_m0_NM(), grad_Rx_MW_IEW(), grad_Rx_M_MW(), grad_Rx_IEW_NM(), grad_R1_M(), grad_R1_NM(), grad_R1_IEW(), grad_R1_MW(), grad_R2_IEW(), grad_R2_MW(), grad_T2_M(), grad_T2_NM(), grad_ω0(), grad_B1()]


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
        M0, m0_M, m0_MW, m0_NM, Rx_IEW_MW, Rx_M_MW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_IEW, R2_MW, T2_M, T2_NM,ω0, B1 = getparameters(p)

        m = Vector{Array{ComplexF64}}(undef,length(α))
        Threads.@threads for i ∈ eachindex(α)
            m[i] = vec(calculatesignal_linearapprox(α[i], TRF[i], TR[i], ω0, B1, m0_M, m0_NM, m0_MW, Rx_M_MW, Rx_IEW_MW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, T2_M, T2_NM, R2slT, R2slT))
        end
        m = reduce(vcat,m)

        m .*= M0
        m = u' * m
        F[1:end÷2]     .= real.(m)
        F[end÷2+1:end] .= imag.(m)
        return F
    end

    function jacobian!(J, _, p)
        M0, m0_M, m0_MW, m0_NM, Rx_IEW_MW, Rx_M_MW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_IEW, R2_MW, T2_M, T2_NM, ω0, B1 = getparameters(p)

        M = Vector{Array{ComplexF64}}(undef,length(α))
        Threads.@threads for i ∈ eachindex(α)
            M[i] = dropdims(calculatesignal_linearapprox(α[i], TRF[i], TR[i], ω0, B1, m0_M, m0_NM, m0_MW, Rx_M_MW, Rx_IEW_MW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, T2_M, T2_NM, R2slT, R2slT; grad_list),dims=2)
        end
        M = reduce(vcat,M)

        M[:,2:end] .*= M0
        M = u' * M

        J[:,1]     = [real(M[:,1]); imag(M[:,1])]
        J[:,2]     = [-imag(M[:,1]); real(M[:,1])]
        J[:,3:end] = [real(M[:,2:end]); imag(M[:,2:end])]
        return J
    end

    result = curve_fit(model!, jacobian!, 1:(2*size(u,2)), [real(data); imag(data)], p0, lower=pmin, upper=pmax, show_trace=show_trace, maxIter=maxIter, inplace=true)

    M0, m0_M, m0_MW, m0_NM, Rx_IEW_MW, Rx_M_MW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_IEW, R2_MW, T2_M, T2_NM, ω0, B1 = getparameters(result.param)

    return qMTparam(M0, m0_M, m0_MW, m0_NM, Rx_IEW_MW, Rx_M_MW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_IEW, R2_MW, T2_M, T2_NM, ω0, B1, norm(result.resid), result)
end

#########################################################################
# Type definitions etc.
#########################################################################
struct qMTparam
    M0
    m0_M
    m0_MW
    m0_NM
    Rx_IEW_MW
    Rx_M_MW
    Rx_IEW_NM
    R1_M
    R1_NM
    R1_IEW
    R1_MW
    R2_IEW
    R2_MW
    T2_M
    T2_NM
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
        A.M0[i...],
        A.m0_M[i...],
        A.m0_MW[i...],
        A.m0_NM[i...],
        A.Rx_IEW_MW[i...],
        A.Rx_M_MW[i...],
        A.Rx_IEW_NM[i...],
        A.R1_M[i...],
        A.R1_NM[i...],
        A.R1_IEW[i...],
        A.R1_MW[i...],
        A.R2_IEW[i...],
        A.R2_MW[i...],
        A.T2_M[i...],
        A.T2_NM[i...],
        A.ω0[i...],
        A.B1[i...],
        A.resid[i...]        ,
        A.lsqfit_result[i...])
end

function Base.setindex!(A::qMTparam, v::qMTparam, i...)
    A.M0[i...]            = v.M0
    A.m0_M[i...]          = v.m0_M
    A.m0_MW[i...]         = v.m0_MW
    A.m0_NM[i...]         = v.m0_NM
    A.Rx_IEW_MW[i...]    = v.Rx_IEW_MW
    A.Rx_M_MW[i...]      = v.Rx_M_MW
    A.Rx_IEW_NM[i...]    = v.Rx_IEW_NM
    A.R1_M[i...]          = v.R1_M
    A.R1_NM[i...]         = v.R1_NM
    A.R1_IEW[i...]        = v.R1_IEW
    A.R1_MW[i...]         = v.R1_MW
    A.R2_IEW[i...]        = v.R2_IEW
    A.R2_MW[i...]         = v.R2_MW
    A.T2_M[i...]          = v.T2_M
    A.T2_NM[i...]         = v.T2_NM
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