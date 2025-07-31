####################################################################
# Bounded ω1 and TRF
####################################################################
"""
    ω1, TRF = get_bounded_ω1_TRF(x)

Transform a vector of length `2Npulses` with values in the range `[-Inf, Inf]` into two vectors of length `Npulses`, which describe the bounded controls `ω1` and `TRF`.

# Arguments
- `x::Vector{Real}`: Control vector of `length = 2Npulses` with values in the range `[-Inf, Inf]`

# Optional Keyword Arguments:
- `ω1_min::Vector{Real}`: elementwise lower bound for ω1 in rad/s
- `ω1_max::Vector{Real}`: elementwise upper bound for ω1 in rad/s
- `TRF_min::Vector{Real}`: elementwise lower bound for TRF in s
- `TRF_max::Vector{Real}`: elementwise bound for TRF in s

# Examples
```jldoctest
julia> x = repeat(range(-1000.0, 1000.0, 100), 2);

julia> ω1, TRF = MRIgeneralizedBloch.get_bounded_ω1_TRF(x)
([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  6283.185307179586, 6283.185307179586, 6283.185307179586, 6283.185307179586, 6283.185307179586, 6283.185307179586, 6283.185307179586, 6283.185307179586, 6283.185307179586, 6283.185307179586], [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001  …  0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005])
```
"""
function get_bounded_ω1_TRF(x; NSeq=1, ω1_min=zeros(length(x) ÷ 2NSeq, NSeq), ω1_max=fill(2e3π, length(x) ÷ 2NSeq, NSeq), TRF_min=fill(100e-6, length(x) ÷ 2NSeq, NSeq), TRF_max=fill(500e-6, length(x) ÷ 2NSeq, NSeq))
    ω1  = @views  ω1_min .+ ( ω1_max .-  ω1_min) .* (1 .+ reshape(tanh.(x[1:end÷2]), :, NSeq)) ./ 2
    TRF = @views TRF_min .+ (TRF_max .- TRF_min) .* (1 .+ reshape(tanh.(x[end÷2+1:end]), :, NSeq)) ./ 2

    if NSeq == 1
        ω1 = vec(ω1)
        TRF = vec(TRF)
    end

    return ω1, TRF
end


"""
    x = bound_ω1_TRF!(ω1, TRF; ω1_min=zeros(size(ω1)), ω1_max=fill(2e3π, size(ω1)), TRF_min=fill(100e-6, size(TRF)), TRF_max=fill(500e-6, size(TRF)))

Bound the controls `ω1` and `TRF` (over-written in place) and return a vector of length `2Npulses * NSeq` with values in the range `[-Inf, Inf]` that relate to the bounded `ω1` and `TRF` via the `tanh` function.

# Arguments
- `ω1`: Control vector of length `Npulses` or matrix with the number of sequences in the second dimension
- `TRF`: Control vector of length `Npulses` or matrix with the number of sequences in the second dimension

# Optional Keyword Arguments:
- `ω1_min`: elementwise lower bound for ω1 in rad/s
- `ω1_max`: elementwise upper bound for ω1 in rad/s
- `TRF_min`: elementwise lower bound for TRF in s
- `TRF_max`: elementwise bound for TRF in s

# Examples
```jldoctest
julia> ω1 = collect(range(0, 2000π, 100));

julia> TRF = collect(range(100e-6, 500e-6, 100));

julia> x = MRIgeneralizedBloch.bound_ω1_TRF!(ω1, TRF)
200-element Vector{Float64}:
 -Inf
  -2.2924837393352853
  -1.9407818989717183
  -1.7328679513998637
  -1.5837912652403254
  -1.4669284349179519
  -1.3704200119626004
  -1.2879392139968635
  -1.2157089824185072
  -1.151292546497023
   ⋮
   1.2157089824185072
   1.2879392139968635
   1.3704200119626009
   1.4669284349179519
   1.5837912652403245
   1.7328679513998637
   1.9407818989717212
   2.2924837393352826
  Inf
```
"""
function bound_ω1_TRF!(ω1, TRF; ω1_min=zeros(size(ω1)), ω1_max=fill(2e3π, size(ω1)), TRF_min=fill(100e-6, size(TRF)), TRF_max=fill(500e-6, size(TRF)))
    ω1 .= min.(ω1, ω1_max)
    ω1 .= max.(ω1, ω1_min)

    TRF .= min.(TRF, TRF_max)
    TRF .= max.(TRF, TRF_min)

    x = similar(ω1, length(ω1) + length(TRF))
    x_ω1  = @views reshape(x[1:length(ω1)], size(ω1))
    x_TRF = @views reshape(x[length(ω1)+1:end], size(TRF))

    x_ω1 .= atanh.((ω1 .- ω1_min) .* 2 ./ (ω1_max .- ω1_min) .- 1)
    x_TRF .= atanh.((TRF .- TRF_min) .* 2 ./ (TRF_max .- TRF_min) .- 1)
    return x
end

function apply_bounds_to_grad!(G, x, grad_ω1, grad_TRF; ω1_min=zeros(size(grad_ω1)), ω1_max=fill(2e3π, size(grad_ω1)), TRF_min=fill(100e-6, size(grad_TRF)), TRF_max=fill(500e-6, size(grad_TRF)))
    G_ω1  = @views reshape(G[1:length(grad_ω1)], size(grad_ω1))
    G_TRF = @views reshape(G[length(grad_ω1)+1:end], size(grad_TRF))
    G_ω1 .= grad_ω1 .* (ω1_max .- ω1_min) ./ 2
    G_TRF .= grad_TRF .* (TRF_max .- TRF_min) ./ 2

    G .*= sech.(x) .^ 2
    return G
end


####################################################################
# Penalties on the flip angle α
####################################################################
"""
    F = second_order_α!(grad_ω1, grad_TRF, ω1, TRF; λ=1, grad_moment=[i[1] == 1 ? :spoiler_dual : :balanced for i ∈ CartesianIndices(ω1)])

Calculate second order penalty of variations of the flip angle α and adds in place to the gradients.

# Arguments
- `grad_ω1::Vector{Real}`: Gradient of control, which will be added in place (matrix if more than 1 sequence are optimized)
- `grad_TRF::Vector{Real}`: Gradient of control, which will be added in place (matrix if more than 1 sequence are optimized)
- `ω1::Vector{Real}`: Control vector (matrix if more than 1 sequence are optimized)
- `TRF::Vector{Real}`: Control vector (matrix if more than 1 sequence are optimized)

# Optional Keyword Arguments:
- `λ::Real`: regularization parameter
- `grad_moment = [i[1] == 1 ? :spoiler_dual : :balanced for i ∈ CartesianIndices(ω1)]`: Different types of gradient moments of each TR are possible (:balanced, :spoiler_dual, :crusher). Skip :crusher and :spoiler_dual TRs second order penalty

# Examples
```jldoctest
julia> ω1 = range(0, 2000π, 100);

julia> TRF = range(100e-6, 500e-6, 100);

julia> grad_ω1 = similar(ω1);

julia> grad_TRF = similar(ω1);

julia> F = MRIgeneralizedBloch.second_order_α!(grad_ω1, grad_TRF, ω1, TRF; λ = 1e-3)
0.005015194549476384
```
"""
function second_order_α!(grad_ω1, grad_TRF, ω1, TRF; λ=1, grad_moment=[i[1] == 1 ? :spoiler_dual : :balanced for i ∈ CartesianIndices(ω1)])
    α = ω1 .* TRF

    F = 0
    for iSeq ∈ axes(ω1, 2)
        idx = @views grad_moment[:, iSeq] .== :balanced
        T = sum(idx)

        αi = @view α[idx, iSeq]
        ω1i = @view ω1[idx, iSeq]
        TRFi = @view TRF[idx, iSeq]

        F += (αi[end-1] / 2 - αi[end] + αi[1] / 2)^2
        F += (αi[end] / 2 - αi[1] + αi[2] / 2)^2
        for t = 2:T-1
            F += (αi[t-1] / 2 - αi[t] + αi[t+1] / 2)^2
        end

        if grad_ω1 !== nothing
            grad_ω1i = @view grad_ω1[idx, iSeq]
            grad_ω1i[end-1] += λ * (αi[end-1] / 2 - αi[end] + αi[1] / 2) * TRFi[end-1]
            grad_ω1i[end] -= 2λ * (αi[end-1] / 2 - αi[end] + αi[1] / 2) * TRFi[end]
            grad_ω1i[1] += λ * (αi[end-1] / 2 - αi[end] + αi[1] / 2) * TRFi[1]

            grad_ω1i[end] += λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * TRFi[end]
            grad_ω1i[1] -= 2λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * TRFi[1]
            grad_ω1i[2] += λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * TRFi[2]

            for t = 2:T-1
                grad_ω1i[t-1] += λ * (αi[t-1] / 2 - αi[t] + αi[t+1] / 2) * TRFi[t-1]
                grad_ω1i[t] -= 2λ * (αi[t-1] / 2 - αi[t] + αi[t+1] / 2) * TRFi[t]
                grad_ω1i[t+1] += λ * (αi[t-1] / 2 - αi[t] + αi[t+1] / 2) * TRFi[t+1]
            end
        end

        if grad_TRF !== nothing
            grad_TRFi = @view grad_TRF[idx, iSeq]
            grad_TRFi[end-1] += λ * (αi[end-1] / 2 - αi[end] + αi[1] / 2) * ω1i[end-1]
            grad_TRFi[end] -= 2λ * (αi[end-1] / 2 - αi[end] + αi[1] / 2) * ω1i[end]
            grad_TRFi[1] += λ * (αi[end-1] / 2 - αi[end] + αi[1] / 2) * ω1i[1]

            grad_TRFi[end] += λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * ω1i[end]
            grad_TRFi[1] -= 2λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * ω1i[1]
            grad_TRFi[2] += λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * ω1i[2]

            for t = 2:T-1
                grad_TRFi[t-1] += λ * (αi[t-1] / 2 - αi[t] + αi[t+1] / 2) * ω1i[t-1]
                grad_TRFi[t] -= 2λ * (αi[t-1] / 2 - αi[t] + αi[t+1] / 2) * ω1i[t]
                grad_TRFi[t+1] += λ * (αi[t-1] / 2 - αi[t] + αi[t+1] / 2) * ω1i[t+1]
            end
        end
    end

    return λ * F
end


####################################################################
# Penalty on TRF
####################################################################
"""
    F = TRF_TV!(grad_TRF, TRF; λ=1, grad_moment=[i[1] == 1 ? :spoiler_dual : :balanced for i ∈ CartesianIndices(TRF)])

Calculate the total variation penalty of `TRF` and add to `grad_TRF` in place.

# Arguments
- `grad_TRF::Vector{Real}`: Gradient of control, which will added in place (matrix if more than 1 sequence are optimized)
- `TRF::Vector{Real}`: Control vector (matrix if more than 1 sequence are optimized)

# Optional Keyword Arguments:
- `λ::Real`: regularization parameter
- `grad_moment = [i[1] == 1 ? :spoiler_dual : :balanced for i ∈ CartesianIndices(TRF)]`: Different types of gradient moments of each TR are possible (:balanced, :spoiler_dual, :crusher). Skip :crusher and :spoiler_dual TRs for the TRF TV penalty

# Examples
```jldoctest
julia> TRF = range(100e-6, 500e-6, 100);

julia> grad_TRF = similar(TRF);

julia> F = MRIgeneralizedBloch.TRF_TV!(grad_TRF, TRF; λ = 1e-3)
3.9595959595959597e-7
```
"""
function TRF_TV!(grad_TRF, TRF; λ=1, grad_moment=[i[1] == 1 ? :spoiler_dual : :balanced for i ∈ CartesianIndices(TRF)])
    idx = grad_moment .== :balanced

    F = 0
    for iα ∈ axes(TRF, 1), iSeq ∈ axes(TRF, 2)
        jα = mod1(iα + 1, size(TRF, 1))
        if idx[iα, iSeq] && idx[jα, iSeq]
            F += abs(TRF[jα, iSeq] - TRF[iα, iSeq])
        end
    end

    if grad_TRF !== nothing
        for iα ∈ axes(TRF, 1), iSeq ∈ axes(TRF, 2)
            jα = mod1(iα + 1, size(TRF, 1))
            if idx[iα, iSeq] && idx[jα, iSeq]
                grad_TRF[iα, iSeq] -= λ * sign(TRF[jα, iSeq] - TRF[iα, iSeq])
                grad_TRF[jα, iSeq] += λ * sign(TRF[jα, iSeq] - TRF[iα, iSeq])
            end
        end
    end
    return λ * F
end

####################################################################
# SAR Penalty
####################################################################
"""
    F = RF_power!(grad_ω1, grad_TRF, ω1, TRF; λ=1, Pmax=3e6, TR=3.5e-3)

Calculate RF power penalty and add the gradients in place.

# Arguments
- `grad_ω1::Vector{Real}`: Gradient of control, which will be added in place (matrix if more than 1 sequence are optimized)
- `grad_TRF::Vector{Real}`: Gradient of control, which will be added in place (matrix if more than 1 sequence are optimized)
- `ω1::Vector{Real}`: Control vector (matrix if more than 1 sequence are optimized)
- `TRF::Vector{Real}`: Control vector (matrix if more than 1 sequence are optimized)

# Optional Keyword Arguments:
- `λ::Real`: regularization parameter
- `Pmax::Real`: Maximum average power deposition in (rad/s)²; everything above this value will be penalized and with an appropriate λ, the resulting power will be equal to or less than this value.
- `TR::Real`: Repetition time of the pulse sequence

# Examples
```jldoctest
julia> ω1 = range(0, 4000π, 100);

julia> TRF = range(100e-6, 500e-6, 100);

julia> grad_ω1 = similar(ω1);

julia> grad_TRF = similar(ω1);

julia> F = MRIgeneralizedBloch.RF_power!(grad_ω1, grad_TRF, ω1, TRF; λ=1e3, Pmax=3e6, TR=3.5e-3)
9.418321886730644e15
```
"""
function RF_power!(grad_ω1, grad_TRF, ω1, TRF; λ=1, Pmax=3e6, TR=3.5e-3)
    Nα = size(ω1, 1)

    P(x) = x < 0 ? 0 : λ * x^2
    dPdx(x) = x < 0 ? 0 : 2λ * x

    c_ΔSAR = zero(eltype(ω1))
    for iSeq ∈ axes(grad_ω1, 2)
        @views ΔSAR = sum(ω1[:, iSeq] .^ 2 .* TRF[:, iSeq]) / (Nα * TR) - Pmax
        c_ΔSAR += P(ΔSAR)

        if grad_ω1 !== nothing
            @views grad_ω1[:, iSeq] .+= dPdx(ΔSAR) .* 2ω1[:, iSeq] .* TRF[:, iSeq] ./ (Nα * TR)
        end
        if grad_TRF !== nothing
            @views grad_TRF[:, iSeq] .+= dPdx(ΔSAR) .* ω1[:, iSeq] .^ 2 ./ (Nα * TR)
        end
    end
    return c_ΔSAR
end