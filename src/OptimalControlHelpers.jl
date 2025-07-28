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
julia> x = 1000 * randn(2 * 100);

julia> ω1, TRF = MRIgeneralizedBloch.get_bounded_ω1_TRF(x)
([0.0, 6283.185307179586, 0.0, 0.0, 6283.185307179586, 6283.185307179586, 6283.185307179586, 6283.185307179586, 0.0, 0.0  …  0.0, 0.0, 6283.185307179586, 6283.185307179586, 6283.185307179586, 6283.185307179586, 0.0, 0.0, 1.4115403811440933e-9, 0.0], [0.0005, 0.0001, 0.0001, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0001, 0.0005  …  0.0005, 0.0005, 0.0001, 0.0001, 0.0001, 0.0004999999277453363, 0.0005, 0.0001, 0.0001, 0.0001])
```
"""
function get_bounded_ω1_TRF(x; ω1_min  =          zeros(length(x)÷2), 
                               ω1_max  = 2e3π   *  ones(length(x)÷2), 
                               TRF_min = 100e-6 *  ones(length(x)÷2), 
                               TRF_max = 500e-6 *  ones(length(x)÷2))

    ω1  =  ω1_min .+ ( ω1_max .-  ω1_min) .* (1 .+ tanh.(x[1:end ÷ 2]      )) ./ 2
    TRF = TRF_min .+ (TRF_max .- TRF_min) .* (1 .+ tanh.(x[end ÷ 2 + 1:end])) ./ 2

    return (ω1, TRF)
end


"""
    x = bound_ω1_TRF!(ω1, TRF; ω1_min = 0, ω1_max = 2e3π, TRF_min = 100e-6, TRF_max = 500e-6)

Bound the controls `ω1` and `TRF` (over-written in place) and return a vector of length `2Npulses` with values in the range `[-Inf, Inf]` that relate to the bounded `ω1` and `TRF` via the `tanh` function.

# Arguments
- `ω1::Vector{Real}`: Control vector of length `Npulses`
- `TRF::Vector{Real}`: Control vector of length `Npulses`

# Optional Keyword Arguments (see above for defaults):
- `ω1_min::Real`: lower bound for ω1 in rad/s
- `ω1_max::Real`: upper bound for ω1 in rad/s
- `TRF_min::Real`: lower bound for TRF in s
- `TRF_max::Real`: upper bound for TRF in s

# Examples
```jldoctest
julia> ω1 = 4000π * rand(100);

julia> TRF = 500e-6 * rand(100);

julia> x = MRIgeneralizedBloch.bound_ω1_TRF!(ω1, TRF)
200-element Vector{Float64}:
  Inf
   0.3084393995336273
  -0.222010414402936
   0.1981882670715111
  Inf
  -0.3590504865635206
  Inf
  Inf
  Inf
  Inf
   ⋮
  -1.3069326972020265
 -Inf
   4.449542946745821
   0.06378289522574039
   0.2699178281115771
   1.271412541641303
  -0.46991502137668045
  -1.4377324062335655
  -0.9746661706753423
```
"""

function bound_ω1_TRF!(ω1, TRF; ω1_min  =          zeros(length(x)÷2), 
                                ω1_max  = 2e3π   *  ones(length(x)÷2), 
                                TRF_min = 100e-6 *  ones(length(x)÷2), 
                                TRF_max = 500e-6 *  ones(length(x)÷2))
    ω1 .= min.(ω1, ω1_max)
    ω1 .= max.(ω1, ω1_min)

    TRF .= min.(TRF, TRF_max)
    TRF .= max.(TRF, TRF_min)

    x_ω1  = similar(ω1,  length(ω1) )
    x_TRF = similar(TRF, length(TRF))

    x_ω1  .= atanh.(( ω1 .-  ω1_min) .* 2 ./ ( ω1_max .-  ω1_min) .- 1)
    x_TRF .= atanh.((TRF .- TRF_min) .* 2 ./ (TRF_max .- TRF_min) .- 1)

    return [x_ω1;x_TRF]
end

function apply_bounds_to_grad!(G, x, grad_ω1, grad_TRF; fixed_α = [true; falses(length(x)÷ 2 -1)], ω1_min  =          zeros(length(x)÷2), 
                                                                                                   ω1_max  = 2e3π   *  ones(length(x)÷2), 
                                                                                                   TRF_min = 100e-6 *  ones(length(x)÷2), 
                                                                                                   TRF_max = 500e-6 *  ones(length(x)÷2))

    _, TRF = MRIgeneralizedBloch.get_bounded_ω1_TRF(x; ω1_min, ω1_max, TRF_min, TRF_max)
   
    G_TRF = similar(grad_TRF)
    G_ω1  = similar(grad_ω1)
    

    G_TRF[.~fixed_α] .=  grad_TRF[.~fixed_α]                                              .* (TRF_max[.~fixed_α] - TRF_min[.~fixed_α]) / 2
    G_TRF[  fixed_α] .= (grad_TRF[  fixed_α] .- π ./ TRF[fixed_α].^2 .* grad_ω1[fixed_α]) .* (TRF_max[  fixed_α] - TRF_min[  fixed_α]) / 2

    G_ω1[.~fixed_α] .= grad_ω1[.~fixed_α]  .* ( ω1_max[.~fixed_α] -  ω1_min[.~fixed_α]) / 2
    G_ω1[  fixed_α] .= 0

    G .= [G_ω1; G_TRF]
    G .*= sech.(x).^2

    return G
end


####################################################################
# Penalties on the flip angle α
####################################################################
function TV_α!(grad_ω1, grad_TRF, ω1, TRF; λ = 1)
    α = ω1 .* TRF
    T = length(α)

    F = abs(α[2] - α[end])
    for t = 2:(T - 1)
        F += abs(α[t + 1] - α[t])
    end

    if grad_ω1 !== nothing
        grad_ω1[2]  += λ * sign(α[2] - α[end]) * TRF[2]
        grad_ω1[T]  -= λ * sign(α[2] - α[end]) * TRF[T]
        grad_TRF[2] += λ * sign(α[2] - α[end]) *  ω1[2]
        grad_TRF[T] -= λ * sign(α[2] - α[end]) *  ω1[T]

        for t = 2:(length(α) - 1)
            grad_ω1[t]      -= λ * sign(α[t + 1] - α[t]) * TRF[t]
            grad_ω1[t + 1]  += λ * sign(α[t + 1] - α[t]) * TRF[t + 1]
            grad_TRF[t]     -= λ * sign(α[t + 1] - α[t]) *  ω1[t]
            grad_TRF[t + 1] += λ * sign(α[t + 1] - α[t]) *  ω1[t + 1]
        end
    end
    return λ * F
end

function TV_squared_α!(grad_ω1, grad_TRF, ω1, TRF; λ = 1)
    α = ω1 .* TRF
    T = length(α)

    F = (α[2] - α[end])^2
    for t = 2:(T - 1)
        F += (α[t + 1] - α[t])^2
    end

    if grad_ω1 !== nothing
        grad_ω1[2]  += 2λ * (α[2] - α[T]) * TRF[2]
        grad_ω1[T]  -= 2λ * (α[2] - α[T]) * TRF[T]
        grad_TRF[2] += 2λ * (α[2] - α[T]) *  ω1[2]
        grad_TRF[T] -= 2λ * (α[2] - α[T]) *  ω1[T]

        for t = 2:(T - 1)
            grad_ω1[t]      -= 2λ * (α[t + 1] - α[t]) * TRF[t]
            grad_ω1[t + 1]  += 2λ * (α[t + 1] - α[t]) * TRF[t + 1]
            grad_TRF[t]     -= 2λ * (α[t + 1] - α[t]) *  ω1[t]
            grad_TRF[t + 1] += 2λ * (α[t + 1] - α[t]) *  ω1[t + 1]
        end
    end
    return λ * F
end

function TV_squared_even_α!(grad_ω1, grad_TRF, ω1, TRF; λ = 1)
    α = ω1 .* TRF
    T = length(α)

    F = (α[2] - α[end])^2
    for t = 2:2:(T - 1)
        F += (α[t + 1] - α[t])^2
    end

    if grad_ω1 !== nothing
        grad_ω1[2]  += 2λ * (α[2] - α[T]) * TRF[2]
        grad_ω1[T]  -= 2λ * (α[2] - α[T]) * TRF[T]
        grad_TRF[2] += 2λ * (α[2] - α[T]) *  ω1[2]
        grad_TRF[T] -= 2λ * (α[2] - α[T]) *  ω1[T]

        for t = 2:2:(length(α) - 1)
            grad_ω1[t]      -= 2λ * (α[t + 1] - α[t]) * TRF[t]
            grad_ω1[t + 1]  += 2λ * (α[t + 1] - α[t]) * TRF[t + 1]
            grad_TRF[t]     -= 2λ * (α[t + 1] - α[t]) *  ω1[t]
            grad_TRF[t + 1] += 2λ * (α[t + 1] - α[t]) *  ω1[t + 1]
        end
    end
    return λ * F
end


"""
    F = second_order_α!(grad_ω1, grad_TRF, ω1, TRF; idx=(ω1 .* TRF) .≉ π, λ = 1)

Calculate second order penalty of variations of the flip angle α and adds in place to the gradients.

# Arguments
- `grad_ω1::Vector{Real}`: Gradient of control, which will be added in place
- `grad_TRF::Vector{Real}`: Gradient of control, which will be added in place
- `ω1::Vector{Real}`: Control vector
- `TRF::Vector{Real}`: Control vector

# Optional Keyword Arguments:
- `idx::Vector{Bool}`: index of flip angles that are considered. Set individual individual pulses to `false` to exclude, e.g., inversion pulses
- `λ::Real`: regularization parameter
- `grad_moment = ntuple(i -> i == 1 ? :spoiler : :balanced, length(ω1))`: Different types of gradient moments of each TR are possible (:balanced, :spoiler, :crusher). Skip :crusher and :spoiler TRs second order penalty


# Examples
```jldoctest
julia> ω1 = 4000π * rand(100);

julia> TRF = 500e-6 * rand(100);

julia> grad_ω1 = similar(ω1);

julia> grad_TRF = similar(ω1);

julia> F = MRIgeneralizedBloch.second_order_α!(grad_ω1, grad_TRF, ω1, TRF; λ = 1e-3)
0.3272308747790844
```
"""
function second_order_α!(grad_ω1, grad_TRF, ω1, TRF; λ = 1, nSeq = 1, grad_moment = ntuple(i -> i == 1 ? :spoiler : :balanced, length(ω1)))

    ω1 = reshape(ω1,:,nSeq)
    TRF = reshape(TRF, :, nSeq)
    grad_ω1 = reshape(grad_ω1, :, nSeq)
    grad_TRF = reshape(grad_TRF, :, nSeq)
    grad_moment = reshape(grad_moment,:,nSeq)
    F = 0

    for iSeq in axes(ω1,2)
        α = ω1[:, iSeq] .* TRF[:, iSeq]
        idx = grad_moment[:,iSeq] .== :balanced

        T = sum(idx)

        αi   = @view   α[idx]
        ω1i  = @view  ω1[idx,iSeq]
        TRFi = @view  TRF[idx,iSeq]

        F += (αi[end - 1] / 2 - αi[end] + αi[1] / 2)^2
        F += (αi[end    ] / 2 - αi[1  ] + αi[2] / 2)^2
        for t = 2:T - 1
                F += (αi[t - 1] / 2 - αi[t] + αi[t + 1] / 2)^2
        end

        if grad_ω1 !== nothing
            grad_ω1i = @view grad_ω1[idx,:]
            grad_ω1i[end-1, iSeq] +=  λ * (αi[end-1] / 2 - αi[end] + αi[1] / 2) * TRFi[end-1]
            grad_ω1i[end  , iSeq] -= 2λ * (αi[end-1] / 2 - αi[end] + αi[1] / 2) * TRFi[end]
                grad_ω1i[1    , iSeq] +=  λ * (αi[end-1] / 2 - αi[end] + αi[1] / 2) * TRFi[1]

            grad_ω1i[end, iSeq] +=  λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * TRFi[end]
            grad_ω1i[1  , iSeq] -= 2λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * TRFi[1]
            grad_ω1i[2  , iSeq] +=  λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * TRFi[2]

                for t = 2:T-1
                grad_ω1i[t-1, iSeq] +=  λ * (αi[t-1] / 2 - αi[t] + αi[t+1] / 2) * TRFi[t-1]
                grad_ω1i[t  , iSeq] -= 2λ * (αi[t-1] / 2 - αi[t] + αi[t+1] / 2) * TRFi[t]
                grad_ω1i[t+1, iSeq] +=  λ * (αi[t-1] / 2 - αi[t] + αi[t+1] / 2) * TRFi[t+1]
            end
        end

        if grad_TRF !== nothing
            grad_TRFi = @view grad_TRF[idx,:]
            grad_TRFi[end-1, iSeq] +=  λ * (αi[end-1] / 2 - αi[end] + αi[1] / 2) * ω1i[end-1]
            grad_TRFi[end  , iSeq] -= 2λ * (αi[end-1] / 2 - αi[end] + αi[1] / 2) * ω1i[end]
            grad_TRFi[1    , iSeq] +=  λ * (αi[end-1] / 2 - αi[end] + αi[1] / 2) * ω1i[1]

            grad_TRFi[end, iSeq] +=  λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * ω1i[end]
            grad_TRFi[1  , iSeq] -= 2λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * ω1i[1]
            grad_TRFi[2  , iSeq] +=  λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * ω1i[2]

            for t = 2:T - 1
                grad_TRFi[t-1, iSeq] +=  λ * (αi[t-1] / 2 - αi[t] + αi[t+1] / 2) * ω1i[t-1]
                grad_TRFi[t  , iSeq] -= 2λ * (αi[t-1] / 2 - αi[t] + αi[t+1] / 2) * ω1i[t]
                grad_TRFi[t+1, iSeq] +=  λ * (αi[t-1] / 2 - αi[t] + αi[t+1] / 2) * ω1i[t+1]
            end
        end

    end

    return λ * F
end

function spectral_α!(grad_ω1, grad_TRF, ω1, TRF; λ = 1)
    filter_freq = 4

    α = ω1 .* TRF
    α = α[2:end]
    T = length(α)

    ff = range(-filter_freq, filter_freq, length=T)
    ff = ff.^2
    # ff = 1 .- exp.(-ff.^2)
    ff = diagm(ff)
    ft = ifft(ifftshift(ff), (2))
    ft2 = real.(ft' * ft)

    ft2α = ft2 * α
    F = α' * ft2α
    if grad_ω1 !== nothing
        @views grad_ω1[2:end]  .+= 2λ .* ft2α .* TRF[2:end]
        @views grad_TRF[2:end] .+= 2λ .* ft2α .*  ω1[2:end]
    end
    return λ * F
end

####################################################################
# Penalty on TRF
####################################################################

"""
    F = TRF_TV!(grad_TRF, ω1, TRF; idx=(ω1 .* TRF) .≉ π, λ = 1)

Calculate the total variation penalty of `TRF` and add to `grad_TRF` in place.

# Arguments
- `grad_TRF::Vector{Real}`: Gradient of control, which will added in place
- `ω1::Vector{Real}`: Control vector
- `TRF::Vector{Real}`: Control vector

# Optional Keyword Arguments:
- `idx::Vector{Bool}`: index of flip angles that are considered. Set individual individual pulses to `false` to exclude, e.g., inversion pulses
- `λ::Real`: regularization parameter
- `grad_moment = ntuple(i -> i == 1 ? :spoiler : :balanced, length(ω1))`: Different types of gradient moments of each TR are possible (:balanced, :spoiler, :crusher). Skip :crusher and :spoiler TRs for the TRF TV penalty

# Examples
```jldoctest
julia> ω1 = 4000π * rand(100);

julia> TRF = 500e-6 * rand(100);

julia> grad_TRF = similar(ω1);

julia> F = MRIgeneralizedBloch.TRF_TV!(grad_TRF, ω1, TRF; λ = 1e-3)
1.5456176321183175e-5
```
"""
function TRF_TV!(grad_TRF, ω1, TRF;  λ = 1,  grad_moment = ntuple(i -> i == 1 ? :spoiler : :balanced, length(ω1)))
    idx = trues(size(TRF))
    T = length(TRF)

    for t = 1:(T - 1)
        idx[t] = ((grad_moment[t] != :balanced) || (grad_moment[t+1] != :balanced))
    end

    F = 0
    for t = 1:(T - 1)
        if idx[t]
            F += abs(TRF[t + 1] - TRF[t])
        end
    end

    if grad_TRF !== nothing
        for t = 1:(T - 1)
            if idx[t]
                grad_TRF[t]     -= λ * sign(TRF[t + 1] - TRF[t])
                grad_TRF[t + 1] += λ * sign(TRF[t + 1] - TRF[t])
            end
        end
    end
    return λ * F
end

####################################################################
# SAR Penalty
####################################################################
"""
    F = RF_power!(grad_ω1, grad_TRF, ω1, TRF; idx=(ω1 .* TRF) .≉ π, λ=1, Pmax=3e6, TR=3.5e-3)

Calculate RF power penalty and add the gradients in place.

# Arguments
- `grad_ω1::Vector{Real}`: Gradient of control, which will be added in place
- `grad_TRF::Vector{Real}`: Gradient of control, which will be added in place
- `ω1::Vector{Real}`: Control vector
- `TRF::Vector{Real}`: Control vector

# Optional Keyword Arguments:
- `idx::Vector{Bool}`: index of flip angles that are considered. Set individual individual pulses to `false` to exclude, e.g., inversion pulses
- `λ::Real`: regularization parameter
- `Pmax::Real`: Maximum average power deposition in (rad/s)²; everything above this value will be penalized and with an appropriate λ, the resulting power will be equal to or less than this value.
- `TR::Real`: Repetition time of the pulse sequence

# Examples
```jldoctest
julia> ω1 = 4000π * rand(100);

julia> TRF = 500e-6 * rand(100);

julia> grad_ω1 = similar(ω1);

julia> grad_TRF = similar(ω1);

julia> F = MRIgeneralizedBloch.RF_power!(grad_ω1, grad_TRF, ω1, TRF; λ=1e3, Pmax=3e6, TR=3.5e-3)
1.8164296101582284e14
```
"""
function RF_power!(grad_ω1, grad_TRF, ω1, TRF; λ=1, Pmax=3e6, TR=3.5e-3, nSeq = 1)
    ω1 = reshape(ω1,:,nSeq)
    TRF = reshape(TRF,:,nSeq)
    grad_ω1 = reshape(grad_ω1,:,nSeq)
    grad_TRF = reshape(grad_TRF,:,nSeq)

    N = size(ω1,1)
    ΔSAR = Vector{typeof(ω1[1])}(undef,nSeq)

    P(x)    = x < 0 ? 0 : λ * x^2
    dPdx(x) = x < 0 ? 0 : 2λ * x

    for i in eachindex(ΔSAR)
        ΔSAR[i] = (sum(ω1[:,i].^2 .* TRF[:,i]) / (N * TR) - Pmax)

        if grad_ω1 !== nothing
            grad_ω1[:,i]  .+= dPdx(ΔSAR[i]) .* 2ω1[:,i] .* TRF[:,i] ./ (N * TR)
        end
        if grad_TRF !== nothing
            grad_TRF[:,i] .+= dPdx(ΔSAR[i]) .* ω1[:,i].^2 ./ (N * TR)
        end
    end
    return sum(P.(ΔSAR))
end