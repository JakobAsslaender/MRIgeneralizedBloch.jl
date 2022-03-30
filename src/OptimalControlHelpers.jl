####################################################################
# Bounded ω1 and TRF
####################################################################
"""
    ω1, TRF = get_bounded_ω1_TRF(x; ω1_min = 0, ω1_max = 2e3π, TRF_min = 100e-6, TRF_max = 500e-6)

Transform a vector of length `2Npulses` with values in the range `[-Inf, Inf]` into two vectors of length `Npulses`, which describe the bounded controls `ω1` and `TRF`.

# Arguments
- `x::Vector{<:Number}`: Control vector of `length = 2Npulses` with values in the range `[-Inf, Inf]`

# Optional Keyword Arguments:
- `ω1_min::Number`: lower bound for ω1 in rad/s
- `ω1_max::Number`: upper bound for ω1 in rad/s
- `TRF_min::Number`: lower bound for TRF in s
- `TRF_max::Number`: upper bound for TRF in s

# Examples
```jldoctest
julia> x = 1000 * randn(2 * 100);

julia> ω1, TRF = MRIgeneralizedBloch.get_bounded_ω1_TRF(x)
([6283.185307179586, 0.0, 6283.185307179586, 0.0, 6283.185307179586, 0.0, 6283.185307179586, 0.0, 6283.185307179586, 6283.185307179586  …  6283.185307179586, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6283.185307179586, 0.0, 0.0], [0.0005, 0.0005, 0.0005, 0.0001, 0.0005, 0.0001, 0.0001, 0.0005, 0.0005, 0.0005  …  0.0001, 0.0001, 0.0005, 0.0005, 0.0001, 0.0005, 0.0001, 0.0005, 0.0001, 0.0001])
```
"""
function get_bounded_ω1_TRF(x; ω1_min = 0, ω1_max = 2e3π, TRF_min = 100e-6, TRF_max = 500e-6)
    ω1  =  ω1_min .+ ( ω1_max -  ω1_min) .* (1 .+ tanh.(x[1:end ÷ 2]      )) / 2
    TRF = TRF_min .+ (TRF_max - TRF_min) .* (1 .+ tanh.(x[end ÷ 2 + 1:end])) / 2
    return (ω1, TRF)
end

"""
    x = bound_ω1_TRF!(ω1, TRF; ω1_min = 0, ω1_max = 2e3π, TRF_min = 100e-6, TRF_max = 500e-6)

Bound the controls `ω1` and `TRF` (over-written in place) and return a vector of length `2Npulses` with values in the range `[-Inf, Inf]` that relate to the bounded `ω1` and `TRF` via the `tanh` function.

# Arguments
- `ω1::Vector{<:Number}`: Control vector of length `Npulses`
- `TRF::Vector{<:Number}`: Control vector of length `Npulses`

# Optional Keyword Arguments (see above for defaults):
- `ω1_min::Number`: lower bound for ω1 in rad/s
- `ω1_max::Number`: upper bound for ω1 in rad/s
- `TRF_min::Number`: lower bound for TRF in s
- `TRF_max::Number`: upper bound for TRF in s

# Examples
```jldoctest
julia> ω1 = 4000π * rand(100);

julia> TRF = 500e-6 * rand(100);

julia> x = MRIgeneralizedBloch.bound_ω1_TRF!(ω1, TRF)
200-element Vector{Float64}:
  Inf
   0.16123919298957837
  Inf
  Inf
  Inf
  Inf
   0.17719147195925186
   0.33617937553604216
  -1.1288911659500056
  Inf
   ⋮
 -Inf
  -0.5475311433614531
  -0.6572991695812251
 -Inf
  -0.5487931795341404
  -2.097250369053942
 -Inf
  -1.1536520473794025
   0.07161170293075586
```
"""
function bound_ω1_TRF!(ω1, TRF; ω1_min = 0, ω1_max = 2e3π, TRF_min = 100e-6, TRF_max = 500e-6)
    ω1 .= min.(ω1, ω1_max)
    ω1 .= max.(ω1, ω1_min)

    TRF .= min.(TRF, TRF_max)
    TRF .= max.(TRF, TRF_min)

    x = similar(ω1, 2length(ω1))
    x[1:end÷2]     = atanh.(( ω1 .-  ω1_min) .* 2 ./ ( ω1_max -  ω1_min) .- 1)
    x[end÷2+1:end] = atanh.((TRF .- TRF_min) .* 2 ./ (TRF_max - TRF_min) .- 1)
    return x
end

function apply_bounds_to_grad!(G, x, grad_ω1, grad_TRF; ω1_min = 0, ω1_max = 2e3π, TRF_min = 100e-6, TRF_max = 500e-6)
    G[1:end ÷ 2]       .= grad_ω1  .* ( ω1_max -  ω1_min) / 2
    G[end ÷ 2 + 1:end] .= grad_TRF .* (TRF_max - TRF_min) / 2
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
    F = second_order_α!(grad_ω1, grad_TRF, ω1, TRF; λ = 1)

Calculate second order penalty of variations of the flip angle α and over-write the gradients in place.

# Arguments
- `grad_ω1::Vector{<:Number}`: Gradient of control, which will be over-written in place
- `grad_TRF::Vector{<:Number}`: Gradient of control, which will be over-written in place
- `ω1::Vector{<:Number}`: Control vector
- `TRF::Vector{<:Number}`: Control vector

# Optional Keyword Arguments:
- `λ::Number`: regularization parameter

# Examples
```jldoctest
julia> ω1 = 4000π * rand(100);

julia> TRF = 500e-6 * rand(100);

julia> grad_ω1 = similar(ω1);

julia> grad_TRF = similar(ω1);

julia> F = MRIgeneralizedBloch.second_order_α!(grad_ω1, grad_TRF, ω1, TRF; λ = 1e-3)
0.3394498406831758
```
"""
function second_order_α!(grad_ω1, grad_TRF, ω1, TRF; λ = 1)
    α = ω1 .* TRF
    idx = α .≉ π
    T = sum(idx)

    αi   = @view   α[idx]
    ω1i  = @view  ω1[idx]
    TRFi = @view TRF[idx]

    F  = (αi[end - 1] / 2 - αi[end] + αi[1] / 2)^2
    F += (αi[end    ] / 2 - αi[1  ] + αi[2] / 2)^2
    for t = 2:T - 1
        F += (αi[t - 1] / 2 - αi[t] + αi[t + 1] / 2)^2
    end

    if grad_ω1 !== nothing
        grad_ω1i = @view grad_ω1[idx]
        grad_ω1i[end - 1] +=  λ * (αi[end - 1] / 2 - αi[end] + αi[1  ] / 2) * TRFi[end - 1]
        grad_ω1i[end    ] -= 2λ * (αi[end - 1] / 2 - αi[end] + αi[1  ] / 2) * TRFi[end    ]
        grad_ω1i[1      ] +=  λ * (αi[end - 1] / 2 - αi[end] + αi[1  ] / 2) * TRFi[1      ]

        grad_ω1i[end] +=  λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * TRFi[end]
        grad_ω1i[1  ] -= 2λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * TRFi[1  ]
        grad_ω1i[2  ] +=  λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * TRFi[2  ]

        for t = 2:T - 1
            grad_ω1i[t - 1] +=  λ * (αi[t - 1] / 2 - αi[t] + αi[t + 1] / 2) * TRFi[t - 1]
            grad_ω1i[t    ] -= 2λ * (αi[t - 1] / 2 - αi[t] + αi[t + 1] / 2) * TRFi[t    ]
            grad_ω1i[t + 1] +=  λ * (αi[t - 1] / 2 - αi[t] + αi[t + 1] / 2) * TRFi[t + 1]
        end
    end

    if grad_TRF !== nothing
        grad_TRFi = @view grad_TRF[idx]
        grad_TRFi[end - 1] +=  λ * (αi[end - 1] / 2 - αi[end] + αi[1] / 2) * ω1i[end - 1]
        grad_TRFi[end    ] -= 2λ * (αi[end - 1] / 2 - αi[end] + αi[1] / 2) * ω1i[end    ]
        grad_TRFi[1      ] +=  λ * (αi[end - 1] / 2 - αi[end] + αi[1] / 2) * ω1i[1      ]

        grad_TRFi[end] +=  λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * ω1i[end]
        grad_TRFi[1  ] -= 2λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * ω1i[1  ]
        grad_TRFi[2  ] +=  λ * (αi[end] / 2 - αi[1] + αi[2] / 2) * ω1i[2  ]

        for t = 2:T - 1
            grad_TRFi[t - 1] +=  λ * (αi[t - 1] / 2 - αi[t] + αi[t + 1] / 2) * ω1i[t - 1]
            grad_TRFi[t    ] -= 2λ * (αi[t - 1] / 2 - αi[t] + αi[t + 1] / 2) * ω1i[t    ]
            grad_TRFi[t + 1] +=  λ * (αi[t - 1] / 2 - αi[t] + αi[t + 1] / 2) * ω1i[t + 1]
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
    F = TRF_TV!(grad_TRF, ω1, TRF; λ = 1)

Calculate the total variation penalty of `TRF` and over-write `grad_TRF` in place.

# Arguments
- `grad_TRF::Vector{<:Number}`: Gradient of control, which will be over-written in place
- `ω1::Vector{<:Number}`: Control vector
- `TRF::Vector{<:Number}`: Control vector

# Optional Keyword Arguments:
- `λ::Number`: regularization parameter

# Examples
```jldoctest
julia> ω1 = 4000π * rand(100);

julia> TRF = 500e-6 * rand(100);

julia> grad_TRF = similar(ω1);

julia> F = MRIgeneralizedBloch.TRF_TV!(grad_TRF, ω1, TRF; λ = 1e-3)
1.826335488333797e-5
```
"""
function TRF_TV!(grad_TRF, ω1, TRF; λ = 1)
    T = length(TRF)

    F = 0
    for t = 1:(T - 1)
        if ω1[t] * TRF[t] ≉ π || ω1[t + 1] * TRF[t + 1] ≉ π
            F += abs(TRF[t + 1] - TRF[t])
        end
    end

    if grad_TRF !== nothing
        for t = 1:(T - 1)
            if ω1[t] * TRF[t] ≉ π || ω1[t + 1] * TRF[t + 1] ≉ π
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
    F = RF_power!(grad_ω1, grad_TRF, ω1, TRF; λ=1, Pmax=3e6, TR=3.5e-3)

Calculate RF power penalty and over-write the gradients in place.

# Arguments
- `grad_ω1::Vector{<:Number}`: Gradient of control, which will be over-written in place
- `grad_TRF::Vector{<:Number}`: Gradient of control, which will be over-written in place
- `ω1::Vector{<:Number}`: Control vector
- `TRF::Vector{<:Number}`: Control vector

# Optional Keyword Arguments:
- `λ::Number`: regularization parameter
- `Pmax::Number`: Maximum average power deposition in (rad/s)²; everything above this value will be penalized and with an appropriate λ, the resulting power will be equal to or less than this value.
- `TR::Number`: Repetition time of the pulse sequence

# Examples
```jldoctest
julia> ω1 = 4000π * rand(100);

julia> TRF = 500e-6 * rand(100);

julia> grad_ω1 = similar(ω1);

julia> grad_TRF = similar(ω1);

julia> F = MRIgeneralizedBloch.RF_power!(grad_ω1, grad_TRF, ω1, TRF; λ=1e3, Pmax=3e5)
9.54432950518758e15
```
"""
function RF_power!(grad_ω1, grad_TRF, ω1, TRF; λ=1, Pmax=3e6, TR=3.5e-3)
    N = length(ω1)
    ΔSAR = sum(ω1.^2 .* TRF) / (N * TR) - Pmax

    P(x)    = x < 0 ? 0 : λ * x^2
    dPdx(x) = x < 0 ? 0 : 2λ * x

    if grad_ω1 !== nothing
        grad_ω1  .+= dPdx(ΔSAR) .* 2ω1 .* TRF ./ (N * TR)
    end
    if grad_TRF !== nothing
        grad_TRF .+= dPdx(ΔSAR) .* ω1.^2 ./ (N * TR)
    end
    return P(ΔSAR)
end