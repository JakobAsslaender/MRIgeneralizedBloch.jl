"""
    CRB, grad_ω1, grad_TRF = CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, weights; isInversionPulse = falses(length(ω1)))

Calculate the Cramer-Rao bound of a pulse sequence along with the derivatives wrt. `ω1` and `TRF`.

# Arguments
- `ω1::Vector{<:Number}`: Control vector of `length = Npulses`
- `TRF::Vector{<:Number}`: Control vector of `length = Npulses`
- `TR::Number`: Repetition time in seconds
- `ω0::Number`: Off-resonance frequency in rad/s
- `B1::Number`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `m0s::Number`: Fractional size of the semi-solid pool; should be in range of 0 to 1
- `R1f::Number`: Longitudinal relaxation rate of the free pool in 1/seconds
- `R2f::Number`: Transversal relaxation rate of the free pool in 1/seconds
- `Rx::Number`: Exchange rate between the two spin pools in 1/seconds
- `R1f::Number`: Longitudinal relaxation rate of the semi-solid pool in 1/seconds
- `T2s::Number`: Transversal relaxationt time of the semi-solid pool in seconds
- `R2slT::NTuple{3, Function}`: Tuple of three functions: R2sl(TRF, ω1, B1, T2s), dR2sldB1(TRF, ω1, B1, T2s), and R2sldT2s(TRF, ω1, B1, T2s). Can be generated with [`precompute_R2sl`](@ref)
- `grad_list::Vector{<:grad_param}`: Vector to indicate which gradients should be calculated; the vector elements can either be any subset/order of `grad_list=[grad_m0s(), grad_R1f(), grad_R2f(), grad_Rx(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1()]`; the derivative wrt. to apparent `R1a = R1f = R1s` can be calculated with `grad_R1a()`
- `weights::transpose(Vector{<:Number})`: Row vector of weights given to the Cramer-Rao bounds (CRB) of the individual parameters. The first entry always refers to the CRB of M0, followed by the values defined in `grad_list` in that order. Hence, the Vector `weights` has to have one more entry than `grad_list`

# Optional Keyword Arguments:
- `isInversionPulse::Vector{Bool}`: Indicates with pulses are inversion pulses. Where `true`, the algorithm will simulate crusher gradients before and after the pulse and `ω1` and `TRF` of those pulses are excluded from the optimization.

# Examples
```jldoctest
julia> CRB, grad_ω1, grad_TRF = MRIgeneralizedBloch.CRB_gradient_OCT(rand(100) .* π, rand(100) .* 400e-6 .+ 100e-6, 3.5e-3, 0, 1, 0.15, 0.5, 15, 30, 2, 10e-6, precompute_R2sl(), [grad_m0s(), grad_R1f()], transpose([0, 1, 1]); isInversionPulse = [true, falses(99)...])
(4.326345257791475e20, [0.0, 3.8149138162765024e20, -1.4073464193072628e20, 1.196651310804305e20, -1.9989638164742383e20, 3.6493209787269874e20, -1.5536324770044346e20, 1.7284798914645443e20, -2.0772522228321904e20, 2.7044669618035792e20  …  -6.3928347250126905e19, 1.5671768655962505e20, -1.1080723702078613e20, 5.7454290837437145e19, -7.845659845122268e19, 1.3447202941410777e20, -6.377328799158878e19, 5.1500654841007415e19, -3.97683716404465e19, 1.2714666624472279e20], [0.0, 2.9969441368202493e24, -1.4650231537214497e24, 1.8226585318981564e24, -4.7462030797597407e23, 1.97549455130354e24, -1.5270500210926142e24, 9.099568864141483e23, -1.4811032902868818e24, 1.2220830933233262e24  …  -5.258326682397155e23, 6.918585992930617e23, -2.8293355719514886e23, 4.015546075884712e23, -4.1767064624316516e23, 5.061776753372734e23, -4.1049901738345103e23, 9.311794792265391e23, -4.131984395429261e23, 7.610379029805631e23])

```
c.f. [Optimal Control](@ref)
"""
function CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, weights; isInversionPulse = falses(length(ω1)))
    (E, dEdω1, dEdTRF) = calculate_propagators_ω1(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, isInversionPulse=isInversionPulse)
    Q = calcualte_cycle_propgator(E)
    Y = propagate_magnetization(Q, E)
    (CRB, d) = dCRBdm(Y, weights)
    P = calculate_adjoint_state(d, Q, E)
    (grad_ω1, grad_TRF) = calculate_gradient_inner_product(P, Y, E, dEdω1, dEdTRF)
    return (CRB, grad_ω1, grad_TRF)
end

function OCT_TV_gradient(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, weights, λ_TV)
    (E, dEdω1, dEdTRF) = calculate_propagators_ω1(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list)
    Q = calcualte_cycle_propgator(E)
    Y = propagate_magnetization(Q, E)

    (CRB, d) = dCRBdm(Y, weights)
    P = calculate_adjoint_state(d, Q, E)
    (grad_ω1, grad_TRF) = calculate_gradient_inner_product(P, Y, E, dEdω1, dEdTRF)

    (TV, d) = dthetaTVdm(Y, λ_TV)
    P = calculate_adjoint_state(d, Q, E)
    (grad_ω1_TV, grad_TRF_TV) = calculate_gradient_inner_product(P, Y, E, dEdω1, dEdTRF)

    grad_ω1  .-= grad_ω1_TV
    grad_TRF .-= grad_TRF_TV

    return (CRB+TV, grad_ω1, grad_TRF)
end



function calculate_propagators_ω1(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list::AbstractArray{T}; rfphase_increment=[π], isInversionPulse = [true, falses(length(ω1)-1)...]) where T <: grad_param
    E      = Array{SMatrix{11,11,Float64,121}}(undef, length(ω1), length(rfphase_increment), length(grad_list))
    dEdω1  = Array{SMatrix{11,11,Float64,121}}(undef, length(ω1), length(rfphase_increment), length(grad_list))
    dEdTRF = Array{SMatrix{11,11,Float64,121}}(undef, length(ω1), length(rfphase_increment), length(grad_list))

    cache = ExponentialUtilities.alloc_mem(zeros(22, 22), ExpMethodHigham2005Base())
    dH = zeros(Float64, 22, 22)

    for r ∈ eachindex(rfphase_increment)
        u_rot = z_rotation_propagator(rfphase_increment[r], grad_m0s())
        for g ∈ eachindex(grad_list)
            grad = grad_list[g]

            for t ∈ 1:length(ω1)
                if isInversionPulse[t]
                    calculate_inversion_propagator!(E, dEdω1, dEdTRF, t, r, g, ω1[t], TRF[t], TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad, u_rot, dH, cache)
                else
                    calculte_propagator!(E, dEdω1, dEdTRF, t, r, g, ω1[t], TRF[t], TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad, u_rot, dH, cache)
                end
            end
        end
    end
    return (E, dEdω1, dEdTRF)
end

function calculte_propagator!(E, dEdω1, dEdTRF, t, r, g, ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad, u_rot, dH, cache)

    H_fp = hamiltonian_linear(0.0, B1, ω0, 1.0, m0s, R1f, R2f, Rx, R1s, 0.0, 0.0, 0.0, grad)
    ux = xs_destructor(grad)
    u_fp = ux * exp(H_fp * ((TR - TRF) / 2))

    H_pl = hamiltonian_linear(ω1, B1, ω0, 1, m0s, R1f, R2f, Rx, R1s,
        R2slT[1](TRF, ω1 * TRF, B1, T2s),
        R2slT[2](TRF, ω1 * TRF, B1, T2s),
        R2slT[3](TRF, ω1 * TRF, B1, T2s),
        grad)

    dHdω1 = d_hamiltonian_linear_dω1(B1, 1,
        R2slT[4](TRF, ω1 * TRF, B1, T2s),
        R2slT[6](TRF, ω1 * TRF, B1, T2s),
        R2slT[7](TRF, ω1 * TRF, B1, T2s),
        grad)

    @views dH[1:11,1:11]   .= H_pl
    @views dH[12:22,12:22] .= H_pl
    @views dH[1:11,12:22]  .= 0
    @views dH[12:22,1:11]  .= dHdω1
    dH .*= TRF
    E_pl = exponential!(dH, ExpMethodHigham2005Base(), cache)

    E_pl1 = SMatrix{11,11}(@view E_pl[1:11,1:11])
    E_pl2 = SMatrix{11,11}(@view E_pl[12:end,1:11])
    E[t,r,g]     = u_fp * E_pl1 * u_rot * u_fp
    dEdω1[t,r,g] = u_fp * E_pl2 * u_rot * u_fp

    # TRF
    dHdTRF = H_pl + d_hamiltonian_linear_dTRF_add(TRF,
        R2slT[5](TRF, ω1 * TRF, B1, T2s),
        R2slT[8](TRF, ω1 * TRF, B1, T2s),
        R2slT[9](TRF, ω1 * TRF, B1, T2s),
        grad)
    H_pl *= TRF
    @views dH[1:11,1:11]   .= H_pl
    @views dH[12:22,12:22] .= H_pl
    @views dH[1:11,12:22]  .= 0
    @views dH[12:22,1:11]  .= dHdTRF
    E_pl = exponential!(dH, ExpMethodHigham2005Base(), cache)

    E_pl1 = SMatrix{11,11}(@view E_pl[1:11,1:11])
    E_pl2 = SMatrix{11,11}(@view E_pl[12:end,1:11])
    dEdTRF[t,r,g] = u_fp * ((E_pl2 - (1 / 2 * H_fp * E_pl1)) * u_rot - (1 / 2 * E_pl1 * u_rot * H_fp)) * u_fp
    return nothing
end

function calculate_inversion_propagator!(E, dEdω1, dEdTRF, t, r, g, ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad, u_rot, _, _)
    u_fp = xs_destructor(grad) * exp(hamiltonian_linear(0, B1, ω0, TR / 2, m0s, R1f, R2f, Rx, R1s, 0, 0, 0, grad))
    u_pl = propagator_linear_inversion_pulse(ω1, TRF, B1,
        R2slT[1](TRF, ω1 * TRF, B1, T2s),
        R2slT[2](TRF, ω1 * TRF, B1, T2s),
        R2slT[3](TRF, ω1 * TRF, B1, T2s),
        grad)
    E[t,r,g] = u_fp * u_pl * u_rot * u_fp
    dEdω1[t,r,g] = @SMatrix zeros(11, 11)
    dEdTRF[t,r,g] = @SMatrix zeros(11, 11)
    return nothing
end

function calcualte_cycle_propgator(E)
    Q = Array{SMatrix{11,11,Float64}}(undef, size(E, 2), size(E, 3))

    for g in 1:size(E, 3), r in 1:size(E, 2)
        A = E[1,r,g]
        for t = size(E, 1):-1:2
            A = A * E[t,r,g]
        end
        Q[r,g] = A0(grad_m0s()) - A
    end
    return Q
end

function propagate_magnetization(Q, E)
    Y = similar(E, SVector{11,Float64})

    for r in 1:size(E, 2), g in 1:size(E, 3)
        m = Q[r,g] \ C(grad_m0s())

        Y[1,r,g] = m
        for t = 2:size(E, 1)
            m = E[t,r,g] * m
            Y[t,r,g] = m
        end
    end
    return Y

    return S
end

# the commented line in this function are required for the adjoint state to be correct; but since these entries are not used for the OCT algorithm, we skip calculating them.
function calculate_adjoint_state(d, Q, E)
    P = similar(E, Array{Float64})
    λ = @view P[end,:,:]

    for g in 1:size(E, 3), r in 1:size(E, 2)
        λ[r,g] = d(size(E, 1), r, g)
    end

    for r = 1:size(E, 2)
        for t = size(E, 1) - 1:-1:1
            λ[r,1] = transpose(E[t + 1,r,1]) * λ[r,1]
            λ[r,1] += d(t, r, 1)
            for g = 2:size(E, 3)
                λ[r,g] = transpose(E[t + 1,r,g][6:10,:]) * λ[r,g][6:10]
                λ[r,1][1:5] .+= λ[r,g][1:5]
                # λ[r,1][end]  += λ[r,g][end]
                λ[r,g] .+= d(t, r, g)
            end
        end

    # for g = 2:length(grad_list)
    #     λ[r,g][1:5] .= λ[r,1][1:5]
    #     λ[r,g][end]  = λ[r,1][end]
    # end

        P[end,r,1] = inv(transpose(Q[r,1])) * λ[r,1]
        for g = 2:size(E, 3)
            λ[r,g] = inv(transpose(Q[r,g]))[:,6:10] * λ[r,g][6:10]
            λ[r,1][1:5] .+= λ[r,g][1:5]
            # P[end,r,1][end]  += P[end,r,g][end]
        end

        # step 5: propagate adjoint state
        for t in size(E, 1):-1:2
            for g = 1:size(E, 3)
                _E = E[mod(t, size(E, 1)) + 1,r,g]
                if g == 1
                    P[t - 1,r,g] = transpose(_E) * P[t,r,g]
                else
                    P[t - 1,r,g] = transpose(_E[6:10,:]) * P[t,r,g][6:10]
                    P[t - 1,r,1][1:5] .+= P[t - 1,r,g][1:5]
                    P[t - 1,r,1][end]  += P[t - 1,r,g][end]
                end
                P[t - 1,r,g] .+= d(t, r, g)
            end

        # for g = 2:length(grad_list)
        #     P[t - 1,r,g][1:5] .= P[t - 1,r,1][1:5]
        #     P[t - 1,r,g][end]  = P[t - 1,r,1][end]
        # end
        end
    end
    return P
end

function dCRBdm(Y, w)
    _dCRBdx = Array{Float64}(undef, size(Y, 1), size(Y, 2), size(Y, 3) + 1)
    _dCRBdy = similar(_dCRBdx)
    F = zeros(ComplexF64, size(Y, 3) + 1, size(Y, 3) + 1)
    dFdy = similar(F)
    tmp  = similar(F)

    for g2 in 0:size(Y, 3), g1 in 0:size(Y, 3), r in 1:size(Y, 2), t in 1:size(Y, 1)
        if g1 == 0
            s1 = Y[t,r,1][1] - 1im * Y[t,r,1][2]
        else
            s1 = Y[t,r,g1][6] - 1im * Y[t,r,g1][7]
        end
        if g2 == 0
            s2 = Y[t,r,1][1] + 1im * Y[t,r,1][2]
        else
            s2 = Y[t,r,g2][6] + 1im * Y[t,r,g2][7]
        end
        F[g1 + 1,g2 + 1] += s1 * s2
    end
    Fi = inv(F)
    CRB = w * real.(diag(Fi))

    for g1 in 0:size(Y, 3), r in 1:size(Y, 2), t in 1:size(Y, 1)
        # derivative wrt. x
        dFdy .= 0
        dFdy[g1 + 1,1] = Y[t,r,1][1] + 1im * Y[t,r,1][2]
        dFdy[1,g1 + 1] = Y[t,r,1][1] - 1im * Y[t,r,1][2]
        for g2 in 1:size(Y, 3)
            dFdy[g1 + 1,g2 + 1] = Y[t,r,g2][6] + 1im * Y[t,r,g2][7]
            dFdy[g2 + 1,g1 + 1] = Y[t,r,g2][6] - 1im * Y[t,r,g2][7]
        end
        dFdy[g1 + 1,g1 + 1] = 2 * real(dFdy[g1 + 1,g1 + 1])
        mul!(dFdy, Fi, mul!(tmp, dFdy, Fi))
        _dCRBdx[t,r,g1 + 1] = real.(w * diag(dFdy))

        # derivative wrt. y
        dFdy .= 0
        dFdy[g1 + 1,1] = Y[t,r,1][2] - 1im * Y[t,r,1][1]
        dFdy[1,g1 + 1] = Y[t,r,1][2] + 1im * Y[t,r,1][1]
        for g2 in 1:size(Y, 3)
            dFdy[g1 + 1,g2 + 1] = Y[t,r,g2][7] - 1im * Y[t,r,g2][6]
            dFdy[g2 + 1,g1 + 1] = Y[t,r,g2][7] + 1im * Y[t,r,g2][6]
        end
        dFdy[g1 + 1,g1 + 1] = 2 * real(dFdy[g1 + 1,g1 + 1])
        mul!(dFdy, Fi, mul!(tmp, dFdy, Fi))
        _dCRBdy[t,r,g1 + 1] = real.(w * diag(dFdy))
    end

    d(t, r, g) = @SVector [_dCRBdx[t,r,1], _dCRBdy[t,r,1],0,0,0,_dCRBdx[t,r,g + 1], _dCRBdy[t,r,g + 1],0,0,0,0]

    return (CRB, d)
end

function dthetaTVdm(Y, λ)
    # ϑ = vec([atan(Y[t,r,1][1] / Y[t,r,1][3]) for t=1:size(Y,1), r=1:size(Y,2)])
    # TV = (ϑ[1] - ϑ[end])^2
    # for t=2:length(ϑ)
    #     TV += (ϑ[t] - ϑ[t-1])^2
    # end

    T = size(Y, 1)
    R = size(Y, 2)
    @inline x(t, r) = Y[t,r,1][1]
    @inline z(t, r) = Y[t,r,1][3]
    @inline ϑ(t, r) = atan(x(t, r) / z(t, r))

    TV = zero(eltype(Y[1]))
    _dthetaTVdx = ones(size(Y, 1), size(Y, 2))
    _dthetaTVdz = ones(size(Y, 1), size(Y, 2))

    ϑₙ₋₁ = ϑ(T, R)
    ϑₙ₋₂ = ϑ(T - 1, R)
    tₙ₋₁ = T
    rₙ₋₁ = R
    for r ∈ 1:R
        for t ∈ 1:T
            ϑₙ = ϑ(t, r)
            TV += (ϑₙ - ϑₙ₋₁)^2

            dₙ  = x(t, r)^2 + z(t, r)^2
            _dthetaTVdx[t,r] *= z(t, r) / dₙ
            _dthetaTVdz[t,r] *= -x(t, r) / dₙ
            _dthetaTVdx[tₙ₋₁,rₙ₋₁] *= 2λ * (2ϑₙ₋₁ - ϑₙ - ϑₙ₋₂)
            _dthetaTVdz[tₙ₋₁,rₙ₋₁] *= 2λ * (2ϑₙ₋₁ - ϑₙ - ϑₙ₋₂)

            tₙ₋₁ = t
            rₙ₋₁ = r
            ϑₙ₋₂ = ϑₙ₋₁
            ϑₙ₋₁ = ϑₙ
        end
    end

    d(t, r, _) = @SVector [_dthetaTVdx[t,r], 0, _dthetaTVdz[t,r],0,0,0,0,0,0,0,0]
    return (λ * TV, d)
end

function calculate_gradient_inner_product(P, Y, E, dEdω1, dEdTRF)
    grad_ω1 = zeros(size(E, 1))
    grad_TRF = zeros(size(E, 1))

    for g in 1:size(E, 3), r in 1:size(E, 2), t in 2:size(E, 1)
        if g == 1
            grad_ω1[t]  -= transpose(P[t - 1,r,g]) * (dEdω1[t,r,g] * Y[t - 1,r,g])
            grad_TRF[t] -= transpose(P[t - 1,r,g]) * (dEdTRF[t,r,g] * Y[t - 1,r,g])
        else
            a = dEdω1[t,r,g] * Y[t - 1,r,g]
            b = dEdTRF[t,r,g] * Y[t - 1,r,g]
            @inbounds for i = 6:10
                grad_ω1[t]  -= P[t - 1,r,g][i] * a[i]
                grad_TRF[t]  -= P[t - 1,r,g][i] * b[i]
            end
        end
    end
    return (grad_ω1, grad_TRF)
end