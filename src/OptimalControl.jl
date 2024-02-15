"""
    CRB, grad_ω1, grad_TRF = CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, weights; isInversionPulse = [true; falses(length(ω1)-1)])

Calculate the Cramer-Rao bound of a pulse sequence along with the derivatives wrt. `ω1` and `TRF`.

# Arguments
- `ω1::Vector{Real}`: Control vector of `length = Npulses`
- `TRF::Vector{Real}`: Control vector of `length = Npulses`
- `TR::Real`: Repetition time in seconds
- `ω0::Real`: Off-resonance frequency in rad/s
- `B1::Real`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `m0s::Real`: Fractional size of the semi-solid pool; should be in range of 0 to 1
- `R1f::Real`: Longitudinal relaxation rate of the free pool in 1/seconds
- `R2f::Real`: Transversal relaxation rate of the free pool in 1/seconds
- `Rx::Real`: Exchange rate between the two spin pools in 1/seconds
- `R1f::Real`: Longitudinal relaxation rate of the semi-solid pool in 1/seconds
- `T2s::Real`: Transversal relaxation time of the semi-solid pool in seconds
- `R2slT::NTuple{3, Function}`: Tuple of three functions: R2sl(TRF, ω1, B1, T2s), dR2sldB1(TRF, ω1, B1, T2s), and R2sldT2s(TRF, ω1, B1, T2s). Can be generated with [`precompute_R2sl`](@ref)
- `grad_list::Vector{<:grad_param}`: Vector that specifies the gradients that are calculated; the vector elements can either be any subset/order of `grad_list=[grad_m0s(), grad_R1f(), grad_R2f(), grad_Rx(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1()]`; the derivative wrt. to apparent `R1a = R1f = R1s` can be calculated with `grad_R1a()`
- `weights::transpose(Vector{Real})`: Row vector of weights applied to the Cramer-Rao bounds (CRB) of the individual parameters. The first entry always refers to the CRB of M0, followed by the values defined in `grad_list` in the order defined therein. Hence, the vector `weights` has to have one more entry than `grad_list`

# Optional Keyword Arguments:
- `isInversionPulse::Vector{Bool}`: Indicates all inversion pulses; must have the same length as α; the `default = [true; falses(length(ω1)-1)]` indicates that the first pulse is an inversion pulse and all others are not

# Examples
```jldoctest
julia> CRB, grad_ω1, grad_TRF = MRIgeneralizedBloch.CRB_gradient_OCT(rand(100) .* π, rand(100) .* 400e-6 .+ 100e-6, 3.5e-3, 0, 1, 0.15, 0.5, 15, 30, 2, 10e-6, precompute_R2sl(), [grad_m0s(), grad_R1f()], transpose([0, 1, 1]); isInversionPulse = [true, falses(99)...])
(2.6266536440386683e20, [0.0, -8.357210433553662e19, 1.8062863407658156e20, -9.181952733568582e19, 2.0889419004304123e20, -1.0127412004909923e20, 1.1472963520187394e20, -6.048455202064828e19, 1.6635577264610125e20, -1.2997982001201938e20  …  -4.0462197701237735e19, 4.4051154836362985e19, -5.703747921741744e19, 1.1580676614266505e20, -1.2930234020298534e20, 1.4073548384507303e20, -9.192708958806614e19, 1.3584033382847213e20, -3.697066939905562e19, 6.313101282386484e19], [0.0, -7.51230331957692e23, 9.811932053428692e23, -1.0734285487552513e24, 6.675582483464475e23, -3.1051435697300785e23, 2.8969707405246626e23, -1.1612336440328984e24, 6.698477560905162e23, -1.8718360662340176e22  …  -2.7429211167215447e23, 2.5368127989367466e23, -6.640000159002342e23, 1.7977260470624765e23, -3.6616011555760077e23, 4.9307219096771845e23, -7.650701790011881e23, 4.5704084508410106e23, -1.0229952455676927e24, 9.526419421729279e23])

```
c.f. [Optimal Control](@ref)
"""
function CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, weights; isInversionPulse = [true; falses(length(ω1)-1)], nSeq = 1)
    
    E_cat      = Vector{Array{SMatrix{11,11,Float64,121},3}}(undef, nSeq)
    dEdω1_cat  = similar(E_cat)
    dEdTRF_cat = similar(E_cat)

    Q_cat = Vector{Array{SMatrix{11,11,Float64}}}(undef, nSeq)
    Y_cat = Vector{Array{SVector{11,Float64}}}(undef, nSeq)

    ω1 = reshape(ω1,:,nSeq)
    TRF = reshape(TRF,:,nSeq)
    isInversionPulse = reshape(isInversionPulse,:,nSeq)
    
    grad_ω1 = similar(ω1)
    grad_TRF = similar(ω1)

    Threads.@threads for iSeq = 1:nSeq
        @views E, dEdω1, dEdTRF = calculate_propagators_ω1(ω1[:,iSeq], TRF[:,iSeq], TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, isInversionPulse=isInversionPulse[:,iSeq])
        E_cat[iSeq] = E
        dEdω1_cat[iSeq] = dEdω1
        dEdTRF_cat[iSeq] = dEdTRF

        Q_cat[iSeq] = calcualte_cycle_propgator(E_cat[iSeq])
        Y_cat[iSeq] = propagate_magnetization(Q_cat[iSeq], E_cat[iSeq])
    end

    (CRB, d) = dCRBdm(cat(Y_cat...,dims=1), weights)

    for iSeq = 1:nSeq

        P = calculate_adjoint_state(d, Q_cat[iSeq], E_cat[iSeq], iSeq)

        (grad_ω1[:,iSeq], grad_TRF[:,iSeq]) = calculate_gradient_inner_product(P, Y_cat[iSeq], E_cat[iSeq], dEdω1_cat[iSeq], dEdTRF_cat[iSeq])

    end
    
    grad_ω1 = vec(grad_ω1)
    grad_TRF = vec(grad_TRF)

    return (CRB, grad_ω1, grad_TRF)
end



function CRB_gradient_OCT_ortho_multi(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, weights, _dUidVk; isInversionPulse = [true; falses(length(ω1)-1)], nSeq = 1)


    E_cat      = Vector{Array{SMatrix{11,11,Float64,121},3}}(undef, nSeq)
    dEdω1_cat  = similar(E_cat)
    dEdTRF_cat = similar(E_cat)

    Q_cat = Vector{Array{SMatrix{11,11,Float64}}}(undef, nSeq)
    Y_cat = Vector{Array{SVector{11,Float64}}}(undef, nSeq)

    ω1 = reshape(ω1,:,nSeq)
    TRF = reshape(TRF,:,nSeq)
    isInversionPulse = reshape(isInversionPulse,:,nSeq)
    
    grad_ω1 = similar(ω1)
    grad_TRF = similar(ω1)

    Threads.@threads for iSeq = 1:nSeq
        @views E, dEdω1, dEdTRF = calculate_propagators_ω1(ω1[:,iSeq], TRF[:,iSeq], TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, isInversionPulse=isInversionPulse[:,iSeq])
        E_cat[iSeq] = E
        dEdω1_cat[iSeq] = dEdω1
        dEdTRF_cat[iSeq] = dEdTRF

        Q_cat[iSeq] = calcualte_cycle_propgator(E_cat[iSeq])
        Y_cat[iSeq] = propagate_magnetization(Q_cat[iSeq], E_cat[iSeq])
    end



    # (CRB, d) = dCRBdm(Y, weights)
    (C, d) = dCostdm(cat(Y_cat...,dims=1), weights,_dUidVk)
    for iSeq = 1:nSeq
        P = calculate_adjoint_state(d, Q_cat[iSeq], E_cat[iSeq], iSeq)
        (grad_ω1[:,iSeq], grad_TRF[:,iSeq]) = calculate_gradient_inner_product(P, Y_cat[iSeq], E_cat[iSeq], dEdω1_cat[iSeq], dEdTRF_cat[iSeq])
    end

    grad_ω1 = vec(grad_ω1)
    grad_TRF = vec(grad_TRF)
    return (C, grad_ω1, grad_TRF)
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
end

# the commented line in this function are required for the adjoint state to be correct; but since these entries are not used for the OCT algorithm, we skip calculating them.
function calculate_adjoint_state(d, Q, E, iSeq)
    P = similar(E, Array{Float64})
    λ = @view P[end,:,:]

    for g in 1:size(E, 3), r in 1:size(E, 2)
    λ[r,g] = d(size(E, 1) + (iSeq-1)*size(E, 1), r, g)

    end

    for r = 1:size(E, 2)
        for t = size(E, 1) - 1:-1:1
            λ[r,1] = transpose(E[t + 1,r,1]) * λ[r,1]
            λ[r,1] += d(t + (iSeq-1)*size(E, 1) , r, 1)
            for g = 2:size(E, 3)
                λ[r,g] = transpose(E[t + 1,r,g][6:10,:]) * λ[r,g][6:10]
                λ[r,1][1:5] .+= λ[r,g][1:5]
                # λ[r,1][end]  += λ[r,g][end]
                λ[r,g] .+= d(t + (iSeq-1)*size(E, 1), r, g)
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
                P[t - 1,r,g] .+= d(t + (iSeq-1)*size(E, 1), r, g)
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


function dCostdm(Y, w,_dUidVk)
    _dCdx = Array{Float64}(undef, size(Y, 1), size(Y, 3) + 1)
    _dCdy = similar(_dCdx)
    
    V = Array{ComplexF64}(undef, size(Y,1),size(Y, 3) + 1) # not implemented for phase-cycling
    
    for g in 0:size(Y, 3), t in 1:size(Y, 1)
        if g == 0
            V[t,1]     =  Y[t,1,1][1]  + 1im * Y[t,1,1][2]
        else
            V[t,g+1] =  Y[t,1,g][6] + 1im * Y[t,1,g][7]
        end
    end
    
    
    Vshift = [circshift(V,(0,i)) for i = size(V,2)-1: -1 : 0] # shift corresponding derivative to the end
    U      = [CalcU(Vshift[i])   for i = 1:size(V,2)]
    
    Gperp = cat(dims =3, U...)[:,end,:]

    C = 0
    for i in 1:size(Gperp,2)
        C += w[i]/norm(Gperp[:,i],1)
    end

    y = [zeros(ComplexF64,size(Y, 1),) for i = 1:size(Y,3)+1]

    _dCdx .= 0
    _dCdy .= 0
    for j in eachindex(w)
        if (w[j] != 0)


            dUidVk_mat_real!(U[j],Vshift[j],_dUidVk,y)

            for i in 1:size(Y,3)+1
                
                shiftIdx = sortperm(circshift(1:length(U), length(U)-j))                

                _dCdx[:,i]    .+= w[j] / (sum(abs.(Gperp[:,j]))).^2 .* transpose(sum( sign.(real.(Gperp[:,j])).* real.(_dUidVk[end,shiftIdx[i]]'),dims=1))
                _dCdy[:,i]    .+= w[j] / (sum(abs.(Gperp[:,j]))).^2 .* transpose(sum( sign.(imag.(Gperp[:,j])).* imag.(_dUidVk[end,shiftIdx[i]]'),dims=1))
                # the orth. grad of each paramter is always in the last row of the respective U matrix (also Vshift).
                # Thus the derivative of (ds/dR1f)perp wrt. to (ds/dM0) for example is d U[3][:,end]/ d Vshift[3][:,X]. X has to be the index of ds/dM0
                # in the Vshift[j] matrix, shiftIdx helps to undo these circshifts.
            end 
        end
    end

    d(t, r, g) = @SVector [_dCdx[t,1], _dCdy[t,1],0,0,0,_dCdx[t,g + 1], _dCdy[t,g + 1],0,0,0,0]
    return (C, d)
end

function CalcU(V)
    U = similar(V)
    for i = 1:size(U,2)
        U[:,i] = V[:,i]
        for j = i-1:-1:1
            U[:,i] .-= Proj(U[:,j],V[:,i])
        end
    end
    return U
end


function Proj(U,V)
    return (U'*V)/real(U'*U) * U
end


function dVidVk!(U,i,k,_dUidVk_el)
    if i == k
        _dUidVk_el  .= 0
        for iloop in range(1,size(U,1))
            _dUidVk_el[iloop,iloop]  = 1 # unit matrix
        end
    else
        _dUidVk_el .= 0
    end
end


function dUidVk_mat_real!(U,V,_dUidVk,y)

    Threads.@threads for idxV = 1:size(U,2)
        for idxU = 1:size(U,2)
            if (idxV <= idxU)
            
                @views dVidVk!(U,idxU,idxV,_dUidVk[idxU,idxV])

                for l = 1:idxU-1
                    
                    mul!(_dUidVk[idxU,idxV], _dUidVk[l,idxV], I , -(V[:,idxU]'*U[:,l])./(U[:,l]'*U[:,l]), 1)
                    
                    if idxU == idxV
                        @views y[idxV]  .= conj(U[:,l])
                    else
                        @views y[idxV]  .= 0
                    end
                    @views y[idxV] .+= _dUidVk[l,idxV]*V[:,idxU]
                    @views y[idxV] ./= (U[:,l]'*U[:,l])

                    mul!(y[idxV], _dUidVk[l,idxV] , U[:,l], -2*(V[:,idxU]'*U[:,l])/((U[:,l]'*U[:,l])^2) , 1)

                    mul!(_dUidVk[idxU,idxV], y[idxV], transpose(U[:,l]), -1, 1)
                end
            end
        end
    end
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

    for g in 1:size(E, 3), r in 1:size(E, 2)
        if g == 1
            grad_ω1[1]  -= transpose(P[end,r,g]) * (dEdω1[1,r,g] * Y[end,r,g])
            grad_TRF[1] -= transpose(P[end,r,g]) * (dEdTRF[1,r,g] * Y[end,r,g])
        else
            a = dEdω1[1,r,g] * Y[end,r,g]
            b = dEdTRF[1,r,g] * Y[end,r,g]
            @inbounds for i = 6:10
                grad_ω1[1]   -= P[end,r,g][i] * a[i]
                grad_TRF[1]  -= P[end,r,g][i] * b[i]
            end
        end
    end


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