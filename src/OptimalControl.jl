"""
    CRB, grad_ω1, grad_TRF = CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list, weights; grad_moment = [i == 1 ? :spoiler_dual : :balanced for i ∈ eachindex(ω1)])

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
- `Rex::Real`: Exchange rate between the two spin pools in 1/seconds
- `R1f::Real`: Longitudinal relaxation rate of the semi-solid pool in 1/seconds
- `T2s::Real`: Transversal relaxation time of the semi-solid pool in seconds
- `R2slT::NTuple{3, Function}`: Tuple of three functions: R2sl(TRF, ω1, B1, T2s), dR2sldB1(TRF, ω1, B1, T2s), and R2sldT2s(TRF, ω1, B1, T2s). Can be generated with [`precompute_R2sl`](@ref)
- `grad_list::Tuple{<:grad_param}`: Tuple that specifies the gradients that are calculated; the vector elements can either be any subset/order of `grad_list=(grad_m0s(), grad_R1f(), grad_R2f(), grad_Rex(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1())`; the derivative wrt. to apparent `R1a = R1f = R1s` can be calculated with `grad_R1a()`
- `weights::transpose(Vector{Real})`: Row vector of weights applied to the Cramer-Rao bounds (CRB) of the individual parameters. The first entry always refers to the CRB of M0, followed by the values defined in `grad_list` in the order defined therein. Hence, the vector `weights` has to have one more entry than `grad_list`

# Optional Keyword Arguments:
- `grad_moment = [i == 1 ? :spoiler_dual : :balanced for i ∈ eachindex(α)]`: Different types of gradient moments of each TR are possible (`:balanced`, `:crusher`, `:spoiler_dual`, `:spoiler_prepulse`). `:balanced` simulates a TR with all gradient moments nulled. `:crusher` assumes equivalent (non-zero) gradient moments before and simulates the refocussing path of the extended phase graph. `:spoiler_prepulse` nulls all transverse magnetization before the RF pulse, emulating an idealized FLASH. `:spoiler_dual` nulls all transverse magnetization before and after the RF pulse.
- `nSeq = 1`: Allows multiple flip angle pattern to be jointly optimized. CRB (and derivatives) are calucluted by taking the joint signal of all flip angle patterns. The periodic boundary conditions of the magnetization are calucluted within each flip angle pattern.

# Examples
```jldoctest
julia> CRB, grad_ω1, grad_TRF = MRIgeneralizedBloch.CRB_gradient_OCT(rand(100) .* π, rand(100) .* 400e-6 .+ 100e-6, 3.5e-3, 0, 1, 0.15, 0.5, 15, 30, 2, 10e-6, precompute_R2sl(), [grad_m0s(), grad_R1f()], transpose([0, 1, 1])])
(2.6266536440386683e20, [0.0, -8.357210433553662e19, 1.8062863407658156e20, -9.181952733568582e19, 2.0889419004304123e20, -1.0127412004909923e20, 1.1472963520187394e20, -6.048455202064828e19, 1.6635577264610125e20, -1.2997982001201938e20  …  -4.0462197701237735e19, 4.4051154836362985e19, -5.703747921741744e19, 1.1580676614266505e20, -1.2930234020298534e20, 1.4073548384507303e20, -9.192708958806614e19, 1.3584033382847213e20, -3.697066939905562e19, 6.313101282386484e19], [0.0, -7.51230331957692e23, 9.811932053428692e23, -1.0734285487552513e24, 6.675582483464475e23, -3.1051435697300785e23, 2.8969707405246626e23, -1.1612336440328984e24, 6.698477560905162e23, -1.8718360662340176e22  …  -2.7429211167215447e23, 2.5368127989367466e23, -6.640000159002342e23, 1.7977260470624765e23, -3.6616011555760077e23, 4.9307219096771845e23, -7.650701790011881e23, 4.5704084508410106e23, -1.0229952455676927e24, 9.526419421729279e23])

```
c.f. [Optimal Control](@ref)
"""
function CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list, weights; grad_moment=[i == 1 ? :spoiler_dual : :balanced for i ∈ eachindex(ω1)], nSeq=1)

    E_cat = Vector{Array{SMatrix{11,11,Float64,121},3}}(undef, nSeq)
    dEdω1_cat = similar(E_cat)
    dEdTRF_cat = similar(E_cat)

    Q_cat = Vector{Array{SMatrix{11,11,Float64}}}(undef, nSeq)
    Y_cat = Vector{Array{SVector{11,Float64}}}(undef, nSeq)

    ω1 = reshape(ω1, :, nSeq)
    TRF = reshape(TRF, :, nSeq)
    grad_moment = reshape(grad_moment, :, nSeq)

    grad_ω1 = similar(ω1)
    grad_TRF = similar(ω1)

    Threads.@threads for iSeq = 1:nSeq
        @views E, dEdω1, dEdTRF = calculate_propagators_ω1(ω1[:, iSeq], TRF[:, iSeq], TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list, grad_moment=grad_moment[:, iSeq])
        E_cat[iSeq] = E
        dEdω1_cat[iSeq] = dEdω1
        dEdTRF_cat[iSeq] = dEdTRF

        Q_cat[iSeq] = calcualte_cycle_propgator(E_cat[iSeq])
        Y_cat[iSeq] = propagate_magnetization(Q_cat[iSeq], E_cat[iSeq])
    end

    CRB, d = dCRBdm(cat(Y_cat..., dims=1), weights)

    for iSeq = 1:nSeq
        P = calculate_adjoint_state(d, Q_cat[iSeq], E_cat[iSeq], iSeq)
        grad_ω1[:, iSeq], grad_TRF[:, iSeq] = calculate_gradient_inner_product(P, Y_cat[iSeq], E_cat[iSeq], dEdω1_cat[iSeq], dEdTRF_cat[iSeq])
    end

    return CRB, vec(grad_ω1), vec(grad_TRF)
end



function calculate_propagators_ω1(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list; rfphase_increment=[π], grad_moment=[i == 1 ? :spoiler_dual : :balanced for i ∈ eachindex(ω1)])
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
                if grad_moment[t] == :crusher
                    calculate_crushed_pulse_propagator!(E, dEdω1, dEdTRF, t, r, g, ω1[t], TRF[t], TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad, u_rot, dH, cache)
                else
                    calculte_propagator!(E, dEdω1, dEdTRF, t, r, g, ω1[t], TRF[t], TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad, u_rot, dH, cache, grad_moment[t])
                end
            end
        end
    end
    return E, dEdω1, dEdTRF
end

function calculte_propagator!(E, dEdω1, dEdTRF, t, r, g, ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad, u_rot, dH, cache, grad_moment)

    H_fp = hamiltonian_linear(0.0, B1, ω0, 1.0, m0s, R1f, R2f, Rex, R1s, 0.0, 0.0, 0.0, grad)
    ux = grad_moment == :spoiler_dual ? xy_destructor(H_fp) : xs_destructor(H_fp)
    u_fp = ux * exp(H_fp * ((TR - TRF) / 2))

    H_pl = hamiltonian_linear(ω1, B1, ω0, 1, m0s, R1f, R2f, Rex, R1s,
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

    E_pl1 = SMatrix{11,11}(@view E_pl[1:11, 1:11])
    E_pl2 = SMatrix{11,11}(@view E_pl[12:end, 1:11])
    E[t, r, g] = u_fp * E_pl1 * u_rot * u_fp
    dEdω1[t, r, g] = u_fp * E_pl2 * u_rot * u_fp

    # TRF
    dHdTRF = H_pl + d_hamiltonian_linear_dTRF_add(TRF,
        R2slT[5](TRF, ω1 * TRF, B1, T2s),
        R2slT[8](TRF, ω1 * TRF, B1, T2s),
        R2slT[9](TRF, ω1 * TRF, B1, T2s),
        grad)
    H_pl *= TRF
    @views dH[1:11, 1:11] .= H_pl
    @views dH[12:22, 12:22] .= H_pl
    @views dH[1:11, 12:22] .= 0
    @views dH[12:22, 1:11] .= dHdTRF
    E_pl = exponential!(dH, ExpMethodHigham2005Base(), cache)

    E_pl1 = SMatrix{11,11}(@view E_pl[1:11, 1:11])
    E_pl2 = SMatrix{11,11}(@view E_pl[12:end, 1:11])
    dEdTRF[t, r, g] = u_fp * ((E_pl2 - (1 / 2 * H_fp * E_pl1)) * u_rot - (1 / 2 * E_pl1 * u_rot * H_fp)) * u_fp
    return nothing
end

function calculate_crushed_pulse_propagator!(E, dEdω1, dEdTRF, t, r, g, ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad, u_rot, _, _)
    u_fp = exp(hamiltonian_linear(0, B1, ω0, TR / 2, m0s, R1f, R2f, Rex, R1s, 0, 0, 0, grad))
    u_fp = xs_destructor(u_fp) * u_fp
    u_pl = propagator_linear_crushed_pulse(ω1, TRF, B1,
        R2slT[1](TRF, ω1 * TRF, B1, T2s),
        R2slT[2](TRF, ω1 * TRF, B1, T2s),
        R2slT[3](TRF, ω1 * TRF, B1, T2s),
        grad)
    E[t, r, g] = u_fp * u_pl * u_rot * u_fp
    dEdω1[t, r, g] = @SMatrix zeros(11, 11)
    dEdTRF[t, r, g] = @SMatrix zeros(11, 11)
    return nothing
end

function calcualte_cycle_propgator(E)
    Q = Array{SMatrix{11,11,Float64}}(undef, size(E, 2), size(E, 3))

    for g ∈ axes(E, 3), r ∈ axes(E, 2)
        A = E[1, r, g]
        for t = size(E, 1):-1:2
            A = A * E[t, r, g]
        end
        Q[r, g] = A0(A) - A
    end
    return Q
end

function propagate_magnetization(Q, E)
    Y = similar(E, SVector{11,Float64})

    for r ∈ axes(E, 2), g ∈ axes(E, 3)
        m = Q[r, g] \ C(Q[r, g])

        Y[1, r, g] = m
        for t = 2:size(E, 1)
            m = E[t, r, g] * m
            Y[t, r, g] = m
        end
    end
    return Y
end

# the commented line in this function are required for the adjoint state to be correct; but since these entries are not used for the OCT algorithm, we skip calculating them.
function calculate_adjoint_state(d, Q, E, iSeq)
    P = similar(E, Array{Float64})
    λ = @view P[end, :, :]

    for g in 1:size(E, 3), r in 1:size(E, 2)
        λ[r, g] = d(size(E, 1) + (iSeq - 1) * size(E, 1), r, g)
    end

    for r ∈ axes(E, 2)
        for t = size(E, 1)-1:-1:1
            λ[r, 1] = transpose(E[t+1, r, 1]) * λ[r, 1]
            λ[r, 1] += d(t + (iSeq - 1) * size(E, 1), r, 1)
            for g = 2:size(E, 3)
                λ[r, g] = transpose(E[t+1, r, g][6:10, :]) * λ[r, g][6:10]
                λ[r, 1][1:5] .+= λ[r, g][1:5]
                # λ[r,1][end]  += λ[r,g][end]
                λ[r, g] .+= d(t + (iSeq - 1) * size(E, 1), r, g)
            end
        end

        # for g = 2:length(grad_list)
        #     λ[r,g][1:5] .= λ[r,1][1:5]
        #     λ[r,g][end]  = λ[r,1][end]
        # end

        P[end, r, 1] = inv(transpose(Q[r, 1])) * λ[r, 1]
        for g = 2:size(E, 3)
            λ[r, g] = inv(transpose(Q[r, g]))[:, 6:10] * λ[r, g][6:10]
            λ[r, 1][1:5] .+= λ[r, g][1:5]
            # P[end,r,1][end]  += P[end,r,g][end]
        end

        # step 5: propagate adjoint state
        for t in size(E, 1):-1:2
            for g ∈ axes(E, 3)
                _E = E[mod(t, size(E, 1))+1, r, g]
                if g == 1
                    P[t-1, r, g] = transpose(_E) * P[t, r, g]
                else
                    P[t-1, r, g] = transpose(_E[6:10, :]) * P[t, r, g][6:10]
                    P[t-1, r, 1][1:5] .+= P[t-1, r, g][1:5]
                    P[t-1, r, 1][end] += P[t-1, r, g][end]
                end
                P[t-1, r, g] .+= d(t + (iSeq - 1) * size(E, 1), r, g)
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

    for g2 in 0:size(Y, 3), g1 in 0:size(Y, 3), r ∈ axes(Y, 2), t ∈ axes(Y, 1)
        if g1 == 0
            s1 = Y[t, r, 1][1] - 1im * Y[t, r, 1][2]
        else
            s1 = Y[t, r, g1][6] - 1im * Y[t, r, g1][7]
        end
        if g2 == 0
            s2 = Y[t, r, 1][1] + 1im * Y[t, r, 1][2]
        else
            s2 = Y[t, r, g2][6] + 1im * Y[t, r, g2][7]
        end
        F[g1+1, g2+1] += s1 * s2
    end

    Fi = inv(F)
    CRB = w * real.(diag(Fi))

    for g1 in 0:size(Y, 3), r ∈ axes(Y, 2), t ∈ axes(Y, 1)
        # derivative wrt. x
        dFdy .= 0
        dFdy[g1+1, 1] = Y[t, r, 1][1] + 1im * Y[t, r, 1][2]
        dFdy[1, g1+1] = Y[t, r, 1][1] - 1im * Y[t, r, 1][2]
        for g2 ∈ axes(Y, 3)
            dFdy[g1+1, g2+1] = Y[t, r, g2][6] + 1im * Y[t, r, g2][7]
            dFdy[g2+1, g1+1] = Y[t, r, g2][6] - 1im * Y[t, r, g2][7]
        end
        dFdy[g1+1, g1+1] = 2 * real(dFdy[g1+1, g1+1])
        mul!(dFdy, Fi, mul!(tmp, dFdy, Fi))
        _dCRBdx[t, r, g1+1] = real.(w * diag(dFdy))

        # derivative wrt. y
        dFdy .= 0
        dFdy[g1+1, 1] = Y[t, r, 1][2] - 1im * Y[t, r, 1][1]
        dFdy[1, g1+1] = Y[t, r, 1][2] + 1im * Y[t, r, 1][1]
        for g2 ∈ axes(Y, 3)
            dFdy[g1+1, g2+1] = Y[t, r, g2][7] - 1im * Y[t, r, g2][6]
            dFdy[g2+1, g1+1] = Y[t, r, g2][7] + 1im * Y[t, r, g2][6]
        end
        dFdy[g1+1, g1+1] = 2 * real(dFdy[g1+1, g1+1])
        mul!(dFdy, Fi, mul!(tmp, dFdy, Fi))
        _dCRBdy[t, r, g1+1] = real.(w * diag(dFdy))
    end

    d(t, r, g) = @SVector [_dCRBdx[t, r, 1], _dCRBdy[t, r, 1], 0, 0, 0, _dCRBdx[t, r, g+1], _dCRBdy[t, r, g+1], 0, 0, 0, 0]

    return CRB, d
end

function calculate_gradient_inner_product(P, Y, E, dEdω1, dEdTRF)
    grad_ω1 = zeros(size(E, 1))
    grad_TRF = zeros(size(E, 1))

    for g ∈ axes(Y, 3), r ∈ axes(Y, 2)
        if g == 1
            grad_ω1[1] -= transpose(P[end, r, g]) * (dEdω1[1, r, g] * Y[end, r, g])
            grad_TRF[1] -= transpose(P[end, r, g]) * (dEdTRF[1, r, g] * Y[end, r, g])
        else
            a = dEdω1[1, r, g] * Y[end, r, g]
            b = dEdTRF[1, r, g] * Y[end, r, g]
            @inbounds for i = 6:10
                grad_ω1[1] -= P[end, r, g][i] * a[i]
                grad_TRF[1] -= P[end, r, g][i] * b[i]
            end
        end
    end

    for g ∈ axes(Y, 3), r ∈ axes(Y, 2), t ∈ 2:size(Y, 1)
        if g == 1
            grad_ω1[t] -= transpose(P[t-1, r, g]) * (dEdω1[t, r, g] * Y[t-1, r, g])
            grad_TRF[t] -= transpose(P[t-1, r, g]) * (dEdTRF[t, r, g] * Y[t-1, r, g])
        else
            a = dEdω1[t, r, g] * Y[t-1, r, g]
            b = dEdTRF[t, r, g] * Y[t-1, r, g]
            @inbounds for i = 6:10
                grad_ω1[t] -= P[t-1, r, g][i] * a[i]
                grad_TRF[t] -= P[t-1, r, g][i] * b[i]
            end
        end
    end
    return grad_ω1, grad_TRF
end