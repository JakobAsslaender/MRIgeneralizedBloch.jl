"""
    CRB, grad_ω1, grad_TRF = CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, K, nTR, R1s, T2s, R2slT, grad_list, weights; grad_moment=[i[1] == 1 ? :spoiler_dual : :balanced for i ∈ CartesianIndices(ω1)])

Calculate the Cramer-Rao bound of a pulse sequence along with the derivatives wrt. `ω1` and `TRF`.

# Arguments
- `ω1::Vector{Real}`: Control vector of `length = Npulses` (matrix if more than 1 sequence are optimized)
- `TRF::Vector{Real}`: Control vector of `length = Npulses` (matrix if more than 1 sequence are optimized)
- `TR::Real`: Repetition time in seconds
- `ω0::Real`: Off-resonance frequency in rad/s
- `B1::Real`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `m0s::Real`: Fractional size of the semi-solid pool; should be in range of 0 to 1
- `R1f::Real`: Longitudinal relaxation rate of the free pool in 1/seconds
- `R2f::Real`: Transversal relaxation rate of the free pool in 1/seconds
- `K, nTR::Real`: Exchange rate between the two spin pools in 1/seconds
- `R1f::Real`: Longitudinal relaxation rate of the semi-solid pool in 1/seconds
- `T2s::Real`: Transversal relaxation time of the semi-solid pool in seconds
- `R2slT::NTuple{3, Function}`: Tuple of three functions: R2sl(TRF, ω1, B1, T2s), dR2sldB1(TRF, ω1, B1, T2s), and R2sldT2s(TRF, ω1, B1, T2s). Can be generated with [`precompute_R2sl`](@ref)
- `grad_list::Tuple{<:grad_param}`: Tuple that specifies the gradients that are calculated; the vector elements can either be any subset/order of `grad_list=(grad_m0s(), grad_R1f(), grad_R2f(), grad_k(), grad_K(), grad_nTR(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1())`; the derivative wrt. to apparent `R1a = R1f = R1s` can be calculated with `grad_R1a()`
- `weights::transpose(Vector{Real})`: Row vector of weights applied to the Cramer-Rao bounds (CRB) of the individual parameters. The first entry always refers to the CRB of M0, followed by the values defined in `grad_list` in the order defined therein. Hence, the vector `weights` has to have one more entry than `grad_list`

# Optional Keyword Arguments:
- `grad_moment=[i[1] == 1 ? :spoiler_dual : :balanced for i ∈ CartesianIndices(ω1)]`: Different types of gradient moments of each TR are possible (`:balanced`, `:crusher`, `:spoiler_dual`, `:spoiler_prepulse`). `:balanced` simulates a TR with all gradient moments nulled. `:crusher` assumes equivalent (non-zero) gradient moments before and simulates the refocussing path of the extended phase graph. `:spoiler_prepulse` nulls all transverse magnetization before the RF pulse, emulating an idealized FLASH. `:spoiler_dual` nulls all transverse magnetization before and after the RF pulse.

# Examples
```jldoctest
julia> CRB, grad_ω1, grad_TRF = MRIgeneralizedBloch.CRB_gradient_OCT(range(pi/2, π, 100), range(100e-6, 400e-6, 100), 3.5e-3, 0, 1, 0.15, 0.5, 15, 30, 4, 10e-6, precompute_R2sl(), [grad_m0s(), grad_R2f()], transpose([0, 1, 1]))
(7.036532949438835e17, [-3.390796354537386e15, -1.9839673463682364e16, 1.9478009146395284e16, -1.8267985259974836e16, 1.6854024503161112e16, -1.564870176592121e16, 1.3903706541983584e16, -1.2606082554973334e16, 1.1194927161388668e16, -9.539757823647384e15  …  2.5718054915661044e16, -8.337218806299867e16, 4.778792906164447e16, -8.836386285134571e16, 7.1006544616540664e16, -9.901490876727578e16, 9.339003931389179e16, -1.1880577663606485e17, 1.1196316998153437e17, -1.526589221445061e17], [-6.260353922953729e19, -3.1566204597354915e20, 2.8346374467477307e20, -2.8268175999840482e20, 2.331816059339016e20, -2.376712332529386e20, 1.816604170405834e20, -1.9023039511824227e20, 1.3663891126875767e20, -1.4586637093661966e20  …  3.0242994279981213e20, -5.942607397341693e20, 4.384537591620554e20, -6.72590448103838e20, 5.776809344770309e20, -7.977977759334612e20, 7.085448466252249e20, -9.913215504590694e20, 8.145557315154696e20, -1.2823732424849773e21])

```
c.f. [Optimal Control](@ref)
"""
function CRB_gradient_OCT(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, K, nTR, R1s, T2s, R2slT, grad_list, weights; grad_moment=[i[1] == 1 ? :spoiler_dual : :balanced for i ∈ CartesianIndices(ω1)])
    nSeq = size(ω1, 2)

    E = Vector{Matrix{SMatrix{11,11,Float64}}}(undef, nSeq)
    dEdω1 = similar(E)
    dEdTRF = similar(E)
    Q = Vector{Vector{SMatrix{11,11,Float64}}}(undef, nSeq)
    Y = Vector{Matrix{SVector{11,Float64}}}(undef, nSeq)

    grad_ω1 = similar(ω1)
    grad_TRF = similar(ω1)

    Threads.@threads for iSeq ∈ eachindex(E)
        E[iSeq], dEdω1[iSeq], dEdTRF[iSeq] = @views calculate_propagators_ω1(ω1[:, iSeq], TRF[:, iSeq], TR, ω0, B1, m0s, R1f, R2f, K, nTR, R1s, T2s, R2slT, grad_list; grad_moment=grad_moment[:, iSeq])
        Q[iSeq] = calcualte_cycle_propgator(E[iSeq])
        Y[iSeq] = propagate_magnetization(Q[iSeq], E[iSeq])
    end

    CRB, d = dCRBdm(Y, weights)

    for iSeq ∈ eachindex(E)
        P = calculate_adjoint_state(d[iSeq], Q[iSeq], E[iSeq])
        grad_ω1[:, iSeq], grad_TRF[:, iSeq] = calculate_gradient_inner_product(P, Y[iSeq], E[iSeq], dEdω1[iSeq], dEdTRF[iSeq])
    end

    return CRB, grad_ω1, grad_TRF
end


function calculate_propagators_ω1(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, K, nTR, R1s, T2s, R2slT, grad_list; grad_moment)
    E      = Array{SMatrix{11,11,Float64,121}}(undef, length(ω1), length(grad_list))
    dEdω1  = Array{SMatrix{11,11,Float64,121}}(undef, length(ω1), length(grad_list))
    dEdTRF = Array{SMatrix{11,11,Float64,121}}(undef, length(ω1), length(grad_list))

    cache = ExponentialUtilities.alloc_mem(zeros(22, 22), ExpMethodHigham2005Base())
    dH = zeros(Float64, 22, 22)

    u_rot = z_rotation_propagator(π, grad_m0s())
    for g ∈ eachindex(grad_list)
        grad = grad_list[g]

        for t ∈ 1:length(ω1)
            if grad_moment[t] == :crusher
                calculate_crushed_pulse_propagator!(E, dEdω1, dEdTRF, t, g, ω1[t], TRF[t], TR, ω0, B1, m0s, R1f, R2f, K, nTR, R1s, T2s, R2slT, grad, u_rot, dH, cache)
            else
                calculte_propagator!(E, dEdω1, dEdTRF, t, g, ω1[t], TRF[t], TR, ω0, B1, m0s, R1f, R2f, K, nTR, R1s, T2s, R2slT, grad, u_rot, dH, cache, grad_moment[t])
            end
        end
    end
    return E, dEdω1, dEdTRF
end

function calculte_propagator!(E, dEdω1, dEdTRF, t, g, ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, K, nTR, R1s, T2s, R2slT, grad, u_rot, dH, cache, grad_moment)
    H_fp = hamiltonian_linear(0.0, B1, ω0, 1.0, m0s, R1f, R2f, K, nTR, R1s, 0.0, 0.0, 0.0, grad)
    ux = grad_moment == :spoiler_dual ? xy_destructor(H_fp) : xs_destructor(H_fp)
    u_fp = ux * exp(H_fp * ((TR - TRF) / 2))

    H_pl = hamiltonian_linear(ω1, B1, ω0, 1, m0s, R1f, R2f, K, nTR, R1s,
        R2slT[1](TRF, ω1 * TRF, B1, T2s),
        R2slT[2](TRF, ω1 * TRF, B1, T2s),
        R2slT[3](TRF, ω1 * TRF, B1, T2s),
        grad)

    dHdω1 = d_hamiltonian_linear_dω1(B1, 1,
        R2slT[4](TRF, ω1 * TRF, B1, T2s),
        R2slT[6](TRF, ω1 * TRF, B1, T2s),
        R2slT[7](TRF, ω1 * TRF, B1, T2s),
        grad)

    @views dH[1:11, 1:11] .= H_pl
    @views dH[12:22, 12:22] .= H_pl
    @views dH[1:11, 12:22] .= 0
    @views dH[12:22, 1:11] .= dHdω1
    dH .*= TRF
    E_pl = exponential!(dH, ExpMethodHigham2005Base(), cache)

    E_pl1 = SMatrix{11,11}(@view E_pl[1:11, 1:11])
    E_pl2 = SMatrix{11,11}(@view E_pl[12:end, 1:11])
    E[t, g] = u_fp * E_pl1 * u_rot * u_fp
    dEdω1[t, g] = u_fp * E_pl2 * u_rot * u_fp

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
    dEdTRF[t, g] = u_fp * ((E_pl2 - (1 / 2 * H_fp * E_pl1)) * u_rot - (1 / 2 * E_pl1 * u_rot * H_fp)) * u_fp
    return nothing
end

function calculate_crushed_pulse_propagator!(E, dEdω1, dEdTRF, t, g, ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, K, nTR, R1s, T2s, R2slT, grad, u_rot, _, _)
    u_fp = exp(hamiltonian_linear(0, B1, ω0, TR / 2, m0s, R1f, R2f, K, nTR, R1s, 0, 0, 0, grad))
    u_fp = xs_destructor(u_fp) * u_fp
    u_pl = propagator_linear_crushed_pulse(ω1, TRF, B1,
        R2slT[1](TRF, ω1 * TRF, B1, T2s),
        R2slT[2](TRF, ω1 * TRF, B1, T2s),
        R2slT[3](TRF, ω1 * TRF, B1, T2s),
        grad)
    E[t, g] = u_fp * u_pl * u_rot * u_fp
    dEdω1[t, g] = @SMatrix zeros(11, 11)
    dEdTRF[t, g] = @SMatrix zeros(11, 11)
    return nothing
end

function calcualte_cycle_propgator(E)
    Q = Vector{SMatrix{11,11,Float64}}(undef, size(E, 2))

    for g ∈ axes(E, 2)
        A = E[1, g]
        for t = size(E, 1):-1:2
            A = A * E[t, g]
        end
        Q[g] = A0(A) - A
    end
    return Q
end

function propagate_magnetization(Q, E)
    Y = similar(E, SVector{11,Float64})

    for g ∈ axes(E, 2)
        m = Q[g] \ C(Q[g])
        Y[1, g] = m
        for t = 2:size(E, 1)
            m = E[t, g] * m
            Y[t, g] = m
        end
    end
    return Y
end

# the commented line in this function are required for the adjoint state to be correct; but since these entries are not used for the OCT algorithm, we skip calculating them.
function calculate_adjoint_state(d, Q, E)
    P = similar(E, Array{Float64})
    λ = @view P[end, :]

    for g ∈ 1:size(E, 2)
        λ[g] = d(size(E, 1), g)
    end

    for t = size(E, 1)-1:-1:1
        λ[1] = transpose(E[t+1, 1]) * λ[1]
        λ[1] += d(t, 1)
        for g = 2:size(E, 2)
            λ[g] = transpose(E[t+1, g][6:10, :]) * λ[g][6:10]
            λ[1][1:5] .+= λ[g][1:5]
            # λ[1][end]  += λ[g][end]
            λ[g] .+= d(t, g)
        end
    end

    # for g = 2:length(grad_list)
    #     λ[g][1:5] .= λ[1][1:5]
    #     λ[g][end]  = λ[1][end]
    # end

    P[end, 1] = inv(transpose(Q[1])) * λ[1]
    for g = 2:size(E, 2)
        λ[g] = inv(transpose(Q[g]))[:, 6:10] * λ[g][6:10]
        λ[1][1:5] .+= λ[g][1:5]
        # P[end,1][end]  += P[end,g][end]
    end

    # step 5: propagate adjoint state
    for t ∈ size(E, 1):-1:2
        for g ∈ axes(E, 2)
            _E = E[mod(t, size(E, 1))+1, g]
            if g == 1
                P[t-1, g] = transpose(_E) * P[t, g]
            else
                P[t-1, g] = transpose(_E[6:10, :]) * P[t, g][6:10]
                P[t-1, 1][1:5] .+= P[t-1, g][1:5]
                P[t-1, 1][end] += P[t-1, g][end]
            end
            P[t-1, g] .+= d(t, g)
        end

        # for g = 2:length(grad_list)
        #     P[t - 1,g][1:5] .= P[t - 1,1][1:5]
        #     P[t - 1,g][end]  = P[t - 1,1][end]
        # end
    end
    return P
end

function dCRBdm(Y, w)
    N_grad = size(Y[1], 2) # w/o M0

    F = zeros(ComplexF64, N_grad + 1, N_grad + 1)
    for iSeq ∈ eachindex(Y), g2 ∈ 0:N_grad, g1 ∈ 0:N_grad, t ∈ axes(Y[iSeq], 1)
        s1 = g1 == 0 ? Y[iSeq][t, 1][1] - 1im * Y[iSeq][t, 1][2] : Y[iSeq][t, g1][6] - 1im * Y[iSeq][t, g1][7]
        s2 = g2 == 0 ? Y[iSeq][t, 1][1] + 1im * Y[iSeq][t, 1][2] : Y[iSeq][t, g2][6] + 1im * Y[iSeq][t, g2][7]
        F[g1+1, g2+1] += s1 * s2
    end
    Fi = inv(F)
    CRB = w * real.(diag(Fi))

    _dCRBdx = [Array{Float64}(undef, size(Yi, 1), size(Yi, 2) + 1) for Yi ∈ Y]
    _dCRBdy = [Array{Float64}(undef, size(Yi, 1), size(Yi, 2) + 1) for Yi ∈ Y]
    dFdy = similar(F)
    tmp = similar(F)
    for iSeq ∈ eachindex(Y), g1 ∈ 0:N_grad, t ∈ axes(Y[iSeq], 1)
        # derivative wrt. x
        dFdy .= 0
        dFdy[g1+1, 1] = Y[iSeq][t, 1][1] + 1im * Y[iSeq][t, 1][2]
        dFdy[1, g1+1] = Y[iSeq][t, 1][1] - 1im * Y[iSeq][t, 1][2]
        for g2 ∈ axes(Y[iSeq], 2)
            dFdy[g1+1, g2+1] = Y[iSeq][t, g2][6] + 1im * Y[iSeq][t, g2][7]
            dFdy[g2+1, g1+1] = Y[iSeq][t, g2][6] - 1im * Y[iSeq][t, g2][7]
        end
        dFdy[g1+1, g1+1] = 2 * real(dFdy[g1+1, g1+1])
        mul!(dFdy, Fi, mul!(tmp, dFdy, Fi))
        _dCRBdx[iSeq][t, g1+1] = real.(w * diag(dFdy))

        # derivative wrt. y
        dFdy .= 0
        dFdy[g1+1, 1] = Y[iSeq][t, 1][2] - 1im * Y[iSeq][t, 1][1]
        dFdy[1, g1+1] = Y[iSeq][t, 1][2] + 1im * Y[iSeq][t, 1][1]
        for g2 ∈ axes(Y[iSeq], 2)
            dFdy[g1+1, g2+1] = Y[iSeq][t, g2][7] - 1im * Y[iSeq][t, g2][6]
            dFdy[g2+1, g1+1] = Y[iSeq][t, g2][7] + 1im * Y[iSeq][t, g2][6]
        end
        dFdy[g1+1, g1+1] = 2 * real(dFdy[g1+1, g1+1])
        mul!(dFdy, Fi, mul!(tmp, dFdy, Fi))
        _dCRBdy[iSeq][t, g1+1] = real.(w * diag(dFdy))
    end

    d = [(t, g) -> @SVector [_dCRBdx[iSeq][t, 1], _dCRBdy[iSeq][t, 1], 0, 0, 0, _dCRBdx[iSeq][t, g+1], _dCRBdy[iSeq][t, g+1], 0, 0, 0, 0] for iSeq ∈ eachindex(_dCRBdx)]

    return CRB, d
end

function calculate_gradient_inner_product(P, Y, E, dEdω1, dEdTRF)
    grad_ω1 = zeros(size(E, 1))
    grad_TRF = zeros(size(E, 1))

    for g ∈ axes(Y, 2), t ∈ axes(Y, 1)
        tm1 = mod1(t - 1, size(Y, 1))
        if g == 1
            grad_ω1[t] -= transpose(P[tm1, g]) * (dEdω1[t, g] * Y[tm1, g])
            grad_TRF[t] -= transpose(P[tm1, g]) * (dEdTRF[t, g] * Y[tm1, g])
        else
            a = dEdω1[t, g] * Y[tm1, g]
            b = dEdTRF[t, g] * Y[tm1, g]
            @inbounds for i = 6:10
                grad_ω1[t] -= P[tm1, g][i] * a[i]
                grad_TRF[t] -= P[tm1, g][i] * b[i]
            end
        end
    end
    return grad_ω1, grad_TRF
end