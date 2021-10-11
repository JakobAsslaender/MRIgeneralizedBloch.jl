function OCT_gradient(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT, grad_list, weights)
    (E, dEdω1, dEdTRF) = MRIgeneralizedBloch.calculate_propagators_ω1(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT, grad_list)
    Q = calcualte_cycle_propgator(E)
    Y = propagate_magnetization(Q, E)
    (CRB, d) = dLdy(Y, weights)
    P = calculate_adjoint_state(d, Q, E)
    (grad_ω1, grad_TRF) = calculate_gradient_inner_product(P, Y, E, dEdω1, dEdTRF)
    return (CRB, grad_ω1, grad_TRF)
end


function calculate_propagators_ω1(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, R2slT, grad_list::AbstractArray{T}; rfphase_increment=[π]) where T <: grad_param
    E      = Array{SMatrix{11,11,Float64,121}}(undef, length(ω1), length(rfphase_increment), length(grad_list))
    dEdω1  = Array{SMatrix{11,11,Float64,121}}(undef, length(ω1), length(rfphase_increment), length(grad_list))
    dEdTRF = Array{SMatrix{11,11,Float64,121}}(undef, length(ω1), length(rfphase_increment), length(grad_list))

    # R2s_vec = evaluate_R2sl_vector_OCT(ω1 .* TRF, TRF, B1, T2s, R2slT, grad_list)
    cache = [ExponentialUtilities.alloc_mem(zeros(22, 22), ExpMethodHigham2005Base()) for _ = 1:Threads.nthreads()]
    dH = [zeros(Float64, 22, 22) for _ = 1:Threads.nthreads()]

    for r in eachindex(rfphase_increment)
        u_rot = z_rotation_propagator(rfphase_increment[r], grad_m0s())
        Threads.@threads for t in 1:length(ω1)
            for g in eachindex(grad_list)
                if t == 1
                    u_fp = xs_destructor(grad_list[g]) * exp(hamiltonian_linear(0, B1, ω0, TR / 2, m0s, R1, R2f, Rx, 0, 0, 0, grad_list[g]))
                    u_pl = propagator_linear_inversion_pulse(ω1[1], TRF[1], B1, 
                    R2slT[1](TRF[1], ω1[1] * TRF[1], B1, T2s), 
                    R2slT[2](TRF[1], ω1[1] * TRF[1], B1, T2s), 
                    R2slT[3](TRF[1], ω1[1] * TRF[1], B1, T2s), 
                    grad_list[g])
                    E[1,r,g] = u_fp * u_pl * u_rot * u_fp
                    dEdω1[1,r,g] = @SMatrix zeros(11, 11)
                    dEdTRF[1,r,g] = @SMatrix zeros(11, 11)
                else
                    H_fp = hamiltonian_linear(0, B1, ω0, 1, m0s, R1, R2f, Rx, 0, 0, 0, grad_list[g])
                    u_fp = xs_destructor(grad_list[g]) * exp(H_fp * ((TR - TRF[t]) / 2))

                    H_pl = hamiltonian_linear(ω1[t], B1, ω0, 1, m0s, R1, R2f, Rx, 
                    R2slT[1](TRF[t], ω1[t] * TRF[t], B1, T2s), 
                    R2slT[2](TRF[t], ω1[t] * TRF[t], B1, T2s), 
                    R2slT[3](TRF[t], ω1[t] * TRF[t], B1, T2s), 
                    grad_list[g])

                    dHdω1 = d_hamiltonian_linear_dω1(B1, 1, 
                    R2slT[4](TRF[t], ω1[t] * TRF[t], B1, T2s), 
                    R2slT[6](TRF[t], ω1[t] * TRF[t], B1, T2s), 
                    R2slT[7](TRF[t], ω1[t] * TRF[t], B1, T2s), 
                    grad_list[g])
        
                    @views dH[Threads.threadid()][1:11,1:11]   .= H_pl
                    @views dH[Threads.threadid()][12:22,12:22] .= H_pl
                    @views dH[Threads.threadid()][1:11,12:22]  .= 0
                    @views dH[Threads.threadid()][12:22,1:11]  .= dHdω1
                    dH[Threads.threadid()] .*= TRF[t]
                    E_pl = exponential!(dH[Threads.threadid()], ExpMethodHigham2005Base(), cache[Threads.threadid()])

                    E_pl1 = SMatrix{11,11}(@view E_pl[1:11,1:11])
                    E_pl2 = SMatrix{11,11}(@view E_pl[12:end,1:11])
                    E[t,r,g]     = u_fp * E_pl1 * u_rot * u_fp
                    dEdω1[t,r,g] = u_fp * E_pl2 * u_rot * u_fp

                    # TRF
                    dHdTRF = H_pl .+ d_hamiltonian_linear_dTRF_add(TRF[t], 
                    R2slT[5](TRF[t], ω1[t] * TRF[t], B1, T2s), 
                    R2slT[8](TRF[t], ω1[t] * TRF[t], B1, T2s), 
                    R2slT[9](TRF[t], ω1[t] * TRF[t], B1, T2s), 
                    grad_list[g])
                    H_pl *= TRF[t]
                    @views dH[Threads.threadid()][1:11,1:11]   .= H_pl
                    @views dH[Threads.threadid()][12:22,12:22] .= H_pl
                    @views dH[Threads.threadid()][1:11,12:22]  .= 0
                    @views dH[Threads.threadid()][12:22,1:11]  .= dHdTRF
                    E_pl = exponential!(dH[Threads.threadid()], ExpMethodHigham2005Base(), cache[Threads.threadid()])

                    E_pl1 = SMatrix{11,11}(@view E_pl[1:11,1:11])
                    E_pl2 = SMatrix{11,11}(@view E_pl[12:end,1:11])
                    dEdTRF[t,r,g] = u_fp * ((E_pl2 - (1 / 2 * H_fp * E_pl1)) * u_rot - (1 / 2 * E_pl1 * u_rot * H_fp)) * u_fp
                end
            end
        end
    end
    return (E, dEdω1, dEdTRF)
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
        m = Q[r,g] \ MRIgeneralizedBloch.C(grad_m0s())
    
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
    λ = @views P[end,:,:]

    for g in 1:size(E, 3), r in 1:size(E, 2)
        λ[r,g] = d(size(E, 1), r, g)
    end

    for r = 1:size(E, 2)
        for t = size(E, 1) - 1:-1:1
            λ[r,1] = transpose(E[t + 1,r,1]) * λ[r,1]
            λ[r,1] += d(t, r, 1)
            for g = 2:size(E, 3)
                @views λ[r,g] = transpose(E[t + 1,r,g][6:10,:]) * λ[r,g][6:10]
                @views λ[r,1][1:5] .+= λ[r,g][1:5]
                # λ[r,1][end]  += λ[r,g][end]
                λ[r,g] += d(t, r, g)
            end
        end

    # for g = 2:length(grad_list)
    #     λ[r,g][1:5] .= λ[r,1][1:5]
    #     λ[r,g][end]  = λ[r,1][end]
    # end

        P[end,r,1] = inv(transpose(Q[r,1])) * λ[r,1]
        for g = 2:size(E, 3)
            @views λ[r,g] = inv(transpose(Q[r,g]))[:,6:10] * λ[r,g][6:10]
            @views λ[r,1][1:5] .+= λ[r,g][1:5]
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
                P[t - 1,r,g] += d(t, r, g)
            end

        # for g = 2:length(grad_list)
        #     P[t - 1,r,g][1:5] .= P[t - 1,r,1][1:5]
        #     P[t - 1,r,g][end]  = P[t - 1,r,1][end]
        # end
        end
    end
    return P
end

# function dLdy_old(y, w)
#     F = y' * y
#     println(F)
#     Fi = inv(F)

#     _dLdy = similar(y)
#     dFdy = similar(F)
#     tmp  = similar(F)

#     for t in 1:size(y, 1), g in 1:size(y, 2)
#         dFdy .= 0
#         for i = 1:size(y, 2)
#             dFdy[i,g] = conj(y[t,i])
#             dFdy[g,i] = y[t,i]
#         end
#         dFdy[g,g] = 2 * real(dFdy[g,g])
        
#         mul!(dFdy, Fi, mul!(tmp, dFdy, Fi))
#         _dLdy[t,g] = w * diag(dFdy)
#     end
#     return _dLdy    
# end

function dLdy(Y, w)
    # d = similar(Y, SVector{11,Float64})
    _dLdx = Array{Float64}(undef, size(Y, 1), size(Y, 2), size(Y, 3) + 1)
    _dLdy = similar(_dLdx)
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
    CRB = real.(w * diag(Fi))

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
        _dLdx[t,r,g1 + 1] = real.(w * diag(dFdy))

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
        _dLdy[t,r,g1 + 1] = real.(w * diag(dFdy))
    end

    d(t, r, g) = @SVector [_dLdx[t,r,1], _dLdy[t,r,1],0,0,0,_dLdx[t,r,g + 1], _dLdy[t,r,g + 1],0,0,0,0]

    return (CRB, d)
end

function calculate_gradient_inner_product(P, Y, E, dEdω1, dEdTRF)
    grad_ω1 = zeros(size(E, 1))
    grad_TRF = zeros(size(E, 1))

    for g in 1:size(E, 3), r in 1:size(E, 2), t in 2:size(E, 1)
        if g == 1
            grad_ω1[t]  -= transpose(P[t - 1,r,g]) * (dEdω1[t,r,g] * Y[t - 1,r,g])
            grad_TRF[t] -= transpose(P[t - 1,r,g]) * (dEdTRF[t,r,g] * Y[t - 1,r,g])
        else
            grad_ω1[t]  -= transpose(P[t - 1,r,g][6:10]) * (dEdω1[t,r,g][6:10,:] * Y[t - 1,r,g])
            grad_TRF[t] -= transpose(P[t - 1,r,g][6:10]) * (dEdTRF[t,r,g][6:10,:] * Y[t - 1,r,g])
        end
    end
    return (grad_ω1, grad_TRF)
end