function calc_CRB(s, w)
    s = reshape(s, size(s,1)*size(s,2), size(s,3))
    real.(w * diag(inv(s' * s)))
end

function grad_ω1_fd(w, ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list)
    Δω1 = 1e-2
    _grad_ω1 = similar(ω1)

    α = ω1 .* TRF
    s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=grad_list)
    CRB0 = calc_CRB(s0, w)

    for t in eachindex(ω1)
        α[t] = (ω1[t] + Δω1) * TRF[t]
        
        ds = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=grad_list)
        CRB1 = calc_CRB(ds, w)

        α[t] = ω1[t] .* TRF[t]

        _grad_ω1[t] = (CRB1 - CRB0) / Δω1
    end
    _grad_ω1[1] = 0
    return _grad_ω1  
end

function grad_TRF_fd(w, ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list)
    ΔTRF = 1e-9
    _grad_TRF = similar(ω1)

    α = ω1 .* TRF
    s0 = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=grad_list)
    CRB0 = calc_CRB(s0, w)

    for t in eachindex(ω1)
        TRF[t] += ΔTRF
        α[t] = ω1[t] * TRF[t]
        
        ds = calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list=grad_list)
        CRB1 = calc_CRB(ds, w)
        _grad_TRF[t] = (CRB1 - CRB0) / ΔTRF

        TRF[t] -= ΔTRF
        α[t] = ω1[t] * TRF[t]
    end
    _grad_TRF[1] = 0
    return _grad_TRF  
end



function dCRBdm_fd(m, w)
    _dCRBdx = similar(m, Float64)
    _dCRBdy = similar(m, Float64)
    Δm = 1e-9
    
    CRB0 = calc_CRB(m, w)
    for i in eachindex(m)
        dm = copy(m)
        dm[i] = m[i] .+ Δm 
        _dCRBdx[i] = real(CRB0 - calc_CRB(dm, w)) / Δm

        dm = copy(m)
        dm[i] = m[i] + 1im * Δm 
        _dCRBdy[i] += real(CRB0 - calc_CRB(dm, w)) / Δm
    end

    d(t, r, g) = @SVector [_dCRBdx[t,r,1], _dCRBdy[t,r,1],0,0,0,_dCRBdx[t,r,g + 1], _dCRBdy[t,r,g + 1],0,0,0,0]

    return (CRB0, d)
end



###########################
function grad_TV_ω1_fd(weights, ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, λ_TV)
    Δω1 = 1e-2
    _grad_ω1_fd = similar(ω1)

    (F0, _grad_TV_ω1, _) = MRIgeneralizedBloch.OCT_TV_gradient(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, weights, λ_TV)

    for t in eachindex(ω1)
        ω1[t] += Δω1
        (F1, _, _) = MRIgeneralizedBloch.OCT_TV_gradient(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rx, R1s, T2s, R2slT, grad_list, weights, λ_TV)
        ω1[t] -= Δω1

        _grad_ω1_fd[t] = (F1 - F0) / Δω1
    end
    _grad_ω1_fd[1] = 0
    return (_grad_TV_ω1, _grad_ω1_fd)
end