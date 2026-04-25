# ============================================================================
# Deprecated function names — will be removed in a future release
# ============================================================================

# Renamed in v0.11: calculatesignal_* → simulate_*
# Updated in v0.12: insert M0=1 since the old functions did not have M0
function calculatesignal_linearapprox(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT; kwargs...)
    Base.depwarn("`calculatesignal_linearapprox` is deprecated, use `simulate_linearapprox` instead (note the added `M0` argument).", :calculatesignal_linearapprox)
    simulate_linearapprox(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT; kwargs...)
end
export calculatesignal_linearapprox

function calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s; kwargs...)
    Base.depwarn("`calculatesignal_gbloch_ide` is deprecated, use `simulate_gbloch_ide` instead (note the added `M0` argument).", :calculatesignal_gbloch_ide)
    simulate_gbloch_ide(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, Rex, R1s, T2s; kwargs...)
end
export calculatesignal_gbloch_ide

function calculatesignal_graham_ode(α, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s; kwargs...)
    Base.depwarn("`calculatesignal_graham_ode` is deprecated, use `simulate_graham_ode` instead (note the added `M0` argument).", :calculatesignal_graham_ode)
    simulate_graham_ode(α, TRF, TR, ω0, B1, 1, m0s, R1f, R2f, Rex, R1s, T2s; kwargs...)
end
export calculatesignal_graham_ode

# Renamed in v0.11: CRB_gradient_OCT → crb_gradient
function CRB_gradient_OCT(args...; kwargs...)
    Base.depwarn("`CRB_gradient_OCT` is deprecated, use `crb_gradient` instead.", :CRB_gradient_OCT)
    crb_gradient(args...; kwargs...)
end

# Renamed in v0.11: bound_ω1_TRF! → bound_omega1_TRF!
function bound_ω1_TRF!(args...; kwargs...)
    Base.depwarn("`bound_ω1_TRF!` is deprecated, use `bound_omega1_TRF!` instead.", :bound_ω1_TRF!)
    bound_omega1_TRF!(args...; kwargs...)
end

# Renamed in v0.11: get_bounded_ω1_TRF → get_bounded_omega1_TRF
function get_bounded_ω1_TRF(args...; kwargs...)
    Base.depwarn("`get_bounded_ω1_TRF` is deprecated, use `get_bounded_omega1_TRF` instead.", :get_bounded_ω1_TRF)
    get_bounded_omega1_TRF(args...; kwargs...)
end

# Renamed in v0.11: second_order_α! → penalty_alpha_curvature!
function second_order_α!(args...; kwargs...)
    Base.depwarn("`second_order_α!` is deprecated, use `penalty_alpha_curvature!` instead.", :second_order_α!)
    penalty_alpha_curvature!(args...; kwargs...)
end

# Renamed in v0.11: RF_power! → penalty_RF_power!
function RF_power!(args...; kwargs...)
    Base.depwarn("`RF_power!` is deprecated, use `penalty_RF_power!` instead.", :RF_power!)
    penalty_RF_power!(args...; kwargs...)
end

# Renamed in v0.11: TRF_TV! → penalty_TRF_variation!
function TRF_TV!(args...; kwargs...)
    Base.depwarn("`TRF_TV!` is deprecated, use `penalty_TRF_variation!` instead.", :TRF_TV!)
    penalty_TRF_variation!(args...; kwargs...)
end
