# ============================================================================
# Optimal Control for Cramér-Rao Bound Minimization
#
# State vector convention (11 components per gradient parameter):
#   [xf, yf, zf, xs, zs, ∂xf/∂θ, ∂yf/∂θ, ∂zf/∂θ, ∂xs/∂θ, ∂zs/∂θ, 1]
#    1    2    3   4   5     6       7        8       9       10      11
#
# The measured signal is s = xf + i·yf (components 1 and 2).
# The signal derivative wrt. parameter θ is ∂s/∂θ = m[6] + i·m[7].
#
# The Fisher information matrix is the real-valued matrix:
#   F[g1,g2] = Σ_t (∂xf/∂θ_g1 · ∂xf/∂θ_g2 + ∂yf/∂θ_g1 · ∂yf/∂θ_g2)
#            = Re{ Σ_t conj(∂s/∂θ_g1) · ∂s/∂θ_g2 }
#
# Adjoint-state optimal control pipeline (per sequence):
#   1. build_propagators            – E[t,g] and derivatives dE/dω₁, dE/dTRF
#   2. steady_state_operator        – cycle operator Q[g] for periodic boundaries
#   3. steady_state_magnetization   – solve for and propagate the periodic state
#   4. assemble_fisher_matrix       – F from steady-state signal derivatives
#   5. crb_and_derivatives          – CRB = wᵀ diag(F⁻¹) and ∂CRB/∂m sources
#   6. adjoint_backpropagate        – co-state P driven by ∂CRB/∂m sources
#   7. control_gradients            – ∂CRB/∂ω₁ and ∂CRB/∂TRF via ⟨P, dE·m⟩
# ============================================================================

"""
    CRB, grad_ω1, grad_TRF = crb_gradient(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list, weights; grad_moment=...)

Calculate the Cramér-Rao bound (CRB) and its gradients wrt. the RF controls
`ω1` (amplitude) and `TRF` (duration) using the adjoint state method.

The CRB is computed as `wᵀ diag(F⁻¹)`, where `F` is the Fisher information
matrix assembled from the steady-state signal derivatives, and `w` are
user-supplied weights for each fitted parameter.

# Arguments
- `ω1`: RF amplitude per pulse in rad/s. Vector (single sequence) or matrix (columns = sequences).
- `TRF`: RF pulse duration per pulse in seconds. Same shape as `ω1`.
- `TR::Real`: Repetition time in seconds
- `ω0::Real`: Off-resonance frequency in rad/s
- `B1::Real`: Normalized transmit B1 field, i.e. B1 = 1 corresponds to a well-calibrated B1 field
- `m0s::Real`: Fractional size of the semi-solid pool; should be in range of 0 to 1
- `R1f::Real`: Longitudinal relaxation rate of the free pool in 1/seconds
- `R2f::Real`: Transversal relaxation rate of the free pool in 1/seconds
- `Rex::Real`: Exchange rate between the two spin pools in 1/seconds
- `R1s::Real`: Longitudinal relaxation rate of the semi-solid pool in 1/seconds
- `T2s::Real`: Transversal relaxation time of the semi-solid pool in seconds
- `R2slT::NTuple{3, Function}`: Tuple of three functions: R2sl(TRF, ω1, B1, T2s), dR2sldB1(TRF, ω1, B1, T2s), and R2sldT2s(TRF, ω1, B1, T2s). Can be generated with [`precompute_R2sl`](@ref)
- `grad_list::Tuple{<:grad_param}`: Tuple that specifies the gradients that are calculated; any subset/order of `(grad_M0(), grad_m0s(), grad_R1f(), grad_R2f(), grad_Rex(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1())`; the derivative wrt. to apparent `R1a = R1f = R1s` can be calculated with `grad_R1a()`. Including `grad_M0()` is recommended to account for the equilibrium magnetization in the CRB.
- `weights::transpose(Vector{Real})`: Row vector of weights applied to the Cramér-Rao bounds of the individual parameters, matching `grad_list` in order and length.

# Optional Keyword Arguments:
- `grad_moment`: Gradient spoiling scheme per pulse (`:balanced`, `:crusher`, `:spoiler_dual`, `:spoiler_prepulse`). `:balanced` simulates a TR with all gradient moments nulled. `:crusher` assumes equivalent (non-zero) gradient moments before and simulates the refocussing path of the extended phase graph. `:spoiler_prepulse` nulls all transverse magnetization before the RF pulse, emulating an idealized FLASH. `:spoiler_dual` nulls all transverse magnetization before and after the RF pulse.

# Examples
```jldoctest
julia> CRB, grad_ω1, grad_TRF = crb_gradient(range(pi/2, π, 100), range(100e-6, 400e-6, 100), 3.5e-3, 0, 1, 0.15, 0.5, 15, 30, 4, 10e-6, precompute_R2sl(), [grad_M0(), grad_m0s(), grad_R2f()], transpose([0, 1, 1]));

```
See also: [Optimal Control](@ref)
"""
function crb_gradient(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list, weights; grad_moment=[i[1] == 1 ? :spoiler_dual : :balanced for i ∈ CartesianIndices(ω1)])
    nSeq = size(ω1, 2)
    @assert length(weights) == length(grad_list) "weights must have the same length as grad_list"

    propagators   = Vector{Matrix{SMatrix{11,11,Float64}}}(undef, nSeq)
    dprop_dω1     = similar(propagators)
    dprop_dTRF    = similar(propagators)
    cycle_ops     = Vector{Vector{SMatrix{11,11,Float64}}}(undef, nSeq)
    magnetization = Vector{Matrix{SVector{11,Float64}}}(undef, nSeq)

    grad_ω1  = similar(ω1)
    grad_TRF = similar(ω1)

    # Forward pass: build propagators and compute steady-state magnetization
    Threads.@threads for iSeq ∈ eachindex(propagators)
        propagators[iSeq], dprop_dω1[iSeq], dprop_dTRF[iSeq] = @views build_propagators(
            ω1[:, iSeq], TRF[:, iSeq], TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list;
            grad_moment=grad_moment[:, iSeq])
        cycle_ops[iSeq]     = steady_state_operator(propagators[iSeq])
        magnetization[iSeq] = steady_state_magnetization(cycle_ops[iSeq], propagators[iSeq])
    end

    # CRB and its derivative wrt. magnetization components
    CRB, crb_source = crb_and_derivatives(magnetization, weights)

    # Backward pass: adjoint backpropagation → control gradients
    for iSeq ∈ eachindex(propagators)
        source_iSeq = (t, g) -> crb_source(iSeq, t, g)
        costate = adjoint_backpropagate(source_iSeq, cycle_ops[iSeq], propagators[iSeq])
        grad_ω1[:, iSeq], grad_TRF[:, iSeq] = control_gradients(
            costate, magnetization[iSeq], propagators[iSeq], dprop_dω1[iSeq], dprop_dTRF[iSeq])
    end

    return CRB, grad_ω1, grad_TRF
end


# ============================================================================
# Step 1: Build propagators and their derivatives wrt. ω₁ and TRF
# ============================================================================

"""
    E, dEdω1, dEdTRF = build_propagators(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list; grad_moment)

Compute the per-pulse propagator `E[t, g]` and its derivatives `dE/dω₁[t, g]`
and `dE/dTRF[t, g]` for each time step `t` and gradient parameter index `g`.

Each propagator maps the 11-component state vector from one TR to the next,
incorporating free precession, RF pulse, phase cycling, and gradient spoiling.

Derivatives are computed via augmented 22×22 matrix exponentials:
`exp([H 0; dH/dθ H] · τ)` yields both `exp(H·τ)` and `d(exp(H·τ))/dθ`
in a single matrix exponential.
"""
function build_propagators(ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad_list; grad_moment)
    E      = Array{SMatrix{11,11,Float64,121}}(undef, length(ω1), length(grad_list))
    dEdω1  = Array{SMatrix{11,11,Float64,121}}(undef, length(ω1), length(grad_list))
    dEdTRF = Array{SMatrix{11,11,Float64,121}}(undef, length(ω1), length(grad_list))

    cache = ExponentialUtilities.alloc_mem(zeros(22, 22), ExpMethodHigham2005Base())
    augmented_H = zeros(Float64, 22, 22)

    prop_phasecycle = z_rotation_propagator(π, grad_m0s())
    for g ∈ eachindex(grad_list)
        grad = grad_list[g]

        for t ∈ 1:length(ω1)
            if grad_moment[t] == :crusher
                build_crushed_propagator!(E, dEdω1, dEdTRF, t, g, ω1[t], TRF[t], TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad, prop_phasecycle, augmented_H, cache)
            else
                build_pulse_propagator!(E, dEdω1, dEdTRF, t, g, ω1[t], TRF[t], TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad, prop_phasecycle, augmented_H, cache, grad_moment[t])
            end
        end
    end
    return E, dEdω1, dEdTRF
end

"""
    build_pulse_propagator!(E, dEdω1, dEdTRF, t, g, ...)

Compute the propagator and its ω₁/TRF derivatives for a single non-crushed
RF pulse at time index `t` and gradient parameter index `g`, writing results
into `E[t,g]`, `dEdω1[t,g]`, and `dEdTRF[t,g]`.

The full TR propagator is: `prop_freeprec · prop_pulse · prop_phasecycle · prop_freeprec`,
where `prop_freeprec` includes gradient spoiling and half-TR free precession.
"""
function build_pulse_propagator!(E, dEdω1, dEdTRF, t, g, ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad, prop_phasecycle, augmented_H, cache, grad_moment)
    # Free-precession Hamiltonian (no RF) and half-TR propagator with spoiling
    H_freeprec = hamiltonian_linear(0.0, B1, ω0, 1.0, 1.0, m0s, R1f, R2f, Rex, R1s, 0.0, 0.0, 0.0, grad)
    spoiler = grad_moment == :spoiler_dual ? xy_destructor(H_freeprec) : xs_destructor(H_freeprec)
    prop_freeprec = spoiler * exp(H_freeprec * ((TR - TRF) / 2))

    # Pulse Hamiltonian (during RF)
    H_pulse = hamiltonian_linear(ω1, B1, ω0, 1, 1.0, m0s, R1f, R2f, Rex, R1s,
        R2slT.R2sl(TRF, ω1 * TRF, B1, T2s),
        R2slT.dR2sl_dT2s(TRF, ω1 * TRF, B1, T2s),
        R2slT.dR2sl_dB1(TRF, ω1 * TRF, B1, T2s),
        grad)

    # --- Derivative wrt. ω₁ via augmented matrix exponential ---
    dHdω1 = d_hamiltonian_linear_dω1(B1, 1,
        R2slT.dR2sl_dω1(TRF, ω1 * TRF, B1, T2s),
        R2slT.d2R2sl_dT2s_dω1(TRF, ω1 * TRF, B1, T2s),
        R2slT.d2R2sl_dB1_dω1(TRF, ω1 * TRF, B1, T2s),
        grad)

    # Augmented system: [H_pulse, 0; dH/dω₁, H_pulse] * TRF
    @views augmented_H[1:11, 1:11]   .= H_pulse
    @views augmented_H[12:22, 12:22] .= H_pulse
    @views augmented_H[1:11, 12:22]  .= 0
    @views augmented_H[12:22, 1:11]  .= dHdω1
    augmented_H .*= TRF
    exp_aug = exponential!(augmented_H, ExpMethodHigham2005Base(), cache)

    prop_pulse      = SMatrix{11,11}(@view exp_aug[1:11, 1:11])
    dprop_pulse_dω1 = SMatrix{11,11}(@view exp_aug[12:end, 1:11])
    E[t, g]     = prop_freeprec * prop_pulse * prop_phasecycle * prop_freeprec
    dEdω1[t, g] = prop_freeprec * dprop_pulse_dω1 * prop_phasecycle * prop_freeprec

    # --- Derivative wrt. TRF via augmented matrix exponential ---
    dHdTRF = H_pulse + d_hamiltonian_linear_dTRF_add(TRF,
        R2slT.dR2sl_dTRF(TRF, ω1 * TRF, B1, T2s),
        R2slT.d2R2sl_dT2s_dTRF(TRF, ω1 * TRF, B1, T2s),
        R2slT.d2R2sl_dB1_dTRF(TRF, ω1 * TRF, B1, T2s),
        grad)
    H_pulse *= TRF
    @views augmented_H[1:11, 1:11]   .= H_pulse
    @views augmented_H[12:22, 12:22] .= H_pulse
    @views augmented_H[1:11, 12:22]  .= 0
    @views augmented_H[12:22, 1:11]  .= dHdTRF
    exp_aug = exponential!(augmented_H, ExpMethodHigham2005Base(), cache)

    prop_pulse        = SMatrix{11,11}(@view exp_aug[1:11, 1:11])
    dprop_pulse_dTRF  = SMatrix{11,11}(@view exp_aug[12:end, 1:11])
    # Chain rule: TRF also affects the free-precession duration (TR - TRF)/2
    dEdTRF[t, g] = prop_freeprec * ((dprop_pulse_dTRF - (1 / 2 * H_freeprec * prop_pulse)) * prop_phasecycle - (1 / 2 * prop_pulse * prop_phasecycle * H_freeprec)) * prop_freeprec
    return nothing
end

"""
    build_crushed_propagator!(E, dEdω1, dEdTRF, t, g, ...)

Compute the propagator for a crushed (inversion) pulse at time index `t` and
gradient parameter index `g`. Crushed pulses have zero derivatives wrt. ω₁
and TRF (they are not optimized).
"""
function build_crushed_propagator!(E, dEdω1, dEdTRF, t, g, ω1, TRF, TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT, grad, prop_phasecycle, _, _)
    prop_freeprec = exp(hamiltonian_linear(0, B1, ω0, TR / 2, 1.0, m0s, R1f, R2f, Rex, R1s, 0, 0, 0, grad))
    prop_freeprec = xs_destructor(prop_freeprec) * prop_freeprec
    prop_pulse = propagator_linear_crushed_pulse(ω1, TRF, B1,
        R2slT.R2sl(TRF, ω1 * TRF, B1, T2s),
        R2slT.dR2sl_dT2s(TRF, ω1 * TRF, B1, T2s),
        R2slT.dR2sl_dB1(TRF, ω1 * TRF, B1, T2s),
        grad)
    E[t, g] = prop_freeprec * prop_pulse * prop_phasecycle * prop_freeprec
    dEdω1[t, g]  = @SMatrix zeros(11, 11)
    dEdTRF[t, g] = @SMatrix zeros(11, 11)
    return nothing
end


# ============================================================================
# Steps 2-3: Steady-state boundary conditions and magnetization propagation
# ============================================================================

"""
    cycle_ops = steady_state_operator(propagators)

Compute the cycle operator `Q[g] = A₀(A) - A` for each gradient parameter,
where `A = E[1] · E[end] · ... · E[2]` is the full-cycle propagator. The
steady-state magnetization satisfies `Q · m = c`, where `c` encodes the
thermal equilibrium source and `A₀` zeroes the affine (thermal equilibrium) row.
"""
function steady_state_operator(propagators)
    cycle_ops = Vector{SMatrix{11,11,Float64}}(undef, size(propagators, 2))

    for g ∈ axes(propagators, 2)
        A = propagators[1, g]
        for t = size(propagators, 1):-1:2
            A = A * propagators[t, g]
        end
        cycle_ops[g] = A0(A) - A
    end
    return cycle_ops
end

"""
    magnetization = steady_state_magnetization(cycle_ops, propagators)

Solve for the periodic steady-state magnetization and propagate it through
all time steps. Returns `magnetization[t, g]` as an 11-component `SVector`
for each time step `t` and gradient parameter `g`.
"""
function steady_state_magnetization(cycle_ops, propagators)
    magnetization = similar(propagators, SVector{11,Float64})

    for g ∈ axes(propagators, 2)
        m = cycle_ops[g] \ C(cycle_ops[g])
        magnetization[1, g] = m
        for t = 2:size(propagators, 1)
            m = propagators[t, g] * m
            magnetization[t, g] = m
        end
    end
    return magnetization
end


# ============================================================================
# Steps 4-5: Fisher information matrix, CRB, and ∂CRB/∂m
# ============================================================================

"""
    CRBSourceTerms(dCRB_dxf, dCRB_dyf)

Callable struct storing the source terms `∂CRB/∂m` for adjoint backpropagation.
Call as `source(iSeq, t, g)` to get an 11-component `SVector` with non-zero
entries at positions 6 (∂CRB/∂(∂xf/∂θ)) and 7 (∂CRB/∂(∂yf/∂θ)).
"""
struct CRBSourceTerms
    dCRB_dxf::Vector{Array{Float64}}
    dCRB_dyf::Vector{Array{Float64}}
end

function (s::CRBSourceTerms)(iSeq, t, g)
    @SVector [0, 0, 0, 0, 0, s.dCRB_dxf[iSeq][t, g], s.dCRB_dyf[iSeq][t, g], 0, 0, 0, 0]
end

"""
    fisher = assemble_fisher_matrix(magnetization)

Assemble the Fisher information matrix from the steady-state magnetization.

The signal derivative wrt. parameter `g` at time `t` has real part `m[t,g][6]`
(= ∂xf/∂θ_g) and imaginary part `m[t,g][7]` (= ∂yf/∂θ_g). The FIM is the
real symmetric matrix:

    F[g1, g2] = Σ_{t,seq} (∂xf/∂θ_g1 · ∂xf/∂θ_g2 + ∂yf/∂θ_g1 · ∂yf/∂θ_g2)
"""
function assemble_fisher_matrix(magnetization)
    N_grad = size(magnetization[1], 2)
    fisher = zeros(Float64, N_grad, N_grad)
    for iSeq ∈ eachindex(magnetization), g2 ∈ 1:N_grad, g1 ∈ 1:N_grad, t ∈ axes(magnetization[iSeq], 1)
        fisher[g1, g2] += magnetization[iSeq][t, g1][6] * magnetization[iSeq][t, g2][6] +
                          magnetization[iSeq][t, g1][7] * magnetization[iSeq][t, g2][7]
    end
    return fisher
end

"""
    CRB, crb_source = crb_and_derivatives(magnetization, weights)

Compute the weighted Cramér-Rao bound `CRB = wᵀ diag(F⁻¹)` and the source
terms `∂CRB/∂m` needed to drive the adjoint backpropagation.

Returns `crb_source` as a [`CRBSourceTerms`](@ref) callable struct, where
`crb_source(iSeq, t, g)` returns an 11-component `SVector` with non-zero
entries only at positions 6 (∂CRB/∂(∂xf/∂θ)) and 7 (∂CRB/∂(∂yf/∂θ)).
These source terms use the chain rule through `CRB = wᵀ diag(F⁻¹)` and
`∂F⁻¹/∂x = -F⁻¹ (∂F/∂x) F⁻¹`.
"""
function crb_and_derivatives(magnetization, weights)
    N_grad = size(magnetization[1], 2)

    fisher = assemble_fisher_matrix(magnetization)
    fisher_inv = inv(fisher)
    CRB = dot(weights, diag(fisher_inv))

    # Compute ∂CRB/∂(∂xf/∂θ) and ∂CRB/∂(∂yf/∂θ) for each (t, g, seq)
    dCRB_dxf = [Array{Float64}(undef, size(Yi, 1), size(Yi, 2)) for Yi ∈ magnetization]
    dCRB_dyf = [Array{Float64}(undef, size(Yi, 1), size(Yi, 2)) for Yi ∈ magnetization]
    dFdm = similar(fisher)
    tmp  = similar(fisher)

    for iSeq ∈ eachindex(magnetization), g1 ∈ 1:N_grad, t ∈ axes(magnetization[iSeq], 1)
        # ∂CRB/∂(∂xf/∂θ_g1) at time t: uses ∂F/∂(∂xf/∂θ_g1)
        dFdm .= 0
        for g2 ∈ 1:N_grad
            dFdm[g1, g2] = magnetization[iSeq][t, g2][6]
            dFdm[g2, g1] = magnetization[iSeq][t, g2][6]
        end
        dFdm[g1, g1] = 2 * magnetization[iSeq][t, g1][6]
        mul!(dFdm, fisher_inv, mul!(tmp, dFdm, fisher_inv))
        dCRB_dxf[iSeq][t, g1] = dot(weights, diag(dFdm))

        # ∂CRB/∂(∂yf/∂θ_g1) at time t: uses ∂F/∂(∂yf/∂θ_g1)
        dFdm .= 0
        for g2 ∈ 1:N_grad
            dFdm[g1, g2] = magnetization[iSeq][t, g2][7]
            dFdm[g2, g1] = magnetization[iSeq][t, g2][7]
        end
        dFdm[g1, g1] = 2 * magnetization[iSeq][t, g1][7]
        mul!(dFdm, fisher_inv, mul!(tmp, dFdm, fisher_inv))
        dCRB_dyf[iSeq][t, g1] = dot(weights, diag(dFdm))
    end

    return CRB, CRBSourceTerms(dCRB_dxf, dCRB_dyf)
end


# ============================================================================
# Step 6: Adjoint backpropagation
# ============================================================================

"""
    costate = adjoint_backpropagate(crb_source, cycle_ops, propagators)

Compute the adjoint (co-state) variables by backpropagating the CRB source
terms through the pulse sequence.

The adjoint equation is the time-reversed analog of the forward propagation,
driven by the source terms `∂CRB/∂m(t,g)`. The computation has two phases:

**Phase 1 — Accumulate boundary condition:** Sum all source contributions
backward through the sequence to find `λ = Σ_t E[T→t]ᵀ · source(t)`,
then solve the periodic boundary condition `P[end] = Q⁻ᵀ · λ`.

**Phase 2 — Backward propagation:** Starting from `P[end]`, propagate the
co-state backward: `P[t-1] = E[t+1]ᵀ · P[t] + source(t)`, accumulating
source terms at each step.
"""
function adjoint_backpropagate(crb_source, cycle_ops, propagators)
    costate = similar(propagators, Array{Float64})

    # Phase 1: accumulate source terms backward to find boundary λ
    λ = @view costate[end, :]
    for g ∈ 1:size(propagators, 2)
        λ[g] = crb_source(size(propagators, 1), g)
    end
    for t = size(propagators, 1)-1:-1:1
        for g ∈ axes(propagators, 2)
            λ[g] = transpose(propagators[t+1, g]) * λ[g]
            λ[g] += crb_source(t, g)
        end
    end

    # Solve periodic boundary condition: P[end] = Q⁻ᵀ · λ
    for g ∈ axes(propagators, 2)
        costate[end, g] = inv(transpose(cycle_ops[g])) * λ[g]
    end

    # Phase 2: backward propagation with source injection
    for t ∈ size(propagators, 1):-1:2
        for g ∈ axes(propagators, 2)
            next_prop = propagators[mod(t, size(propagators, 1))+1, g]
            costate[t-1, g] = transpose(next_prop) * costate[t, g]
            costate[t-1, g] .+= crb_source(t, g)
        end
    end
    return costate
end


# ============================================================================
# Step 7: Control gradients via inner product ⟨costate, dE · magnetization⟩
# ============================================================================

"""
    grad_ω1, grad_TRF = control_gradients(costate, magnetization, propagators, dprop_dω1, dprop_dTRF)

Compute the gradients of the CRB wrt. the RF controls `ω1` and `TRF` by
evaluating the inner product `⟨P[t-1], dE[t]/dθ · m[t-1]⟩` at each time step,
where `P` is the co-state (adjoint variable) and `m` is the magnetization.
"""
function control_gradients(costate, magnetization, propagators, dprop_dω1, dprop_dTRF)
    grad_ω1  = zeros(size(propagators, 1))
    grad_TRF = zeros(size(propagators, 1))

    for g ∈ axes(magnetization, 2), t ∈ axes(magnetization, 1)
        t_prev = mod1(t - 1, size(magnetization, 1))
        grad_ω1[t]  -= transpose(costate[t_prev, g]) * (dprop_dω1[t, g]  * magnetization[t_prev, g])
        grad_TRF[t] -= transpose(costate[t_prev, g]) * (dprop_dTRF[t, g] * magnetization[t_prev, g])
    end
    return grad_ω1, grad_TRF
end

##############################################################################
# 3-pool Optimal Control
##############################################################################
"""
    CRB, grad_ω1, grad_TRF = crb_gradient(ω1, TRF, TR, ω0, B1, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, T2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw, R2slT, grad_list, weights; ...)

3-pool version of crb_gradient for the free water / restricted water / macromolecular model.
"""
function crb_gradient(ω1, TRF, TR, ω0, B1, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, T2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw, R2slT, grad_list, weights; grad_moment=[i[1] == 1 ? :spoiler_dual : :balanced for i ∈ CartesianIndices(ω1)])
    nSeq = size(ω1, 2)

    E = Vector{Matrix{Matrix{Float64}}}(undef, nSeq)
    dEdω1 = similar(E)
    dEdTRF = similar(E)
    Q = Vector{Vector{Matrix{Float64}}}(undef, nSeq)
    Y = Vector{Matrix{Vector{Float64}}}(undef, nSeq)

    grad_ω1 = similar(ω1)
    grad_TRF = similar(ω1)

    Threads.@threads for iSeq ∈ eachindex(E)
        E[iSeq], dEdω1[iSeq], dEdTRF[iSeq] = @views build_propagators_3pool(
            ω1[:, iSeq], TRF[:, iSeq], TR, ω0, B1, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, T2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw, R2slT, grad_list; grad_moment=grad_moment[:, iSeq])
        Q[iSeq] = steady_state_operator_3pool(E[iSeq])
        Y[iSeq] = steady_state_magnetization_3pool(Q[iSeq], E[iSeq])
    end

    CRB, d = crb_and_derivatives_3pool(Y, weights)

    for iSeq ∈ eachindex(E)
        P = adjoint_backpropagate_3pool(d[iSeq], Q[iSeq], E[iSeq])
        grad_ω1[:, iSeq], grad_TRF[:, iSeq] = control_gradients_3pool(P, Y[iSeq], E[iSeq], dEdω1[iSeq], dEdTRF[iSeq])
    end

    return CRB, grad_ω1, grad_TRF
end

function build_propagators_3pool(ω1, TRF, TR, ω0, B1, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, T2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw, R2slT, grad_list; grad_moment)
    E      = Array{Matrix{Float64}}(undef, length(ω1), length(grad_list))
    dEdω1  = Array{Matrix{Float64}}(undef, length(ω1), length(grad_list))
    dEdTRF = Array{Matrix{Float64}}(undef, length(ω1), length(grad_list))

    cache = ExponentialUtilities.alloc_mem(zeros(34, 34), ExpMethodHigham2005Base())
    dH = zeros(Float64, 34, 34)

    u_rot = z_rotation_propagator_3pool(π, grad_m0_rw())
    for g ∈ eachindex(grad_list)
        grad = grad_list[g]

        for t ∈ 1:length(ω1)
            if grad_moment[t] == :crusher
                build_crushed_propagator_3pool!(E, dEdω1, dEdTRF, t, g, ω1[t], TRF[t], TR, ω0, B1, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, T2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw, R2slT, grad, u_rot, dH, cache)
            else
                build_pulse_propagator_3pool!(E, dEdω1, dEdTRF, t, g, ω1[t], TRF[t], TR, ω0, B1, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, T2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw, R2slT, grad, u_rot, dH, cache, grad_moment[t])
            end
        end
    end
    return E, dEdω1, dEdTRF
end

function build_pulse_propagator_3pool!(E, dEdω1, dEdTRF, t, g, ω1, TRF, TR, ω0, B1, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, T2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw, R2slT, grad, u_rot, dH, cache, grad_moment)
    M0 = 1.0  # M0 handled externally for optimal control
    H_fp = hamiltonian_linear(0.0, B1, ω0, 1.0, M0, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, 0.0, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw, 0.0, 0.0, grad)
    ux = grad_moment == :spoiler_dual ? xy_destructor(H_fp) : x_mm_destructor(H_fp)
    u_fp = ux * exp(H_fp * ((TR - TRF) / 2))

    H_pl = hamiltonian_linear(ω1, B1, ω0, 1, M0, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, R2slT[1](TRF, ω1 * TRF, B1, T2_mm), Rx_fw_mm, Rx_rw_fw, Rx_mm_rw,
        R2slT[2](TRF, ω1 * TRF, B1, T2_mm),
        R2slT[3](TRF, ω1 * TRF, B1, T2_mm),
        grad)

    dHdω1 = d_hamiltonian_linear_dω1_3pool(B1, 1,
        R2slT[4](TRF, ω1 * TRF, B1, T2_mm),
        R2slT[6](TRF, ω1 * TRF, B1, T2_mm),
        R2slT[7](TRF, ω1 * TRF, B1, T2_mm),
        grad)

    @views dH[1:17, 1:17] .= H_pl
    @views dH[18:34, 18:34] .= H_pl
    @views dH[1:17, 18:34] .= 0
    @views dH[18:34, 1:17] .= dHdω1
    dH .*= TRF
    E_pl = exponential!(dH, ExpMethodHigham2005Base(), cache)

    E_pl1 = @view E_pl[1:17, 1:17]
    E_pl2 = @view E_pl[18:end, 1:17]
    E[t, g] = u_fp * E_pl1 * u_rot * u_fp
    dEdω1[t, g] = u_fp * E_pl2 * u_rot * u_fp

    # TRF derivative
    dHdTRF = H_pl + d_hamiltonian_linear_dTRF_add_3pool(TRF,
        R2slT[5](TRF, ω1 * TRF, B1, T2_mm),
        R2slT[8](TRF, ω1 * TRF, B1, T2_mm),
        R2slT[9](TRF, ω1 * TRF, B1, T2_mm),
        grad)
    H_pl *= TRF
    @views dH[1:17, 1:17] .= H_pl
    @views dH[18:34, 18:34] .= H_pl
    @views dH[1:17, 18:34] .= 0
    @views dH[18:34, 1:17] .= dHdTRF
    E_pl = exponential!(dH, ExpMethodHigham2005Base(), cache)

    E_pl1 = @view E_pl[1:17, 1:17]
    E_pl2 = @view E_pl[18:end, 1:17]
    dEdTRF[t, g] = u_fp * ((E_pl2 - (1 / 2 * H_fp * E_pl1)) * u_rot - (1 / 2 * E_pl1 * u_rot * H_fp)) * u_fp
    return nothing
end

function build_crushed_propagator_3pool!(E, dEdω1, dEdTRF, t, g, ω1, TRF, TR, ω0, B1, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, T2_mm, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw, R2slT, grad, u_rot, _, _)
    M0 = 1.0
    u_fp = exp(hamiltonian_linear(0, B1, ω0, TR / 2, M0, m0_rw, m0_mm, R1_fw, R1_rw, R1_mm, R2_fw, R2_rw, 0.0, Rx_fw_mm, Rx_rw_fw, Rx_mm_rw, 0.0, 0.0, grad))
    # TODO: implement crushed pulse propagator for 3-pool
    @info "3-pool crushed pulse propagator not yet implemented"
    E[t, g] = zeros(17, 17)
    dEdω1[t, g] = zeros(17, 17)
    dEdTRF[t, g] = zeros(17, 17)
    return nothing
end

function steady_state_operator_3pool(E)
    Q = Vector{Matrix{Float64}}(undef, size(E, 2))
    for g ∈ axes(E, 2)
        A = E[1, g]
        for t = size(E, 1):-1:2
            A = A * E[t, g]
        end
        Q[g] = A0(A) - A
    end
    return Q
end

function steady_state_magnetization_3pool(Q, E)
    Y = similar(E, Vector{Float64})
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

function crb_and_derivatives_3pool(Y, w)
    N_grad = size(Y[1], 2)

    F = zeros(ComplexF64, N_grad + 1, N_grad + 1)
    for iSeq ∈ eachindex(Y), g2 ∈ 0:N_grad, g1 ∈ 0:N_grad, t ∈ axes(Y[iSeq], 1)
        s1 = g1 == 0 ? Y[iSeq][t, 1][1] - 1im * Y[iSeq][t, 1][2] + Y[iSeq][t, 1][4] - 1im * Y[iSeq][t, 1][5] : Y[iSeq][t, g1][9] - 1im * Y[iSeq][t, g1][10] + Y[iSeq][t, g1][12] - 1im * Y[iSeq][t, g1][13]
        s2 = g2 == 0 ? Y[iSeq][t, 1][1] + 1im * Y[iSeq][t, 1][2] + Y[iSeq][t, 1][4] + 1im * Y[iSeq][t, 1][5] : Y[iSeq][t, g2][9] + 1im * Y[iSeq][t, g2][10] + Y[iSeq][t, g2][12] + 1im * Y[iSeq][t, g2][13]
        F[g1+1, g2+1] += s1 * s2
    end
    Fi = inv(F)
    CRB = w * real.(diag(Fi))

    _dCRBdx_fw = [Array{Float64}(undef, size(Yi, 1), size(Yi, 2) + 1) for Yi ∈ Y]
    _dCRBdy_fw = [Array{Float64}(undef, size(Yi, 1), size(Yi, 2) + 1) for Yi ∈ Y]
    _dCRBdx_rw = [Array{Float64}(undef, size(Yi, 1), size(Yi, 2) + 1) for Yi ∈ Y]
    _dCRBdy_rw = [Array{Float64}(undef, size(Yi, 1), size(Yi, 2) + 1) for Yi ∈ Y]
    dFdy = similar(F)
    tmp = similar(F)
    for iSeq ∈ eachindex(Y), g1 ∈ 0:N_grad, t ∈ axes(Y[iSeq], 1)
        # derivative wrt. x fw
        dFdy .= 0
        dFdy[g1+1, 1] = Y[iSeq][t, 1][1] + 1im * Y[iSeq][t, 1][2] + Y[iSeq][t, 1][4] + 1im * Y[iSeq][t, 1][5]
        dFdy[1, g1+1] = Y[iSeq][t, 1][1] - 1im * Y[iSeq][t, 1][2] + Y[iSeq][t, 1][4] - 1im * Y[iSeq][t, 1][5]
        for g2 ∈ axes(Y[iSeq], 2)
            dFdy[g1+1, g2+1] = Y[iSeq][t, g2][9] + 1im * Y[iSeq][t, g2][10] + Y[iSeq][t, g2][12] + 1im * Y[iSeq][t, g2][13]
            dFdy[g2+1, g1+1] = Y[iSeq][t, g2][9] - 1im * Y[iSeq][t, g2][10] + Y[iSeq][t, g2][12] - 1im * Y[iSeq][t, g2][13]
        end
        dFdy[g1+1, g1+1] = 2 * real(dFdy[g1+1, g1+1])
        mul!(dFdy, Fi, mul!(tmp, dFdy, Fi))
        _dCRBdx_fw[iSeq][t, g1+1] = real.(w * diag(dFdy))

        # derivative wrt. x rw
        dFdy .= 0
        dFdy[g1+1, 1] = Y[iSeq][t, 1][4] + 1im * Y[iSeq][t, 1][5] + Y[iSeq][t, 1][1] + 1im * Y[iSeq][t, 1][2]
        dFdy[1, g1+1] = Y[iSeq][t, 1][4] - 1im * Y[iSeq][t, 1][5] + Y[iSeq][t, 1][1] - 1im * Y[iSeq][t, 1][2]
        for g2 ∈ axes(Y[iSeq], 2)
            dFdy[g1+1, g2+1] = Y[iSeq][t, g2][12] + 1im * Y[iSeq][t, g2][13] + Y[iSeq][t, g2][9] + 1im * Y[iSeq][t, g2][10]
            dFdy[g2+1, g1+1] = Y[iSeq][t, g2][12] - 1im * Y[iSeq][t, g2][13] + Y[iSeq][t, g2][9] - 1im * Y[iSeq][t, g2][10]
        end
        dFdy[g1+1, g1+1] = 2 * real(dFdy[g1+1, g1+1])
        mul!(dFdy, Fi, mul!(tmp, dFdy, Fi))
        _dCRBdx_rw[iSeq][t, g1+1] = real.(w * diag(dFdy))

        # derivative wrt. y fw
        dFdy .= 0
        dFdy[g1+1, 1] = Y[iSeq][t, 1][2] - 1im * Y[iSeq][t, 1][1] + Y[iSeq][t, 1][5] - 1im * Y[iSeq][t, 1][4]
        dFdy[1, g1+1] = Y[iSeq][t, 1][2] + 1im * Y[iSeq][t, 1][1] + Y[iSeq][t, 1][5] + 1im * Y[iSeq][t, 1][4]
        for g2 ∈ axes(Y[iSeq], 2)
            dFdy[g1+1, g2+1] = Y[iSeq][t, g2][10] - 1im * Y[iSeq][t, g2][9] + Y[iSeq][t, g2][13] - 1im * Y[iSeq][t, g2][12]
            dFdy[g2+1, g1+1] = Y[iSeq][t, g2][10] + 1im * Y[iSeq][t, g2][9] + Y[iSeq][t, g2][13] + 1im * Y[iSeq][t, g2][12]
        end
        dFdy[g1+1, g1+1] = 2 * real(dFdy[g1+1, g1+1])
        mul!(dFdy, Fi, mul!(tmp, dFdy, Fi))
        _dCRBdy_fw[iSeq][t, g1+1] = real.(w * diag(dFdy))

        # derivative wrt. y rw
        dFdy .= 0
        dFdy[g1+1, 1] = Y[iSeq][t, 1][5] - 1im * Y[iSeq][t, 1][4] + Y[iSeq][t, 1][2] - 1im * Y[iSeq][t, 1][1]
        dFdy[1, g1+1] = Y[iSeq][t, 1][5] + 1im * Y[iSeq][t, 1][4] + Y[iSeq][t, 1][2] + 1im * Y[iSeq][t, 1][1]
        for g2 ∈ axes(Y[iSeq], 2)
            dFdy[g1+1, g2+1] = Y[iSeq][t, g2][13] - 1im * Y[iSeq][t, g2][12] + Y[iSeq][t, g2][10] - 1im * Y[iSeq][t, g2][9]
            dFdy[g2+1, g1+1] = Y[iSeq][t, g2][13] + 1im * Y[iSeq][t, g2][12] + Y[iSeq][t, g2][10] + 1im * Y[iSeq][t, g2][9]
        end
        dFdy[g1+1, g1+1] = 2 * real(dFdy[g1+1, g1+1])
        mul!(dFdy, Fi, mul!(tmp, dFdy, Fi))
        _dCRBdy_rw[iSeq][t, g1+1] = real.(w * diag(dFdy))
    end

    d = [(t, g) -> [_dCRBdx_fw[iSeq][t, 1], _dCRBdy_fw[iSeq][t, 1], 0, _dCRBdx_rw[iSeq][t, 1], _dCRBdy_rw[iSeq][t, 1], 0, 0, 0, _dCRBdx_fw[iSeq][t, g+1], _dCRBdy_fw[iSeq][t, g+1], 0, _dCRBdx_rw[iSeq][t, g+1], _dCRBdy_rw[iSeq][t, g+1], 0, 0, 0, 0] for iSeq ∈ eachindex(_dCRBdx_fw)]

    return CRB, d
end

function adjoint_backpropagate_3pool(d, Q, E)
    P = Array{Vector{Float64}}(undef, size(E, 1), size(E, 2))
    λ = @view P[end, :]

    for g ∈ 1:size(E, 2)
        λ[g] = d(size(E, 1), g)
    end

    for t = size(E, 1)-1:-1:1
        λ[1] = transpose(E[t+1, 1]) * λ[1]
        λ[1] += d(t, 1)
        for g = 2:size(E, 2)
            λ[g] = transpose(E[t+1, g][9:16, :]) * λ[g][9:16]
            λ[1][1:8] .+= λ[g][1:8]
            λ[g] .+= d(t, g)
        end
    end

    P[end, 1] = inv(transpose(Q[1])) * λ[1]
    for g = 2:size(E, 2)
        λ[g] = inv(transpose(Q[g]))[:, 9:16] * λ[g][9:16]
        λ[1][1:8] .+= λ[g][1:8]
    end

    for t ∈ size(E, 1):-1:2
        for g ∈ axes(E, 2)
            _E = E[mod(t, size(E, 1))+1, g]
            if g == 1
                P[t-1, g] = transpose(_E) * P[t, g]
            else
                P[t-1, g] = transpose(_E[9:16, :]) * P[t, g][9:16]
                P[t-1, 1][1:8] .+= P[t-1, g][1:8]
                P[t-1, 1][end] += P[t-1, g][end]
            end
            P[t-1, g] .+= d(t, g)
        end
    end
    return P
end

function control_gradients_3pool(P, Y, E, dEdω1, dEdTRF)
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
            @inbounds for i = 9:16
                grad_ω1[t] -= P[tm1, g][i] * a[i]
                grad_TRF[t] -= P[tm1, g][i] * b[i]
            end
        end
    end
    return grad_ω1, grad_TRF
end