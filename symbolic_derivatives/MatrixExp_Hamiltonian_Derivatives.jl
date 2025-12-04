using Pkg
Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()
using Symbolics
using Symbolics: SConst
using MRIgeneralizedBloch

##
@variables ω1, B1, ω0, T, m0s, R1f, R2f, Rex, R1s, R1a, T2s, R2s, dR2sdT2s, dR2sdB1, grad_type
@variables TRF, dR2sdB1, dR2sdω1, dR2sdTRF, dR2sdT2sdω1, dR2sdB1dω1, dR2sdT2sdTRF, dR2sdB1dTRF

f_R2s(T2s, B1, ω1, TRF) = R2s
@register_symbolic f_R2s(T2s, B1, ω1, TRF)
@register_derivative f_R2s(T2s, B1, ω1, TRF) 1 SConst(dR2sdT2s)
@register_derivative f_R2s(T2s, B1, ω1, TRF) 2 SConst(dR2sdB1)

H = hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rex, R1s, f_R2s(T2s, B1, ω1, TRF))
Z = zeros(Int, size(H))

## #########################################################################################
# derivatives wrt. MT parameters (used for CRB calculations & NLLS fitting)
############################################################################################
fs_str = ""
for p ∈ [m0s, R1f, R2f, Rex, R1s, T2s, B1, ω0, R1a]
    if isequal(p, R1a)
        Ḣ = expand_derivatives.(Differential(R1f).(H) .+ Differential(R1s).(H))
    else
        Ḣ = expand_derivatives.(Differential(p).(H))
    end

    dHdp = vcat(
        hcat(H[1:end-1, 1:end-1], Z[1:end-1, 1:end-1], H[1:end-1, end]),
        hcat(Ḣ[1:end-1, 1:end-1], H[1:end-1, 1:end-1], Ḣ[1:end-1, end]),
        zeros(Int, 1, 2size(H, 2) - 1)
    )

    dHdp = substitute(dHdp, Dict([f_R2s(T2s, B1, ω1, TRF) => R2s]))

    f_expr = build_function(dHdp, ω1, B1, ω0, T, m0s, R1f, R2f, Rex, R1s, R2s, dR2sdT2s, dR2sdB1, grad_type;
        force_SA=true,
    )

    f_str = string(f_expr[1])
    f_str = f_str[1:9] * "hamiltonian_linear" * f_str[10:end]

    idcs = findfirst("grad_type", f_str)
    f_str = f_str[1:idcs[end]] * "::grad_$p" * f_str[idcs[end]+1:end]

    idcs = findfirst("(SymbolicUtils.Code.create_array)(StaticArraysCore.SArray, nothing, Val{2}(), Val{($(size(dHdp, 1)), $(size(dHdp, 2)))}(), ", f_str)
    f_str = f_str[1:idcs[1]-1] * "SMatrix{$(size(dHdp, 1)), $(size(dHdp, 2))}(" * f_str[idcs[end]+1:end]
    global fs_str *= f_str
    global fs_str *= "\n"
end

## #########################################################################################
# derivatives wrt. ω1 (used for optimal control)
############################################################################################
f_dR2sdω1(T2s, B1) = dR2sdω1
@register_symbolic f_dR2sdω1(T2s, B1)

@register_derivative f_R2s(T2s, B1, ω1, TRF) 3 f_dR2sdω1(T2s, B1)
@register_derivative f_dR2sdω1(T2s, B1) 1 SConst(dR2sdT2sdω1)
@register_derivative f_dR2sdω1(T2s, B1) 2 SConst(dR2sdB1dω1)

Ḣ = expand_derivatives.(Differential(ω1).(H))
Ḣ = substitute(Ḣ, Dict([f_dR2sdω1(T2s, B1) => dR2sdω1]))

# no grad_type
f_expr = build_function(Matrix(Ḣ), B1, T, dR2sdω1, dR2sdT2sdω1, dR2sdB1dω1, grad_type;
    force_SA=true,
)

f_str = string(f_expr[1])
f_str = f_str[1:9] * "d_hamiltonian_linear_dω1" * f_str[10:end]

idcs = findfirst("dR2sdT2sdω1", f_str)
f_str = f_str[1:idcs[end]] * "=nothing" * f_str[idcs[end]+1:end]
idcs = findfirst("dR2sdB1dω1", f_str)
f_str = f_str[1:idcs[end]] * "=nothing" * f_str[idcs[end]+1:end]
idcs = findfirst("grad_type", f_str)
f_str = f_str[1:idcs[end]] * "=nothing" * f_str[idcs[end]+1:end]

idcs = findfirst("(SymbolicUtils.Code.create_array)(StaticArraysCore.SArray, nothing, Val{2}(), Val{($(size(Ḣ, 1)), $(size(Ḣ, 2)))}(), ", f_str)
f_str = f_str[1:idcs[1]-1] * "SMatrix{$(size(Ḣ, 1)), $(size(Ḣ, 2))}(" * f_str[idcs[end]+1:end]
fs_str *= f_str
fs_str *= "\n"


# grad_type::grad_param (generic)
dHdp = vcat(
    hcat(Ḣ[1:end-1, 1:end-1], Z[1:end-1, 1:end-1], Ḣ[1:end-1, end]),
    hcat(Z[1:end-1, 1:end-1], Ḣ[1:end-1, 1:end-1], Ḣ[1:end-1, end]),
    zeros(Int, 1, 2size(H, 2) - 1)
)

f_expr = build_function(dHdp, B1, T, dR2sdω1, dR2sdT2sdω1, dR2sdB1dω1, grad_type;
    force_SA=true,
)

f_str = string(f_expr[1])
f_str = f_str[1:9] * "d_hamiltonian_linear_dω1" * f_str[10:end]

idcs = findfirst("grad_type", f_str)
f_str = f_str[1:idcs[end]] * "::grad_param" * f_str[idcs[end]+1:end]

idcs = findfirst("(SymbolicUtils.Code.create_array)(StaticArraysCore.SArray, nothing, Val{2}(), Val{($(size(dHdp, 1)), $(size(dHdp, 2)))}(), ", f_str)
f_str = f_str[1:idcs[1]-1] * "SMatrix{$(size(dHdp, 1)), $(size(dHdp, 2))}(" * f_str[idcs[end]+1:end]

fs_str *= f_str
fs_str *= "\n"


# grad_type::grad_B1 and grad_type::grad_T2s
for p ∈ [B1, T2s]
    Ḣ = expand_derivatives.(Differential(ω1).(H))
    dḢdp = expand_derivatives.(Differential(p).(Ḣ))

    dHdp = vcat(
        hcat(Ḣ[1:end-1, 1:end-1], Z[1:end-1, 1:end-1], Ḣ[1:end-1, end]),
        hcat(dḢdp[1:end-1, 1:end-1], Ḣ[1:end-1, 1:end-1], Ḣ[1:end-1, end]),
        zeros(Int, 1, 2size(H, 2) - 1)
    )
    dHdp = substitute(dHdp, Dict([f_dR2sdω1(T2s, B1) => dR2sdω1]))

    f_expr = build_function(dHdp, B1, T, dR2sdω1, dR2sdT2sdω1, dR2sdB1dω1, grad_type;
        force_SA=true,
    )

    f_str = string(f_expr[1])
    f_str = f_str[1:9] * "d_hamiltonian_linear_dω1" * f_str[10:end]

    idcs = findfirst("grad_type", f_str)
    f_str = f_str[1:idcs[end]] * "::grad_$p" * f_str[idcs[end]+1:end]

    idcs = findfirst("(SymbolicUtils.Code.create_array)(StaticArraysCore.SArray, nothing, Val{2}(), Val{($(size(dHdp, 1)), $(size(dHdp, 2)))}(), ", f_str)
    f_str = f_str[1:idcs[1]-1] * "SMatrix{$(size(dHdp, 1)), $(size(dHdp, 2))}(" * f_str[idcs[end]+1:end]

    global fs_str *= f_str
    global fs_str *= "\n"
end

## #########################################################################################
# derivatives wrt. TRF (used for optimal control)
############################################################################################
f_dR2sdTRF(T2s, B1) = dR2sdTRF
@register_symbolic f_dR2sdTRF(T2s, B1)

@register_derivative f_R2s(T2s, B1, ω1, TRF) 4 f_dR2sdTRF(T2s, B1)
@register_derivative f_dR2sdTRF(T2s, B1) 1 SConst(dR2sdT2sdTRF)
@register_derivative f_dR2sdTRF(T2s, B1) 2 SConst(dR2sdB1dTRF)

Ḣ = expand_derivatives.(Differential(TRF).(H))
Ḣ = substitute(Ḣ, Dict([f_dR2sdTRF(T2s, B1) => dR2sdTRF]))

# no grad_type
f_expr = build_function(Matrix(Ḣ), T, dR2sdTRF, dR2sdT2sdTRF, dR2sdB1dTRF, grad_type;
    force_SA=true,
)

f_str = string(f_expr[1])
f_str = f_str[1:9] * "d_hamiltonian_linear_dTRF_add" * f_str[10:end]

idcs = findfirst("dR2sdT2sdTRF", f_str)
f_str = f_str[1:idcs[end]] * "=nothing" * f_str[idcs[end]+1:end]
idcs = findfirst("dR2sdB1dTRF", f_str)
f_str = f_str[1:idcs[end]] * "=nothing" * f_str[idcs[end]+1:end]
idcs = findfirst("grad_type", f_str)
f_str = f_str[1:idcs[end]] * "=nothing" * f_str[idcs[end]+1:end]

idcs = findfirst("(SymbolicUtils.Code.create_array)(StaticArraysCore.SArray, nothing, Val{2}(), Val{($(size(Ḣ, 1)), $(size(Ḣ, 2)))}(), ", f_str)
f_str = f_str[1:idcs[1]-1] * "SMatrix{$(size(Ḣ, 1)), $(size(Ḣ, 2))}(" * f_str[idcs[end]+1:end]
fs_str *= f_str
fs_str *= "\n \n"


# grad_type::grad_param (generic)
dHdp = vcat(
    hcat(Ḣ[1:end-1, 1:end-1], Z[1:end-1, 1:end-1], Ḣ[1:end-1, end]),
    hcat(Z[1:end-1, 1:end-1], Ḣ[1:end-1, 1:end-1], Ḣ[1:end-1, end]),
    zeros(Int, 1, 2size(H, 2) - 1)
)

f_expr = build_function(dHdp, T, dR2sdTRF, dR2sdT2sdTRF, dR2sdB1dTRF, grad_type;
    force_SA=true,
)

f_str = string(f_expr[1])
f_str = f_str[1:9] * "d_hamiltonian_linear_dTRF_add" * f_str[10:end]

idcs = findfirst("grad_type", f_str)
f_str = f_str[1:idcs[end]] * "::grad_param" * f_str[idcs[end]+1:end]

idcs = findfirst("(SymbolicUtils.Code.create_array)(StaticArraysCore.SArray, nothing, Val{2}(), Val{($(size(dHdp, 1)), $(size(dHdp, 2)))}(), ", f_str)
f_str = f_str[1:idcs[1]-1] * "SMatrix{$(size(dHdp, 1)), $(size(dHdp, 2))}(" * f_str[idcs[end]+1:end]

fs_str *= f_str
fs_str *= "\n"


# grad_type::grad_B1 and grad_type::grad_T2s
for p ∈ [B1, T2s]
    Ḣ = expand_derivatives.(Differential(TRF).(H))
    dḢdp = expand_derivatives.(Differential(p).(Ḣ))

    dHdp = vcat(
        hcat(   Ḣ[1:end-1, 1:end-1], Z[1:end-1, 1:end-1], Ḣ[1:end-1, end]),
        hcat(dḢdp[1:end-1, 1:end-1], Ḣ[1:end-1, 1:end-1], Ḣ[1:end-1, end]),
        zeros(Int, 1, 2size(H, 2) - 1)
    )
    dHdp = substitute(dHdp, Dict([f_dR2sdTRF(T2s, B1) => dR2sdTRF]))

    f_expr = build_function(dHdp, T, dR2sdTRF, dR2sdT2sdTRF, dR2sdB1dTRF, grad_type;
        force_SA=true,
    )

    f_str = string(f_expr[1])
    f_str = f_str[1:9] * "d_hamiltonian_linear_dTRF_add" * f_str[10:end]

    idcs = findfirst("grad_type", f_str)
    f_str = f_str[1:idcs[end]] * "::grad_$p" * f_str[idcs[end]+1:end]

    idcs = findfirst("(SymbolicUtils.Code.create_array)(StaticArraysCore.SArray, nothing, Val{2}(), Val{($(size(dHdp, 1)), $(size(dHdp, 2)))}(), ", f_str)
    f_str = f_str[1:idcs[1]-1] * "SMatrix{$(size(dHdp, 1)), $(size(dHdp, 2))}(" * f_str[idcs[end]+1:end]

    global fs_str *= f_str
    global fs_str *= "\n"
end

## #########################################################################################
# remove cryptic and changing comments
############################################################################################
starts = findall("#=", fs_str)
ends   = findall("=#", fs_str)
@assert length(starts) == length(ends) "Mismatched #= and =# markers"

fsc_str = fs_str[1:starts[1][1]-1]
for i in 1:length(starts)
    from = ends[i][2] + 1
    to = i == length(starts) ? lastindex(fs_str) : starts[i+1][1] - 1
    global fsc_str *= fs_str[from:to]
end

# remove whitespace lines
fsc_str = join(filter(x -> !isempty(strip(x)), split(fsc_str, '\n')), "\n")

## #########################################################################################
write("src/MatrixExp_Hamiltonian_Gradients.jl", fsc_str)