using Pkg
Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()
using Symbolics
using MRIgeneralizedBloch

##
@variables ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R1a, T2s, R2s, dR2sdT2s, dR2sdB1, grad_type

f_R2s(T2s, B1) = R2s
@register_symbolic f_R2s(T2s, B1)
Symbolics.derivative(::typeof(f_R2s), args::NTuple{2,Any}, ::Val{1}) = dR2sdT2s
Symbolics.derivative(::typeof(f_R2s), args::NTuple{2,Any}, ::Val{2}) = dR2sdB1

H = hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, f_R2s(T2s, B1))

##
fs_str = ""
for p ∈ [m0s, R1f, R2f, Rx, R1s, T2s, B1, ω0, R1a]
    if isequal(p, R1a)
        Ḣ = expand_derivatives.(Differential(R1f).(H) .+ Differential(R1s).(H))
    else
        Ḣ = expand_derivatives.(Differential(p).(H))
    end

    dHdp = vcat(
        hcat(H[1:end-1, 1:end-1], zeros(Int, size(H, 1) - 1, size(H, 2) - 1), H[1:end-1, end]),
        hcat(Ḣ[1:end-1, 1:end-1], H[1:end-1, 1:end-1], Ḣ[1:end-1, end]),
        zeros(Int, 1, 2size(H, 2) - 1)
    )

    dHdp = substitute(dHdp, Dict([f_R2s(T2s, B1) => R2s]))
    dHdp = simplify.(dHdp)

    f_expr = build_function(dHdp, ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, dR2sdT2s, dR2sdB1, grad_type;
        force_SA=true,
    )

    f_str = string(f_expr[1])
    f_str = f_str[1:9] * "hamiltonian_linear" * f_str[10:end]

    idcs = findfirst("grad_type", f_str)
    f_str = f_str[1:idcs[end]] * "::grad_$p" * f_str[idcs[end]+1:end]

    idcs = findfirst("(SymbolicUtils.Code.create_array)(StaticArraysCore.SArray, nothing, Val{2}(), Val{($(size(dHdp, 1)), $(size(dHdp, 2)))}(), ", f_str)

    f_str = f_str[1:idcs[1]-1] * "SMatrix{$(size(dHdp, 1)), $(size(dHdp, 2))}(" * f_str[idcs[end]+1:end]
    global fs_str *= f_str
    global fs_str *= "\n \n"
end

##
write("src/MatrixExp_Hamiltonian_Gradients.jl", fs_str)
