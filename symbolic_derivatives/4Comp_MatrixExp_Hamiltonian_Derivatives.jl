using Pkg
Pkg.activate("symbolic_derivatives")
Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()
using Symbolics
using MRIgeneralizedBloch

##
@variables ω1, B1, ω0, T, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG, T2_C, T2_PG, dR2dT2_C, dR2dB1_C, dR2dT2_PG, dR2dB1_PG, grad_type

f_R2_C(T2_C, B1) = R2_C
@register_symbolic f_R2_C(T2_C, B1)
Symbolics.derivative(::typeof(f_R2_C), args::NTuple{2,Any}, ::Val{1}) = dR2dT2_C
Symbolics.derivative(::typeof(f_R2_C), args::NTuple{2,Any}, ::Val{2}) = dR2dB1_C

f_R2_PG(T2_PG, B1) = R2_PG
@register_symbolic f_R2_PG(T2_PG, B1)
Symbolics.derivative(::typeof(f_R2_PG), args::NTuple{2,Any}, ::Val{1}) = dR2dT2_PG
Symbolics.derivative(::typeof(f_R2_PG), args::NTuple{2,Any}, ::Val{2}) = dR2dB1_PG

H = hamiltonian_linear(ω1, B1, ω0, T, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, f_R2_C(T2_C, B1), f_R2_PG(T2_PG, B1))

##
fs_str = ""
for p ∈ [m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, T2_C, T2_PG, B1, ω0]
    D = Differential(p)
    Ḣ = expand_derivatives.(D.(H))

    dHdp = vcat(
        hcat(H[1:end-1, 1:end-1], zeros(Int, size(H,1)-1, size(H,2)-1), H[1:end-1, end]),
        hcat(Ḣ[1:end-1, 1:end-1], H[1:end-1, 1:end-1], Ḣ[1:end-1, end]),
        zeros(Int, 1, 2size(H,2)-1)
    )

    dHdp = substitute(dHdp, Dict([f_R2_C(T2_C, B1) => R2_C, f_R2_PG(T2_PG, B1) => R2_PG]))
    dHdp = simplify.(dHdp)

    f_expr = build_function(dHdp,
    ω1, B1, ω0, T, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG, dR2dT2_C, dR2dB1_C, dR2dT2_PG, dR2dB1_PG, grad_type;
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
write("src/4Comp_MatrixExp_Hamiltonian_Gradients.jl", fs_str)
