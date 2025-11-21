using Pkg
Pkg.activate("symbolic_derivatives")
Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()
using Symbolics
using MRIgeneralizedBloch

##
@variables ω1, B1, ω0, T, m0_M, m0_NM, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM, T2_M, T2_NM, dR2dT2_M, dR2dB1_M, dR2dT2_NM, dR2dB1_NM, grad_type

f_R2_M(T2_M, B1) = R2_M
@register_symbolic f_R2_M(T2_M, B1)
Symbolics.derivative(::typeof(f_R2_M), args::NTuple{2,Any}, ::Val{1}) = dR2dT2_M
Symbolics.derivative(::typeof(f_R2_M), args::NTuple{2,Any}, ::Val{2}) = dR2dB1_M

f_R2_NM(T2_NM, B1) = R2_NM
@register_symbolic f_R2_NM(T2_NM, B1)
Symbolics.derivative(::typeof(f_R2_NM), args::NTuple{2,Any}, ::Val{1}) = dR2dT2_NM
Symbolics.derivative(::typeof(f_R2_NM), args::NTuple{2,Any}, ::Val{2}) = dR2dB1_NM

H = hamiltonian_linear(ω1, B1, ω0, T, m0_M, m0_NM, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, f_R2_M(T2_M, B1), f_R2_NM(T2_NM, B1))

##
fs_str = ""
for p ∈ [m0_M, m0_NM, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, T2_M, T2_NM, B1, ω0]
    D = Differential(p)
    Ḣ = expand_derivatives.(D.(H))

    dHdp = vcat(
        hcat(H[1:end-1, 1:end-1], zeros(Int, size(H,1)-1, size(H,2)-1), H[1:end-1, end]),
        hcat(Ḣ[1:end-1, 1:end-1], H[1:end-1, 1:end-1], Ḣ[1:end-1, end]),
        zeros(Int, 1, 2size(H,2)-1)
    )

    dHdp = substitute(dHdp, Dict([f_R2_M(T2_M, B1) => R2_M, f_R2_NM(T2_NM, B1) => R2_NM]))
    dHdp = simplify.(dHdp)

    f_expr = build_function(dHdp,
    ω1, B1, ω0, T, m0_M, m0_NM, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM, dR2dT2_M, dR2dB1_M, dR2dT2_NM, dR2dB1_NM, grad_type;
        force_SA=false,
    )

    f_str = string(f_expr[1])
    f_str = f_str[1:9] * "hamiltonian_linear" * f_str[10:end]

    idcs = findfirst("grad_type", f_str)
    f_str = f_str[1:idcs[end]] * "::grad_$p" * f_str[idcs[end]+1:end]

    idcs = findfirst("(SymbolicUtils.Code.create_array)(Array, nothing, Val{2}(), Val{($(size(dHdp, 1)), $(size(dHdp, 2)))}(), ", f_str)
    f_str = f_str[1:idcs[1]-1] * "reshape([" * f_str[idcs[end]+1:end]

    idcs = findfirst(", 0)\n", f_str)
    f_str = f_str[1:idcs[1]-1] * ", 0], $(size(dHdp, 1)), $(size(dHdp, 2)))\n" * f_str[idcs[end]+1:end]

    global fs_str *= f_str
    global fs_str *= "\n \n"
end

##
# Find all start and end markers
starts = findall("#=", fs_str)
ends   = findall("=#", fs_str)

# Sanity check: number of starts and ends should match
@assert length(starts) == length(ends) "Mismatched #= and =# markers"

# Start with the part before the first block comment
fsc_str = fs_str[1:starts[1][1]-1]

# Append all non-comment regions
for i in 1:length(starts)
    from = ends[i][2] + 1
    to = i == length(starts) ? lastindex(fs_str) : starts[i+1][1] - 1
    global fsc_str *= fs_str[from:to]
end

# remove whitespace lines
fsc_str = join(filter(x -> !isempty(strip(x)), split(fsc_str, '\n')), "\n")

##
write("src/4Comp_MatrixExp_Hamiltonian_Gradients.jl", fsc_str)
