using Pkg
Pkg.activate("docs")
Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()

using MRIgeneralizedBloch
using Documenter
using Literate
using Plots # to not capture precompilation output

# HTML Plotting Functionality
struct HTMLPlot
    p # :: Plots.Plot
end
const ROOT_DIR = joinpath(@__DIR__, "build")
const PLOT_DIR = joinpath(ROOT_DIR, "plots")
function Base.show(io::IO, ::MIME"text/html", p::HTMLPlot)
    mkpath(PLOT_DIR)
    path = joinpath(PLOT_DIR, string(hash(p) % UInt32, ".html"))
    Plots.savefig(p.p, path)
    if get(ENV, "CI", "false") == "true" # for prettyurl
        print(io, "<object type=\"text/html\" data=\"../../$(relpath(path, ROOT_DIR))\" style=\"width:100%;height:425px;\"></object>")
    else
        print(io, "<object type=\"text/html\" data=\"../$(relpath(path, ROOT_DIR))\" style=\"width:100%;height:425px;\"></object>")
    end
end

# Notebook hack to display inline math correctly
function notebook_filter(str)
    re = r"(?<!`)``(?!`)"  # Two backquotes not preceded by nor followed by another
    replace(str, re => "\$")
end

# Literate
OUTPUT = joinpath(@__DIR__, "src/build_literate")

FILE = joinpath(@__DIR__, "src/Greens_functions.jl")
Literate.markdown(FILE, OUTPUT)
# Literate.notebook(FILE, OUTPUT, preprocess=notebook_filter)
Literate.script(  FILE, OUTPUT)

FILE = joinpath(@__DIR__, "src/Simulation_ContinuousWave.jl")
Literate.markdown(FILE, OUTPUT)
# Literate.notebook(FILE, OUTPUT, preprocess=notebook_filter)
Literate.script(  FILE, OUTPUT)

FILE = joinpath(@__DIR__, "src/Simulation_Pulse.jl")
Literate.markdown(FILE, OUTPUT)
# Literate.notebook(FILE, OUTPUT, preprocess=notebook_filter)
Literate.script(  FILE, OUTPUT)

FILE = joinpath(@__DIR__, "src/Analyze_NMR_Data.jl")
Literate.markdown(FILE, OUTPUT)
# Literate.notebook(FILE, OUTPUT, preprocess=notebook_filter)
Literate.script(  FILE, OUTPUT)

DocMeta.setdocmeta!(MRIgeneralizedBloch, :DocTestSetup, :(using MRIgeneralizedBloch); recursive=true)

makedocs(;
    doctest = false,
    modules=[MRIgeneralizedBloch],
    authors="Jakob Asslaender <jakob.asslaender@nyumc.org> and contributors",
    repo="https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl/blob/{commit}{path}#{line}",
    sitename="MRIgeneralizedBloch.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JakobAsslaender.github.io/MRIgeneralizedBloch.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Quick Start Tutorial" => "tutorial.md",
        "generalized Bloch Paper" => Any[
        "build_literate/Greens_functions.md",
        "build_literate/Simulation_ContinuousWave.md",
        "build_literate/Simulation_Pulse.md",
        "build_literate/Analyze_NMR_Data.md",
        ],
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/JakobAsslaender/MRIgeneralizedBloch.jl",
)
