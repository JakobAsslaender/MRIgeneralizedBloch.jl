using MRIgeneralizeBloch
using Documenter

DocMeta.setdocmeta!(MRIgeneralizeBloch, :DocTestSetup, :(using MRIgeneralizeBloch); recursive=true)

makedocs(;
    modules=[MRIgeneralizeBloch],
    authors="Jakob Asslaender <jakob.asslaender@nyumc.org> and contributors",
    repo="https://github.com/JakobAsslaender/MRIgeneralizeBloch.jl/blob/{commit}{path}#{line}",
    sitename="MRIgeneralizeBloch.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JakobAsslaender.github.io/MRIgeneralizeBloch.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JakobAsslaender/MRIgeneralizeBloch.jl",
)
