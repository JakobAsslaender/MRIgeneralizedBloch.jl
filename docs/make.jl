using MyPkg
using Documenter

DocMeta.setdocmeta!(MyPkg, :DocTestSetup, :(using MyPkg); recursive=true)

makedocs(;
    modules=[MyPkg],
    authors="Jakob Asslaender <jakob.asslaender@nyumc.org> and contributors",
    repo="https://github.com/JakobAsslaender/MyPkg.jl/blob/{commit}{path}#{line}",
    sitename="MyPkg.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JakobAsslaender.github.io/MyPkg.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JakobAsslaender/MyPkg.jl",
)
