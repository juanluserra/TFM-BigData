#let leading = 1.2em

#set text(
    lang: "ES",
    hyphenate: false,
)

#set par(
    leading: leading,
    justify: true,
    spacing: leading*1.5,
)

#set page(
    number-align: center,
    numbering: "1",
)

#show heading: it => {
  it
  v(leading, weak: true) 
}

= Apuntes de Taxonomía de Ciberataques Neuronales

