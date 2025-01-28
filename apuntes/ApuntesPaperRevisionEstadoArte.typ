#let leading = 1em

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

#set heading(
    numbering: "1."
)

#show heading: it => {
  it
  v(leading, weak: true) 
}

#strong(text(size: 15pt)[Apuntes de la Revisión del Estado del Arte de Ciberataques Neuronales])

\
#strong(outline(indent: 1em))
#pagebreak()

= Resumen
