#align(center)[
#text(size: 8.8pt)[
#table(
      columns: 10,
      column-gutter: 4pt,
      row-gutter: 1pt,
      align: (left, left, center, center, center, center, center, center, center, center),
      stroke: none,
      inset: (x, y) => (
        x: if y == 0 or y == 1 or x == 1 { 3pt } else { 0.5pt },
        y: if y == 0 or y == 1 { 4pt } else if y == 2 { 3pt } else if y == 3 { 2.9pt } else if x == 1 { 1.4pt } else { 1.1pt }
      ),

      // Header rows
      table.hline(stroke: .6pt),
      table.cell(rowspan: 2, [*Tipo*], align: horizon),
      table.cell(rowspan: 2, [*Modelo*], align: horizon),
      table.cell(rowspan: 2, [*MAP*], align: horizon),
      table.cell(colspan: 4, [*Métricas Rasas* $(k <= 10)$]),
      table.cell(colspan: 3, [*Métricas Profundas*]),

      table.hline(start: 3, end: 7, stroke: 0.3pt),
      table.hline(start: 7, end: 10, stroke: 0.3pt),

      // Sub-headers
      [P\@1], [P\@3], [P\@5], [P\@10], [P\@100], [P\@1000], [R-Prec],

      table.hline(stroke: .6pt),
      table.cell(colspan: 10, []),

      // Modelos Esparsos
      table.cell(rowspan: 7, [Esparso], align: horizon),
      [BM25+ (_offset_ positivo)], [*0.309*], [0.660], [0.491], [*0.440*], [*0.341*], [0.077], [0.011], [0.316],
      [BM25L (length normalization)], [0.309], [*0.667*], [0.502], [0.432], [0.336], [0.077], [*0.011*], [*0.318*],
      [Lucene (suavização do IDF)], [0.309], [*0.667*], [0.502], [0.432], [0.336], [0.077], [*0.011*], [*0.318*],
      [ATIRE (normalização alternativa)], [0.308], [*0.667*], [0.502], [0.432], [0.336], [0.077], [*0.011*], [*0.318*],
      [BMX (entropia + semântica)], [0.305], [0.640], [0.496], [0.436], [0.335], [*0.077*], [0.011], [0.318],
      [Pyserini BM25 com RM3], [0.302], [0.647], [*0.507*], [0.432], [0.339], [0.076], [0.010], [0.315],
      [BM25 Robertson (Baseline)], [0.296], [0.613], [0.464], [0.412], [0.325], [0.075], [0.010], [0.310],

      table.cell(colspan: 10, []),
      table.hline(stroke: .6pt),
      table.cell(colspan: 10, []),

      // Modelos Semânticos
      table.cell(rowspan: 6, [Semântico], align: horizon),
      [jina-embeddings-v3], [*0.402*], [0.713], [*0.629*], [*0.556*], [*0.443*], [*0.085*], [*0.011*], [*0.407*],
      [Qwen-Embedding-0.6B], [0.373], [*0.767*], [0.618], [0.539], [0.402], [0.079], [0.011], [0.373],
      [gte-multilingual-base (Alibaba)], [0.329], [0.673], [0.549], [0.473], [0.372], [0.080], [0.011], [0.338],
      [gte-lamdec-pairs], [0.300], [0.607], [0.489], [0.437], [0.337], [0.074], [0.010], [0.311],
      [gemma-lamdec-pairs], [0.154], [0.347], [0.293], [0.247], [0.193], [0.045], [0.009], [0.176],
      [qwen-lamdec-pairs], [0.143], [0.360], [0.287], [0.229], [0.167], [0.044], [0.008], [0.155],

      table.cell(colspan: 10, []),
      table.hline(stroke: .6pt),
      table.cell(colspan: 10, []),

      // Fusão
      table.cell(rowspan: 6, [Fusão], align: horizon),
      [_Mixed_ (WMNZ + WSUM)], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.462]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.813]], [0.711], [0.637], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.493]], [0.093], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.011]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.448]],
      [Sum Fusion], [0.461], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.813]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.713]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.639]], [0.492], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.093]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.011]], [0.444],
      [Weighted Sum], [0.461], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.813]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.713]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.639]], [0.492], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.093]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.011]], [0.444],
      [MNZ (Min-Non-Zero)], [0.461], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.813]], [0.711], [0.636], [0.492], [0.092], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.011]], [0.446],
      [GMNZ (Geometric MNZ)], [0.461], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.813]], [0.711], [0.636], [0.492], [0.092], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.011]], [0.446],
      [E muitos outros algoritmos...], [], [], [], [], [], [], [],

      table.hline(stroke: .6pt),
    )
  ]
  #v(.4em)
  #align(center)[Tabela 1. Precisão para o grupo 1, 2, 3. Valores em *itálico* indicam a melhor métrica dentro de cada categoria (Esparso, Semântico, Fusão). Valores em #rect(fill: rgb("#DAA520"), inset: 1pt)[dourado] indicam a melhor métrica global.]
]


#align(center)[
#text(size: 8.8pt)[
#table(
      columns: 10,
      column-gutter: 4pt,
      row-gutter: 1pt,
      align: (left, left, center, center, center, center, center, center, center, center),
      stroke: none,
      inset: (x, y) => (
        x: if y == 0 or y == 1 or x == 1 { 3pt } else { 0.5pt },
        y: if y == 0 or y == 1 { 4pt } else if y == 2 { 3pt } else if y == 3 { 2.9pt } else if x == 1 { 1.4pt } else { 1.1pt }
      ),

      // Header rows
      table.hline(stroke: .6pt),
      table.cell(rowspan: 2, [*Tipo*], align: horizon),
      table.cell(rowspan: 2, [*Modelo*], align: horizon),
      table.cell(rowspan: 2, [*MAP*], align: horizon),
      table.cell(colspan: 5, [*Métricas Rasas* $(k <= 10)$]),
      table.cell(colspan: 2, [*Métricas Profundas*]),

      table.hline(start: 3, end: 8, stroke: 0.3pt),
      table.hline(start: 8, end: 10, stroke: 0.3pt),

      // Sub-headers
      [R\@1], [R\@3], [R\@5], [R\@10], [F1\@10], [R\@100], [R\@1000],

      table.hline(stroke: .6pt),
      table.cell(colspan: 10, []),

      // Modelos Esparsos
      table.cell(rowspan: 7, [Esparso], align: horizon),
      [BM25+ (_offset_ positivo)], [*0.309*], [0.057], [0.126], [*0.187*], [*0.292*], [*0.313*], [0.648], [0.871],
      [BM25L (length normalization)], [0.309], [*0.058*], [0.129], [0.185], [0.288], [0.308], [0.645], [*0.875*],
      [Lucene (suavização do IDF)], [0.309], [*0.058*], [0.129], [0.185], [0.288], [0.308], [0.645], [*0.875*],
      [ATIRE (normalização alternativa)], [0.308], [*0.058*], [0.129], [0.185], [0.288], [0.308], [0.645], [*0.875*],
      [BMX (entropia + semântica)], [0.305], [0.056], [0.127], [0.186], [0.286], [0.307], [*0.649*], [0.872],
      [Pyserini BM25 com RM3], [0.302], [0.056], [*0.131*], [0.185], [0.289], [0.310], [0.642], [0.864],
      [BM25 Robertson (Baseline)], [0.296], [0.053], [0.119], [0.176], [0.279], [0.298], [0.636], [0.869],

      table.cell(colspan: 10, []),
      table.hline(stroke: .6pt),
      table.cell(colspan: 10, []),

      // Modelos Semânticos
      table.cell(rowspan: 6, [Semântico], align: horizon),
      [jina-embeddings-v3], [*0.402*], [0.061], [*0.160*], [*0.237*], [*0.376*], [*0.405*], [*0.707*], [*0.908*],
      [Qwen-Embedding-0.6B], [0.373], [*0.066*], [0.158], [0.229], [0.340], [0.367], [0.663], [0.877],
      [gte-multilingual-base (Alibaba)], [0.329], [0.057], [0.138], [0.199], [0.313], [0.339], [0.663], [0.870],
      [gte-lamdec-pairs], [0.300], [0.051], [0.123], [0.183], [0.284], [0.307], [0.617], [0.863],
      [gemma-lamdec-pairs], [0.154], [0.029], [0.075], [0.105], [0.162], [0.175], [0.376], [0.714],
      [qwen-lamdec-pairs], [0.143], [0.031], [0.072], [0.097], [0.141], [0.152], [0.363], [0.689],

      table.cell(colspan: 10, []),
      table.hline(stroke: .6pt),
      table.cell(colspan: 10, []),

      // Fusão
      table.cell(rowspan: 6, [Fusão], align: horizon),
      [_Mixed_ (WMNZ + WSUM)], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.462]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.070]], [0.181], [0.270], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.420]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.451]], [0.777], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.919]],
      [Sum Fusion], [0.461], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.070]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.182]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.270]], [0.419], [0.450], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.779]], [0.919],
      [Weighted Sum], [0.461], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.070]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.182]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.270]], [0.419], [0.450], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.779]], [0.919],
      [MNZ (Min-Non-Zero)], [0.461], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.070]], [0.181], [0.269], [0.419], [0.450], [0.772], [0.919],
      [GMNZ (Geometric MNZ)], [0.461], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.070]], [0.181], [0.269], [0.419], [0.450], [0.772], [0.919],
      [E muitos outros algoritmos...], [], [], [], [], [], [], [],

      table.hline(stroke: .6pt),
    )
  ]
  #v(.4em)
  #align(center)[Tabela 2. Revocação para o grupo 1, 2, 3. Valores em *itálico* indicam a melhor métrica dentro de cada categoria (Esparso, Semântico, Fusão). Valores em #rect(fill: rgb("#DAA520"), inset: 1pt)[dourado] indicam a melhor métrica global.]
]


#align(center)[
#text(size: 8.8pt)[
#table(
      columns: 10,
      column-gutter: 4pt,
      row-gutter: 1pt,
      align: (left, left, center, center, center, center, center, center, center, center),
      stroke: none,
      inset: (x, y) => (
        x: if y == 0 or y == 1 or x == 1 { 3pt } else { 0.5pt },
        y: if y == 0 or y == 1 { 4pt } else if y == 2 { 3pt } else if y == 3 { 2.9pt } else if x == 1 { 1.4pt } else { 1.1pt }
      ),

      // Header rows
      table.hline(stroke: .6pt),
      table.cell(rowspan: 2, [*Tipo*], align: horizon),
      table.cell(rowspan: 2, [*Modelo*], align: horizon),
      table.cell(rowspan: 2, [*MAP*], align: horizon),
      table.cell(colspan: 5, [*Métricas Rasas* $(k <= 10)$]),
      table.cell(colspan: 2, [*Métricas Profundas*]),

      table.hline(start: 3, end: 8, stroke: 0.3pt),
      table.hline(start: 8, end: 10, stroke: 0.3pt),

      // Sub-headers
      [nDCG\@1], [nDCG\@3], [nDCG\@5], [nDCG\@10], [MRR], [nDCG\@100], [nDCG\@1000],

      table.hline(stroke: .6pt),
      table.cell(colspan: 10, []),

      // Modelos Esparsos
      table.cell(rowspan: 7, [Esparso], align: horizon),
      [BM25+ (_offset_ positivo)], [*0.309*], [*0.640*], [0.524], [*0.489*], [*0.440*], [0.752], [*0.587*], [0.638],
      [BM25L (length normalization)], [0.309], [*0.640*], [0.534], [0.487], [0.438], [*0.755*], [0.586], [*0.639*],
      [Lucene (suavização do IDF)], [0.309], [*0.640*], [0.534], [0.487], [0.438], [*0.755*], [0.586], [0.639],
      [ATIRE (normalização alternativa)], [0.308], [*0.640*], [0.534], [0.487], [0.438], [*0.755*], [0.586], [0.639],
      [BMX (entropia + semântica)], [0.305], [0.618], [0.524], [0.487], [0.434], [0.740], [0.584], [0.636],
      [Pyserini BM25 com RM3], [0.302], [0.627], [*0.535*], [0.486], [0.439], [0.743], [0.585], [0.636],
      [BM25 Robertson (Baseline)], [0.296], [0.596], [0.492], [0.459], [0.417], [0.715], [0.567], [0.621],

      table.cell(colspan: 10, []),
      table.hline(stroke: .6pt),
      table.cell(colspan: 10, []),

      // Modelos Semânticos
      table.cell(rowspan: 6, [Semântico], align: horizon),
      [jina-embeddings-v3], [*0.402*], [0.676], [0.623], [*0.582*], [*0.532*], [0.811], [*0.661*], [*0.702*],
      [Qwen-Embedding-0.6B], [0.373], [*0.738*], [*0.628*], [0.580], [0.511], [*0.821*], [0.636], [0.684],
      [gte-multilingual-base (Alibaba)], [0.329], [0.629], [0.549], [0.504], [0.458], [0.773], [0.596], [0.644],
      [gte-lamdec-pairs], [0.300], [0.561], [0.489], [0.461], [0.416], [0.700], [0.551], [0.607],
      [gemma-lamdec-pairs], [0.154], [0.317], [0.293], [0.266], [0.240], [0.461], [0.326], [0.412],
      [qwen-lamdec-pairs], [0.143], [0.326], [0.281], [0.247], [0.216], [0.461], [0.311], [0.393],

      table.cell(colspan: 10, []),
      table.hline(stroke: .6pt),
      table.cell(colspan: 10, []),

      // Fusão
      table.cell(rowspan: 6, [Fusão], align: horizon),
      [_Mixed_ (WMNZ + WSUM)], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.462]], [0.782], [0.719], [0.673], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.605]], [0.885], [0.738], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.764]],
      [Sum Fusion], [0.461], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.784]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.722]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.675]], [0.604], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.885]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.739]], [0.764],
      [Weighted Sum], [0.461], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.784]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.722]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.675]], [0.604], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.885]], [#rect(fill: rgb("#DAA520"), inset: 1pt)[0.739]], [0.764],
      [MNZ (Min-Non-Zero)], [0.461], [0.782], [0.719], [0.672], [0.604], [0.885], [0.736], [0.763],
      [GMNZ (Geometric MNZ)], [0.461], [0.782], [0.719], [0.672], [0.604], [0.885], [0.736], [0.763],
      [E muitos outros algoritmos...], [], [], [], [], [], [], [],

      table.hline(stroke: .6pt),
    )
  ]
  #v(.4em)
  #align(center)[Tabela 3. nDCG para o grupo 1, 2, 3. Valores em *itálico* indicam a melhor métrica dentro de cada categoria (Esparso, Semântico, Fusão). Valores em #rect(fill: rgb("#DAA520"), inset: 1pt)[dourado] indicam a melhor métrica global.]
]
