#align(center)[
#text(size: 8.8pt)[
#table(
      columns: 8,
      column-gutter: 4pt,
      row-gutter: 1pt,
      align: (left, left, center, center, center, center, center, center),
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
      table.cell(colspan: 5, [*Tempo de Consulta (ms)*]),

      table.hline(start: 3, end: 8, stroke: 0.3pt),

      // Sub-headers
      *Média*, *Mediana*, *P95*, *P99*, *Máx*,

      table.hline(stroke: .6pt),
      table.cell(colspan: 8, []),

      // Modelos Esparsos
      table.cell(rowspan: 7, [Esparso], align: horizon),
      [BM25+ (_offset_ positivo)], [0.309], [#rect(fill: rgb("#90EE90"), inset: 1pt)[1.248]], [#rect(fill: rgb("#90EE90"), inset: 1pt)[1.219]], [#rect(fill: rgb("#90EE90"), inset: 1pt)[1.429]], [#rect(fill: rgb("#90EE90"), inset: 1pt)[1.517]], [4.071],
      [BM25L (length normalization)], [0.309], [1.274], [1.249], [1.463], [1.803], [4.062],
      [Lucene (suavização do IDF)], [0.309], [1.265], [1.231], [1.500], [1.636], [4.037],
      [ATIRE (normalização alternativa)], [0.308], [1.302], [1.231], [1.696], [2.440], [4.284],
      [BMX (entropia + semântica)], [0.305], [3.920], [1.734], [2.231], [2.930], [325.695],
      [Pyserini BM25 com RM3], [0.302], [41.844], [41.084], [48.478], [52.060], [124.087],
      [BM25 Robertson (Baseline)], [0.296], [1.351], [1.293], [1.823], [2.966], [#rect(fill: rgb("#90EE90"), inset: 1pt)[3.890]],

      table.cell(colspan: 8, []),
      table.hline(stroke: .6pt),
      table.cell(colspan: 8, []),

      // Modelos Semânticos
      table.cell(rowspan: 6, [Semântico], align: horizon),
      [jina-embeddings-v3], [0.402], [95.789], [93.657], [100.710], [106.097], [371.476],
      [Qwen-Embedding-0.6B], [0.373], [102.256], [101.723], [107.587], [111.873], [165.264],
      [gte-multilingual-base (Alibaba)], [0.329], [*83.274*], [*82.622*], [*90.318*], [*92.333*], [129.927],
      [gte-lamdec-pairs], [0.300], [91.982], [91.799], [99.843], [102.641], [*107.524*],
      [gemma-lamdec-pairs], [0.154], [105.013], [104.405], [111.565], [114.790], [120.042],
      [qwen-lamdec-pairs], [0.143], [107.881], [106.342], [119.141], [125.193], [126.472],

      table.cell(colspan: 8, []),
      table.hline(stroke: .6pt),
      table.cell(colspan: 8, []),

      // Fusão
      table.cell(rowspan: 6, [Fusão], align: horizon),
      [_Mixed_ (WMNZ + WSUM)], [0.462], [137.362], [136.604], [143.119], [148.792], [194.907],
      [Sum Fusion], [0.461], [131.342], [130.926], [137.001], [142.557], [200.150],
      [Weighted Sum], [0.461], [138.660], [137.988], [145.639], [148.696], [211.934],
      [MNZ (Min-Non-Zero)], [0.461], [*129.172*], [*128.746*], [*134.893*], [*137.543*], [*187.551*],
      [GMNZ (Geometric MNZ)], [0.461], [134.204], [133.607], [140.578], [145.428], [208.638],
      [E muitos outros algoritmos...], [], [], [], [], [], [],

      table.hline(stroke: .6pt),
    )
  ]
  #v(.4em)
  #align(center)[Tabela. Tempo de execução de consultas para o grupo 1, 2, 3. Valores em *itálico* indicam o menor tempo (mais rápido) dentro de cada categoria (Esparso, Semântico, Fusão). Valores em #rect(fill: rgb("#90EE90"), inset: 1pt)[verde] indicam o menor tempo global.]
]
