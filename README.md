## Interpolação BM25 + Dense Retrievers no Contexto Jurídico Brasileiro

### Proposta

Avaliar a eficácia da interpolação entre modelos de recuperação densa e modelos
tradicionais de recuperação da informação (BM25) no contexto jurídico
brasileiro. Modelos de _embeddings_ costumam capturar bem sinais fortes de
relevância semântica, mas tendem a falhar em sinais fracos associados a termos
específicos, uma limitação que o BM25 pode compensar.

A interpolação é realizada da seguinte forma:

$$
s(p) = \alpha \cdot s_{\text{BM25}} (p) + (1 - \alpha) \cdot s_{\text{BERT}} (p)
$$

### Metodologia

O dataset utilizado será o JurisTCU[^2], composto por 16 mil documentos
jurisprudenciais do Tribunl de Contas da União e 150 consultas anotadas com
julgamento de relevância.

As consultas são divididas em três categorias:

- Consultas de usuários.
- Consultas sintéticas de palavras-chave.
- Consultas sintéticas de frases completas.

> [!IMPORTANT]
> Talvez avalie na base de dados da Procuradoria Geral do Município do Rio
> (PGM) contendo 231 mil peças de processos jurídicos.
> O critério para julgar um auto judicial como relevante é
> tema do artigo[^3].

Entre os modelos de embeddings que serão avaliados, estão:

- General Text Embeddings (GTE) do Alibaba.
- JINA Embeddings (JINA) do Jina AI.
- Versão finetuned do GTE (GTE-Finetuned) treinado com o linguajar jurídico.

As três abordagens avaliadas serão:

- [BM25 (_baseline_)](https://github.com/xhluca/bm25s).
- Interpolação de scores entre BM25 e DRs (baseados ou não em BERT).
- Interpolação de scores entre BM25 e embeddings fine-tuned.
- Modelos de embeddings (_baseline_).

> [!NOTE]
> Pretendo avaliar o desempenho da recuperação variando o parâmetro de
> interpolação $\alpha$ no intervalo de 0 a 1.

As métricas utilizadas incluem:

- Métricas rasas: Avaliam a precisão local para o topo do Ranking (P@10, MRR@10, nDCG@10).
- Métricas profundas: Medem a consistência global, se o sistema consegue capturar todos os relevantes (MAP, Recall@1000, nDCG@1000).

O modelo fine-tuned (gte-finetune-pgm) foi treinado com a função de perda
Multiple Negative Ranking Loss (MNRL) juntamente do Matryoshka Learning que
permite ajustar os embeddings a diferentes dimensões dependendo dos recursos
computacionais disponíveis.

A geração dos embeddings será feita com auxílio da técnica de Aggregate Mean
Pooling, onde cada documentos é segmentados em chunks (passagens), sendo cada
uma codificada em embeddings de dimensão fixa os quais serão agregados para
representação dos documentos.

### Relação com o Artigo Base

O artigo base[^1] identificou ganhos expressivos ao
interpolar scores entre BM25 e DRs baseados em BERT (RepBERT, ANCE e CLEAR),
sobretudo em métricas profundas.

Embora o presente trabalho se baseie fortemente nesse artigo, também incorpora
ideias de outros dois estudos:

[^1]: [_Recuperadores densos baseados em BERT exigem interpolação com BM25 para recuperação efetiva de passagens_](https://arvinzhuang.github.io/publication/ICTIR2021InterpolateDR).
[^2]: [_JurisTCU: Um conjunto de dados de recuperação de informação em português do Brasil com julgamentos de relevância de consulta_](https://arxiv.org/abs/2503.08379).
[^3]: [_Análise da Eficácia de Fine-Tuning de Embeddings no Contexto Jurídico Brasileiro_](./articles-reference/article-matheus.pdf).

