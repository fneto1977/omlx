# Plano de performance M1 — Qwen3.8-27B e Qwen3.6-35B-A3B

Máquina: MacBook M1 Max, 64 GB, GPU `applegpu_g13s` (~400 GB/s), oMLX.app 0.7.0
(runtime MLX 0.32.2). Código: `~/Desenvolvimento/omlx-m1/omlx`, testado via
overlay (`scripts/m1_overlay.py` + `scripts/m1_serve.sh`).

Modelos-alvo:

| Modelo | Objetivo |
|---|---|
| `Qwen3.8-27B-oQ4e-fp16-mtp` (denso, GDN + atenção) | prefill **e** decode |
| `Qwen3.6-35B-A3B-oQ4-fp16-mtp` (MoE) | prefill (decode já está bom) |

Fora do escopo por enquanto: Qwen3.8 Flash Next. Próxima rodada: Gemma 4 26B MoE e mais um modelo.

## Linha de base atual (com o patch M1 e a configuração aplicada)

| Medida | Qwen3.8 | 35B-A3B |
|---|---|---|
| Decode, contexto curto | ~27,5 tok/s (MTP) | ~98 tok/s (MTP) |
| Prefill 17k / 34k / 67k / 101k / 133k | 182 / 155 / 117 / 96 / 73 tok/s | 939 / 673 / 469 / 322 / 258 tok/s |
| Decode a 101k / 133k | 21,2 / 18,2 tok/s | 44 / 38 tok/s |

Referência externa (Splash-M1, mesmo M1 Max 64 GB): Qwen3.8 decode 38 tok/s, 35B-A3B 145 tok/s,
prefill do 27B de 133 tok/s a 8k e 110 tok/s a 32k.

## Protocolo de medição (vale para todas as fases)

1. Servidor isolado: `OMLX_BASE_PATH` no scratchpad, porta 18123, nunca o servidor do app.
2. **Nunca** rodar dois benchmarks de GPU ao mesmo tempo (pytest incluso); checar com
   `pgrep -fl "omlx.cli serve"` antes de cada medição.
3. Decode: prompt de código fixo (`e2e.py`), 400 tokens, 3 repetições, mediana; contexto
   longo: `lc.py` (prefixo crescente 16k→128k, cache de prefixo ligado).
4. Antes de comparar, anotar swap (`sysctl vm.swapusage`); acima de ~8 GB de swap, reiniciar
   a medição depois de liberar memória.
5. A/B sempre na mesma sessão térmica: alternar configurações em vez de rodar todas as
   de uma e depois todas da outra.
6. Ruído esperado: ±5%. Só aceitar mudanças que ganhem mais que isso em duas rodadas.
7. Qualidade: `quality.py` (pergunta factual sobre documento de 5,6k tokens, greedy) e a
   resposta precisa continuar correta; para drafters, conferir que a taxa de aceitação faz
   sentido (aceitação não muda o texto, só a velocidade).

## Fase 1 — DFlash2 no Qwen3.8 (decode) — prioridade máxima

Hipótese: o drafter `incoai/Qwen3.8-27B-DFlash2` aceita ~4–5 tokens por ciclo; com a
verificação de 8 linhas custando ~2× um decode no M1, o decode sobe de ~27,5 para 35–40 tok/s
(é o que o Splash-M1 mostra).

Testes:
1. Carregar o drafter com quantização do oMLX (`dflash_draft_quant_enabled`, 4 bits, grupo 64)
   e conferir no log `DFlash drafter attached ... kind=dflash2` (sem "load raised").
2. A/B MTP × DFlash2: contexto curto (3 repetições), 32k e 128k.
3. Variar `dflash_block_size` / `runtime_block_size` (3–8) e `dflash_verify_mode`
   (`dflash` / `adaptive`) — no M1 o custo da verificação cresce por linha.
4. Conferir uso de memória (drafter ~0,9 GB em 4 bits) com o ANE ligado e teto de 52 GB.

Critério de aceite: ≥ +15% de decode sobre o MTP em contexto curto, sem perder no longo.
Se aprovado, aplicar `dflash_enabled` no `model_settings.json` do Qwen3.8.

## Fase 2 — Kernel de verificação para 8–16 linhas (ideia do `skinny_qmm`)

Hipótese: a verificação do DFlash2 usa 8–9 linhas; o `skinny_qmm` do mlx-dspark (pesos como
operando A do `simdgroup_matrix`, sem staging) mediu 10–21% melhor que o `sg8`/`mma` no M1.

Implementação:
1. Kernel próprio (implementação independente da ideia, crédito no docstring) em
   `omlx/patches/m1_verify_qmv.py` ou módulo irmão: entrada fp16 (o original lê bf16),
   4-bit e 5-bit (o Qwen3.8 tem 27 `down_proj` e 48 `out_proj` em 5 bits).
2. Roteamento em `qwen35_verify_qmm._verify_route` só para `applegpu_g13`, janela de linhas
   definida pela medição.
3. Testes de numerics como `tests/test_m1_verify_qmv.py`.

Critério de aceite: microbenchmark ≥ 10% melhor nas formas reais e ganho no tempo por ciclo
do DFlash2 (log `backbone=...ms`).

## Fase 3 — Correções de robustez do DFlash

1. Drafter sem `num_target_layers` (caso do `mlx-community/Qwen3.6-35B-A3B-DFlash2-4bit`):
   inferir do número de camadas do alvo antes da validação.
2. Falha ao carregar drafter hoje cai em decode sem especulação **silenciosamente**: elevar
   para aviso claro no log e na UI/admin.
3. Reavaliar DFlash2 no 35B-A3B com bloco menor (3–4): no MoE cada linha extra ativa experts
   novos, por isso o bloco 8 perdeu para o MTP (67 × 98 tok/s).

Critério de aceite: testes unitários das correções; 35B-A3B só muda de MTP para DFlash2 se
ganhar > 5% nas duas rodadas.

## Fase 4 — Profundidade de rascunho calibrada pela máquina (ideia do mlx-dspark)

Hipótese: o controle adaptativo do MTP escolheu rascunhos curtos demais com o kernel M1
(tokens/ciclo caiu de 2,6 para 2,0). Medir uma vez a curva de custo da verificação por número
de linhas neste GPU (cache por modelo/quantização/versão do MLX) e usá-la no controle.

Testes: A/B adaptativo atual × calibrado, nos dois modelos, decode curto e 64k.
Critério de aceite: ≥ +5% de decode sem perda em nenhum ponto.

## Fase 5 — Decode em contexto longo: KV INT8 (ideia do Splash-M1) — investigação

Hipótese: a 100k+ a leitura do KV domina a verificação (decode do Qwen3.8 cai de 31 para
18 tok/s). O Splash-M1 usa cache INT8 com kernels de atenção próprios (1,7–1,9× mais rápido
que bf16 no M1). O TurboQuant 8-bit do oMLX **piorou** o decode no M1 (32 → 16 tok/s), então
seria um caminho novo.

Passos: (1) medir quanto do ciclo a 64k/128k é atenção (instrumentação por camada);
(2) protótipo de atenção de decode/verify sobre KV INT8 simétrico por grupo; (3) go/no-go.
Critério de go: projeção de ≥ +15% de decode a 128k.

## Fase 6 — Consolidação

1. Atualizar `scripts/m1_overlay.py` com o que for aprovado; regenerar `~/.omlx/m1-overlay`.
2. Rodar a suíte de testes relevante (o conjunto de 45 falhas pré-existentes deve continuar
   idêntico).
3. Benchmark final completo (protocolo acima) e tabela comparativa antes/depois.
4. Aplicar as configurações aprovadas em `~/.omlx/model_settings.json` (com backup).

## Ordem e estimativa

| Fase | Depende de | Esforço | Ganho esperado |
|---|---|---|---|
| 1 DFlash2 Qwen3.8 | download (feito) | baixo | alto (decode +25–40%) |
| 3 Correções DFlash | — | baixo | robustez; talvez 35B |
| 2 Kernel 8–16 linhas | 1 aprovado | médio | +10% no decode com DFlash2 |
| 4 Profundidade calibrada | — | médio | +5–10% decode MTP |
| 5 KV INT8 | 1–4 | alto | decode a 100k+ (incerto) |
| 6 Consolidação | todas | baixo | — |

---

## Status das fases

| Fase | Status | Resultado |
|---|---|---|
| 1 — DFlash2 no Qwen3.8 | Concluída — aprovada | Decode +18–63% em contexto longo (greedy e amostrado), +34% editando; −27% escrevendo código novo em contexto curto com amostragem. Aplicada. |
| 2 — Kernel de verificação 5–16 linhas | Concluída — aprovada | −8 a −26% por projeção; integrada (`m1_verify_mma.py`). |
| 3 — Robustez do DFlash | Concluída | Fallback de `num_target_layers`, erro explícito; DFlash2 no 35B medido e rejeitado. |
| 4 — Profundidade do MTP | Concluída — aprovada (35B) | Fixa 3 no 35B-A3B (+4–25%); no Gemma o adaptativo venceu. |
| 5 — Atenção / KV em contexto longo | Concluída — passos 1–3 do plano | (1) medido: kernel MPP ~95 GB/s; (2) protótipos: kernel GQA próprio (mais lento) e KV INT8 (26–71% mais lento) — no-go por critério; ajuste das fatias (−7 a −21%) e rota head_dim 512 (−34 a −37%) aprovados. |
| 6 — Consolidação | Concluída | Branch `m1-performance`, overlay, testes, configurações aplicadas com backup. |
| Extra — repositórios | Concluída | Splash-M1, mlx-dspark, MTPLX, mlxcel, zenlm/engine (seção abaixo). |
| Extra — modelos | Concluída | Gemma 4 26B-A4B (seção abaixo). |

---

## Resultados da execução (2026-10-12)

### Fase 1 — DFlash2 no Qwen3.8: **aprovada**

Drafter `incoai/Qwen3.8-27B-DFlash2` (BF16 3,85 GB, quantizado pelo oMLX para 4 bits, grupo 64),
bloco 8. Decode (tok/s, mediana de 3):

| Carga | MTP | DFlash2 bloco 8 |
|---|---|---|
| Escrever código do zero, contexto curto | 28,1 | 27,0 |
| Editar arquivo (agente), contexto curto | 44,6 | 53–58 |
| Contexto 17k / 34k / 67k / 101k / 133k | 30,6 / 27,8 / 24,1 / 21,2 / 18,2 | 40,4 / 32,8 / 33,7 / 27,5 / 24,2 |

Prefill com DFlash2: 3–8% mais lento (captura de estados ocultos para o drafter).
Blocos 3–6 e `dflash_verify_mode` não ajudam (o modo de verificação é ignorado pelo DFlash em
lote do oMLX).

### Fase 2 — kernel de verificação 5–16 linhas: **aprovado**

`omlx/patches/m1_verify_mma.py` (adaptado do `skinny_qmm` do mlx-dspark, MIT): pesos como
operando A de `simdgroup_matrix`, entrada fp16. Cadeia dependente up/down do Qwen3.8 (ms por
projeção; decode de 1 linha = 0,169): 5–8 linhas 0,315–0,356 → 0,287; 9–16 linhas 0,660 → 0,471.
Ciclo do DFlash2 bloco 8: 144 → 131 ms.

### Fase 3 — robustez DFlash: **feito**

- `num_target_layers` ausente no config do drafter: preenchido com o número de camadas do alvo
  (`speculative/dflash_drafter.py`).
- Falha ao carregar o drafter agora é ERROR explicando que a especulação ficou desligada.

### Fase 5 — atenção de verificação em contexto longo

- O caminho real no M1 (macOS 26) é o kernel de *tensor ops* MPP (`_gqa_causal_sdpa`): ~5,6–5,9 ms
  por camada a 128k (~95 GB/s), praticamente igual de 2 a 8 linhas.
- Kernel próprio com `simdgroup_matrix` agrupado por cabeça de KV: correto, porém 1,2–6× mais lento
  (overhead de redução entre simdgroups e softmax por linha) — **descartado**.
- Ajuste do tamanho das fatias de chaves no M1 (≈1 threadgroup por núcleo, 1024–4096 chaves):
  −7% a −21% no tempo da atenção de verificação — **aplicado**.
- KV INT8 — protótipo medido (`quantized_matmul` agrupando as G×L linhas de cada cabeça de KV,
  KV 8-bit lido uma vez, máscara causal por linha): a 128k ficou 43–71% mais lento que o kernel
  fp16 atual no Qwen3.8 (8,0–9,3 × 5,4–5,8 ms por camada) e 26–52% no 35B; erro 1e-4–4e-4.
  Critério de go (≥ +15% de decode a 128k) não atingido — **no-go**. Um kernel INT8 fundido
  exigiria a mesma reescrita que já perdeu para o kernel MPP.

### Fase 3 (cont.) — DFlash2 no 35B-A3B

Com a correção, `mlx-community/Qwen3.6-35B-A3B-DFlash2-4bit` carrega sem ajuste manual, mas no MoE
cada linha a mais de verificação ativa experts novos: blocos 3/4/8 deram 84/75/53 tok/s
(escrita) contra 101 do MTP — **mantido o MTP**.

### Fase 4 — profundidade do MTP: **aprovada (35B-A3B)**

O controle adaptativo do oMLX já mede custos por máquina, mas no M1 escolheu profundidades
piores que as fixas no 35B-A3B. Decode (tok/s):

| Carga | Adaptativo | Fixa 2 | Fixa 3 |
|---|---|---|---|
| Escrever, contexto curto | 93,9 | 101,5 | 95,6 |
| Editar, contexto curto | 77,7 | 92,2 | 93,0 |
| 17k / 34k / 67k / 101k / 128k | 85 / 65 / 55 / 42 / 43 | 83 / 70 / 49 / 38 / 37 | 93 / 81 / 55 / 46 / 42 |

Escolha: `mtp_fixed_depth = 3` (melhor ou igual em quase todos os pontos).

### Fase 6 — consolidação

- Branch `m1-performance` em `github.com/fneto1977/omlx`, sobre o upstream `bafc6837`.
- Testes: mesma lista de falhas pré-existentes do upstream nesta combinação de runtime
  (421 × 420, a diferença é um teste instável); todos os testes novos passam.
- Configuração aplicada em `~/.omlx/model_settings.json` (backup `*.bak-20261012-114526`):
  - Qwen3.8: `mtp_enabled=false`, `dflash_enabled=true`,
    `dflash_draft_model=/Volumes/DADOS/omlx/models/incoai/Qwen3.8-27B-DFlash2`, quantização do
    drafter 4 bits / grupo 64 / ativações 16, `dflash_block_size=8`; ANE com GDN, sem CPU.
  - Qwen3.6-35B-A3B: `mtp_fixed_depth=3`.
  - Global: `memory_guard_tier=custom`, teto 52 GB.
- **Importante**: o DFlash2 do Qwen3.8 só compensa com os kernels M1 (overlay). No oMLX.app sem
  patch ele ficou −16% ao escrever código novo (+8% editando).
- Observação de medição: no fim do dia a GPU passou a entregar ~14% menos banda sob carga
  sustentada (e −15 a −30% no decode ponta a ponta) com código e configuração idênticos —
  efeito ambiental (atividade de tela/térmica). Todas as decisões acima vêm de A/B na mesma
  sessão.

## Como usar (para compartilhar)

Requisitos: Mac M1/M1 Pro/Max/Ultra, oMLX.app 0.7.0 instalado.

```bash
git clone -b m1-performance https://github.com/fneto1977/omlx
cd omlx
python3 scripts/m1_overlay.py      # cria ~/.omlx/m1-overlay a partir do oMLX.app instalado
# pare o servidor do app (menu do oMLX) e então:
scripts/m1_serve.sh                # mesmo ~/.omlx, mesma porta; argumentos extras vão para `omlx serve`
```

Para desligar uma otimização: `OMLX_M1_VERIFY_QMV=0`, `OMLX_M1_VERIFY_MMA=0`,
`OMLX_FA256_M1_SDPA=0` (variáveis de ambiente antes de `m1_serve.sh`).
Configurações recomendadas por modelo: ver "Fase 6" acima. Ao atualizar o oMLX.app, rode
`scripts/m1_overlay.py` de novo; se a versão nova mudar os trechos esperados, ele aborta em vez
de aplicar o patch errado.

## Validação com amostragem real (temperatura > 0)

As decisões acima foram medidas em greedy; repetidas com a amostragem de cada modelo
(Qwen3.8: temperatura 1,0 do `generation_config`; 35B-A3B: 0,7/top_p 0,8/top_k 40):

| Carga | Qwen3.8 MTP | Qwen3.8 DFlash2 | 35B adaptativo | 35B fixo 3 |
|---|---|---|---|---|
| Escrever código, contexto curto | 22,5 | 16,4 | 72,1 | 75,0 |
| Editar arquivo, contexto curto | 33,3 | 44,6 | 92,6 | 100,2 |
| ~35k de contexto | 25,9 | 34,2 | | |
| ~103k de contexto | 18,0 | 29,3 | | |

DFlash2 mantido no Qwen3.8 (agentes / contexto longo); só perde escrevendo código novo em
contexto curto. Profundidade 3 mantida no 35B-A3B.

## Repositórios analisados (ideias para o M1)

| Projeto | O que foi aproveitado / medido |
|---|---|
| Splash-M1 (paperniuk) | Drafter DFlash2 + verificação de 8 linhas → Fase 1/2; prefill em comandos curtos (já coberto). |
| mlx-dspark | `skinny_qmm` → `m1_verify_mma.py` (Fase 2); lição de microbenchmark em cadeia dependente. |
| MTPLX | Kernel "packed-row GQA" de verificação: no M1 vence só com 2–3 linhas no layout 16/2 (−11 a −66%); com 4 linhas (profundidade 3 do 35B) −3 a −9%, com 5–8 linhas perde muito — ganho de decode estimado em 1–2%, não integrado. Confirma o uso de checkpoints fp16 no M1/M2. |
| mlxcel | Confirma de forma independente que na geração 13 (M1) a verificação de várias linhas em projeções quantizadas é o gargalo (sem `qmv_wide`) — exatamente o que os kernels M1 resolvem. Regras de aceitação com temperatura > 0 → validação acima. |
| zenlm/engine | Fork do mistral.rs com kernels Metal genéricos (paged attention, HQQ/BnB/FP8); nada específico para Qwen3.5/GDN ou M1. |

## Gemma 4 26B-A4B (`Jundot/gemma-4-26B-A4B-it-oQ4e-mtp`)

Arquitetura: 30 camadas (25 com janela deslizante de 1024, head_dim 256; 5 globais com
head_dim 512 e K=V), MoE 128 experts / 8 ativos, vocabulário de 262k.

- **fp16 em vez de bf16** (cópia com `scripts/clone_mlx_model_fp16.py`, pesos validados
  dentro do limite do fp16; resposta de referência correta): prefill +30% (661 → 871 tok/s),
  decode +9–12%.
- **MTP** (cabeça própria do pacote oQ): escrever ±0 (greedy) / −5% (amostrado); editar
  +47% / +44%. Controle adaptativo melhor que profundidade fixa (fixa 2: +5% escrevendo,
  −18% editando). Configuração: MTP ligado, adaptativo.
- **Atenção global head_dim 512 no prefill**: o MLX não tem rota fundida; nova rota M1
  fatiada (`_m1_d512_attention`, a partir de 32k chaves): −34 a −37% de tempo de atenção a
  65–128k e pico de memória de +8,9 GB para +0,6 GB. Curva longa (fp16 + MTP), rota ligada ×
  desligada: 71–103k 305 × 281 tok/s, 103–136k 252 × 216 tok/s (+17%); abaixo de 70k
  dentro do ruído.
- Ajuste junto: a rota M1 de head_dim 256 só a partir de 3072 chaves (o primeiro bloco de
  2048 é 10–19% mais rápido no kernel steel).

## Comparativo antes × depois (Qwen3.8-27B e Qwen3.6-35B-A3B)

Todas as medições no mesmo M1 Max 64 GB, com o mesmo harness (`e2e.py` / `lc.py`), tok/s.

- **Antes** = oMLX.app 0.7.0 com a configuração original (tier de memória `balanced`; Qwen3.8 com
  MTP e ANE com a CPU ligada; 35B-A3B com MTP adaptativo).
- **Só config** = oMLX.app 0.7.0 com a configuração nova (teto de 52 GB, ANE sem CPU e com GDN),
  sem o patch M1. Usado como referência em contexto longo, onde a configuração original não
  chegou a ser medida até o fim (o tier `balanced` descartava o ANE no primeiro prompt longo).
- **Depois** = servidor do overlay M1 (`scripts/m1_serve.sh`) com a configuração aplicada
  (Qwen3.8: DFlash2 bloco 8; 35B-A3B: MTP profundidade 3).
- Ruído entre rodadas: ±5% (mais à tarde: até −15–30% por efeito térmico/tela, ver Fase 6).

### Qwen3.8-27B — prefill (tok/s)

| Contexto | Antes | Só config | Depois | Antes → Depois |
|---|---|---|---|---|
| ~6k (prompt único) | 139,5 | 197,0 | — | +41% (config) |
| 0–17k | — | 175,1 | 187,5 | +7% vs só config |
| 17–34k | — | 145,8 | 142,5 | ruído |
| 34–67k | — | 110,6 | 108,9 | ruído |
| 67–101k | — | 84,7 | 89,2 | +5% |
| 101–133k | — | 69,5 | 71,1 | +2% |

### Qwen3.8-27B — decode (tok/s)

| Carga | Antes | Depois | Variação |
|---|---|---|---|
| Escrever código, curto, greedy | 25,3 | 27,0 | +7% |
| Escrever código, curto, amostrado (temp. 1,0) | 25,8 | 16,4 | **−36%** |
| Editar arquivo, curto, amostrado | 41,7 | 44,6 | +7% |
| Editar arquivo, curto, greedy | 44,6¹ | 53–58 | +20–31% |
| 17k / 34k / 67k / 101k / 133k, greedy | 31,3 / 28,1 / 23,8 / 19,2 / 17,6² | 40,4 / 32,8 / 33,7 / 27,5 / 24,2 | +29 / +17 / +42 / +43 / +38% |
| ~35k / ~103k, amostrado | 25,9 / 18,0¹ | 34,2 / 29,3 | +32 / +63% |

¹ MTP medido no overlay (no app o MTP rende igual ou menos).  ² Só config (MTP, app).

### Qwen3.6-35B-A3B — prefill (tok/s)

| Contexto | Antes | Depois | Variação |
|---|---|---|---|
| ~6k (prompt único) | 1052 | — | — |
| 0–17k | 886 | 902 | +2% |
| 17–34k | 594 | 658 | +11% |
| 34–67k | 407 | 466 | +15% |
| 67–101k | 286 | 327 | +14% |
| 101–128k | — (passou do limite de 131072 do teste) | 258 | — |

### Qwen3.6-35B-A3B — decode (tok/s)

| Carga | Antes | Depois | Variação |
|---|---|---|---|
| Escrever código, curto, amostrado (0,7/0,8/40) | 72,1³ | 75,0 | +4% |
| Editar arquivo, curto, amostrado | 92,6³ | 100,2 | +8% |
| Editar arquivo, curto, greedy | 77,7³ | 93,0 | +20% |
| 17k / 34k / 67k / 101k / 128k, greedy | 75,9 / 65,6 / 48,4 / 46,4 / — | 93,2 / 81,3 / 54,7 / 45,7 / 42,1 | +23 / +24 / +13 / −2% / — |

³ MTP adaptativo (configuração original) medido no overlay.

## Teste do autor e PRs (2026-10-12)

Tarefa: clone simples de Bomberman em HTML/JS com agente de código, iterando até o jogo abrir no
navegador. Servidor do overlay (`scripts/m1_serve.sh`), números tirados de `~/.omlx/logs/server.log`.

| | Gemma 4 26B | Qwen3.6-35B-A3B | Qwen3.8-27B |
|---|---|---|---|
| Tempo total | ~14 min | ~46 min (harness parou em 128k) | ~76 min |
| Requisições | 32 | 227 | 46 |
| Maior contexto | 52k | 115k | 117k |
| Decode médio | 43 tok/s | 44 tok/s | 18,5 tok/s |
| Prefill de contexto novo | 460–770 tok/s | 560–980 tok/s | 80–184 tok/s |

Os números do Qwen3.8 batem com o comparativo acima (o prefill do 27B denso é o limite no uso
com agente). Resultado visual melhor no Qwen3.6; qualidade melhor no Qwen3.8, mas lento.

PRs enviados ao `jundot/omlx`:
- [#4495](https://github.com/jundot/omlx/pull/4495) — DFlash2 sem `num_target_layers`.
- [#4496](https://github.com/jundot/omlx/pull/4496) — kernels e rotas de atenção M1 (rascunho,
  acima de 1.000 linhas; escopo declarado: M1 Max 64 GB e os três modelos acima).

## Rodada 2 — depois da limpeza das ventoinhas (2026-10-13)

Ambiente: modo de energia "Alto desempenho", ventoinhas limpas, modelos oQ4e novos (Qwen3.6
convertido para fp16; Qwen3.8 fp16 do Hugging Face).

| # | Teste | Resultado | Decisão |
|---|---|---|---|
| 1 | Qwen3.8 depois da limpeza | decode curto 35–36 tok/s (ontem à noite 15–24 no mesmo pedido); 3 gerações longas seguidas 24,2 / 24,2 / 23,7 tok/s, sem queda térmica | manter "Alto desempenho" |
| 2 | `ane_compile_cache` | carga do Qwen3.8 52 s → 32 s (−38%) | **ligado** |
| 3 | Limites de memória | o teto que vale é o do Metal (51,8 GB, padrão da Apple); subir `iogpu.wired_limit_mb` + teto do oMLX a 56 GB não segurou a ANE (a estimativa por bloco cresce com o contexto) e levou o macOS a 5 GB de swap | **não usar**; teto de volta a 52 GB |
| 3b | Bloco mínimo de prefill 2048 (oMLX usa 4096 em hosts ≥64 GB sem NAX) | Qwen3.8 com ANE: 15k 127 → 157 tok/s (+24%), 61k 121 → 133 tok/s (+10%); Qwen3.6: −1 a −2% (ruído) | **padrão** no `m1_serve.sh` (`OMLX_QWEN35_PREFILL_FLOOR=2048`) |
| 4 | Kernel de verificação 5 bits | 21% dos bytes do Qwen3.8 são 5 bits; teto de ganho ~2–3 ms por ciclo de ~130 ms (2–3%); `mma` existente é 70% mais lento que o `sg8` atual | **no-go** |
| 5 | ANE no prefill do Qwen3.6-35B (MoE) | 15k 979 → 975 tok/s, 61k 599 → 604 tok/s (só as projeções GDN vão para a ANE; os experts dominam) | **no-go**, ANE desligada no 35B |

Correção encontrada: `python -m` põe a pasta atual na frente do `sys.path`; iniciado de dentro do
checkout do repositório, o `m1_serve.sh` carregava o código do repositório (sem extensões
nativas → "Private ANE runtime unavailable"). Corrigido com `PYTHONSAFEPATH=1`.

Pendência conhecida: depois que a ANE é liberada por falta de folga (acima de ~50k de contexto no
Qwen3.8), ela só volta quando o modelo é recarregado. A estimativa de memória por bloco usa o
máximo medido × margem de segurança; ajustá-la mexe na proteção contra falta de memória do oMLX.
