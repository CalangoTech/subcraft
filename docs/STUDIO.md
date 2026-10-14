# SubCraft Studio 0.2.0-studio.1

Fork de [Aalzard/subcraft](https://github.com/Aalzard/subcraft), a partir de `eb3f374`, mantido em [CalangoTech/subcraft](https://github.com/CalangoTech/subcraft/tree/codex/subcraft-studio). Mantém as licenças MIT/Apache-2.0 e os créditos originais.

## O que mudou

**Bake:** high poly sem UV em OBJ, FBX, glTF e GLB. A malha de destino continua exigindo UV. O caminho de importação de detalhes dispensa MikkTSpace e a duplicação de vértices por costuras UV. A importação de geometria glTF não depende de imagens de materiais externos. Raios que não atingem o high poly usam a geometria low coerente, inclusive para ID/AO/thickness/curvature. Anti-aliasing de 1, 4 ou 16 amostras por pixel, normais renormalizadas e IDs categóricos. Configurações inválidas são recusadas antes de alocar os grandes buffers.

**Camadas:** arraste pela alça à esquerda para reordenar ou colocar dentro de pastas; marcadores indicam o destino. Menus oferecem agrupamento e movimentação para pastas/raiz. F2 ou duplo clique renomeia, Enter confirma e Esc cancela. Bloqueio persistente protege conteúdo e descendentes tanto na interface quanto nas APIs de comandos/pintura. Operações inválidas não alteram o documento; operações sem mudança preservam histórico e redo.

**Máscaras:** Blur, Dilate/Expand, Erode e Contrast não destrutivos, com opacidade e raio de 0–32 texels. Filtros leem vizinhos reais, incluindo limites de tiles, e o dirty tracking propaga o raio acumulado. A biblioteca agora oferece 12 smart masks e 8 smart materials, incluindo cobre patinado, concreto envelhecido e esmalte descascado. Salvar projetos e fragmentos preserva os novos efeitos.

**Espaço de trabalho:** controles rápidos de pincel/canal/cor; conta-gotas de material em 3D e 2D; inspeção de canais sem iluminação; projeção ortográfica; exposição; presets Painting, 3D + 2D e Classic. Quality 100/75/50% controla somente a resolução do viewport. A pintura e a exportação continuam na resolução da textura.

**Projetos:** recentes, snapshots de recuperação em background e intervalo configurável. A cópia não muda o caminho salvo nem limpa o indicador de alterações. O pacote `PORTABLE` mantém cópias em `Recovery` ao lado do executável; instalações normais usam `%LOCALAPPDATA%/SubCraft Studio/Recovery`. `SUBCRAFT_RECOVERY` pode escolher outra pasta. Mantém as vinte cópias de sessão mais recentes, preservando a recuperação aberta e arquivos comuns.

## Desempenho e correção

- A revisão da sessão evita percorrer toda a árvore de camadas em cada sincronização ociosa.
- Os snapshots dos canais efetivamente enviados à GPU determinam uploads parciais. Exportação e conta-gotas podem avaliar o compositor sem consumir uma atualização pendente do viewport.
- Imagens 3D/2D são reutilizadas até mudar câmera, resolução, canal, material, luz ou identidade do ambiente. A Shelf deixa de agendar redraws quando termina as miniaturas.
- A lista de camadas coleta hierarquia e metadados de bloqueio em uma passagem. O indicador de modificações usa cache por revisão, preservando o retorno ao estado salvo por undo.
- Resultados de raios do bake são processados em blocos de 2048, com reutilização de vetores de direções. O array temporário completo de resultados foi eliminado: até aproximadamente 192 MiB a menos no limite de 4096² amostras.
- Normais misturadas por opacidade são renormalizadas antes de armazenar/exportar.
- Prefixos de máscaras com filtros espaciais são compartilhados entre tiles/threads em um cache restrito ao batch atual, com limite de 64 MiB e fallback local. Uma edição começa outro batch, sem reutilizar resultados antigos.

## Validação reproduzível

Ferramentas usadas neste Windows: Rust 1.99 GNU, GCC/MinGW UCRT e NVIDIA RTX 3060. As ferramentas portáteis ficam em `.tools` no disco D, fora do Git. Para usar o ambiente local já preparado:

```powershell
. .\scripts\use-local-toolchain.ps1
cargo test --workspace
cargo clippy --workspace --all-targets
cargo run --profile dist -p subcraft-app --example studio_validate -- dist\Demo_Studio
cargo test --profile dist -p subcraft-ui --test performance benchmark_idle_sync -- --ignored --nocapture
cargo run --profile dist -p subcraft-bake --example benchmark -- 256 1 3
cargo test --profile dist -p subcraft-compositor --test mask_filters spatial_filter_benchmark -- --ignored --nocapture
.\scripts\build-windows.ps1
```

Em outra máquina, instale Rust estável (1.95+) e o linker apropriado; o script de ambiente local é opcional. O executável distribuído já é compilado e não precisa dessas ferramentas.

A suíte final passou **255 testes**, com dois benchmarks manuais ignorados na execução padrão. O smoke `studio_validate` importa um OBJ high poly real sem UV com 4032 triângulos, faz bake de sete mapas para dois texture sets, aplica materiais, pinta, filtra uma máscara, reorganiza/bloqueia/desfaz/refaz, salva/reabre o projeto e compara mapas e composites. Exporta e decodifica 24 PNGs de 256² nos templates PBR, Unity HDRP e Unreal. A execução final registrada terminou em **0,641 s**, com o bake em **0,097 s**. São tempos desse exemplo pequeno nesta máquina, não uma previsão para assets de produção.

O benchmark ocioso com 101 camadas e 100 mil chamadas mediu **64,180 ms** avaliando o caminho antigo de sincronização versus **0,309 ms** no caminho com revisão, cerca de **208× nessa rotina específica**. Não é uma medida de FPS nem de aceleração de todo o programa. Quality 50% reduz a área renderizada para aproximadamente 25%, sem mudar as texturas.

Uma máscara 1024² com gerador, Dilate 3 e Blur 2, incluindo preparação de geometria, mediu **13,351 s** com avaliações locais repetidas, **722,54 ms** com cache compartilhado sequencial e **132,45 ms** com cache paralelo. Ganhos de aproximadamente **18,5×** e **100,8×** nesse stack de teste, com resultados bitexatos. A retenção medida foi **8,13 MiB**, sob o limite de 64 MiB. Isso compara os caminhos dos novos filtros, não todos os workloads do SubCraft nem uma versão do Substance Painter.

O benchmark de bake UVless 256²/1 amostra, quatro mapas, doze workers e três iterações produziu checksum estável; a mediana registrada foi **417,639 ms**. Tempos variam com carga, geometria, configurações e hardware. Os benchmarks ficam ignorados na suíte padrão para não exigir thresholds dependentes da máquina.

A interface nativa foi aberta pelo executável e capturada pelo próprio modo `--screenshot`, sem automação do mouse/teclado. Veja [a captura](screenshots/studio-main.png). Os testes de GPU também verificam valores de canais isolados e atualização após a prévia de exportação.

## Compatibilidade e limites

Lê projetos schema 1 e 2; novos projetos/fragmentos usam schema 3. Versões antigas do SubCraft recusam schema 3. Preserve seus projetos originais ao migrar.

O limite de bake é 4096² **amostras**: resolução × fator por eixo ≤ 4096. Trabalhos próximos desse limite ainda podem consumir vários GB de RAM. Cancelamento é verificado em checkpoints; importação, BVH e dilatação podem demorar a terminar a etapa atual.

É uma evolução funcional de um programa inicial. Clone/smudge, pintura direta no canvas 2D, UDIM, compositor/bake GPU e leitura nativa de SBSAR continuam fora desta versão. A interface principal mantém os termos técnicos em inglês; o guia rápido acompanha o pacote em português.

Clippy normal conclui; o código herdado ainda tem avisos de estilo sob Rust 1.99 e a dependência `nalgebra 0.26.2` gera um aviso de compatibilidade futura. Isso não impediu os testes nem o build otimizado.
