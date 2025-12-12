# Sistemas Evolutivos: Cobra de Lamarck 
Este repositório contém o projeto final da disciplina **SSC0713: Sistemas Evolutivos Aplicados à Robótica**, ministrada pelo Prof. Dr. Eduardo do Valle Simões, na Universidade de São Paulo (USP).

O projeto implementa um agente autônomo (uma "Cobra") capaz de aprender a jogar o clássico Snake Game utilizando uma **abordagem híbrida de Inteligência Artificial**, que combina **Aprendizado por Reforço (Deep Q-Learning)** para aprendizado intra-vida e **Algoritmos Genéticos (GA)** para evolução entre gerações.

## Sobre o Projeto

O objetivo principal do agente é maximizar o número de maçãs coletadas, evitando colisões com paredes e com o próprio corpo.

Este trabalho explora a **sinergia entre o aprendizado por reforço e a evolução biológica simulada** (o conceito de *Cobra de Lamarck*), onde o conhecimento adquirido durante a vida de um agente é transferido e refinado pela seleção genética da população.

## Principais Características da Arquitetura

### 1. Sistema Híbrido: Deep Q-Learning (DQN) + Algoritmos Genéticos (GA)

* **Deep Q-Learning (DQN):** Treina a tomada de decisão imediata de cada agente com base nas recompensas do jogo (comer maçã ou colidir). A rede neural ajusta seus pesos em tempo real via **Backpropagation + Adam Optimizer**.
* **Algoritmo Genético (GA):** Atua no nível da população, selecionando os melhores agentes de uma geração para reprodução, garantindo que as estratégias de sucesso sejam passadas adiante.

### 2. Arquitetura da Rede Neural (Linear\_QNet)

Cada agente possui uma rede neural feedforward, construída com PyTorch, para mapear o estado do jogo para uma ação.

| Componente | Detalhes |
| :--- | :--- |
| **Entradas (Estado)** | **11 neurônios.** Representam o ambiente (perigo nas 3 direções, direção atual (4), e posição da comida (4)). |
| **Camada Oculta** | 256 neurônios (configurável) com Ativação **ReLU**. |
| **Saídas (Ação)** | **3 neurônios.** O agente escolhe uma entre as três ações possíveis: `[1, 0, 0]` (Seguir reto), `[0, 1, 0]` (Virar à direita), ou `[0, 0, 1]` (Virar à esquerda). |

### 3. Processamento Paralelo e Diversidade

O projeto utiliza o módulo `torch.multiprocessing` para treinar 20 agentes simultaneamente, acelerando o processo evolutivo. A **Alta Diversidade Genética** é mantida pela introdução de agentes "imigrantes" (novos agentes aleatórios) a cada geração, prevenindo a convergência prematura a mínimos locais.

## Metodologia Evolutiva

O treinamento avança em gerações de 20 agentes, seguindo este ciclo:

### 1. Treinamento (Deep Q-Learning)
* Agentes jogam em paralelo, aprendendo via DQN.
* Recompensas: Positiva ao comer, Negativa ao colidir.

### 2. Seleção
* Agentes são ranqueados com base em seu *score*.
* Os **10 melhores** são selecionados como pais para a próxima geração.

### 3. Reprodução (Crossover)
* Descendentes são gerados por uma **média ponderada dos pesos** das redes neurais dos pais, onde a ponderação é proporcional ao *score* dos pais (ou seja, `α·P1 + β·P2`).

### 4. Mutação
* São aplicados dois tipos de ruído Gaussiano nos pesos dos descendentes:
    * **Small Gaussian:** Para pequenas correções nos pesos (exploração fina).
    * **Big Gaussian:** Para grandes alterações (exploração ampla e saltos em novos espaços de solução).

### 5. Imigração
* **5 novos agentes aleatórios** são adicionados a cada geração para manter o *pool* genético diversificado.

## Resultados Esperados

O comportamento dos agentes demonstra uma melhoria progressiva, fruto da sinergia entre o aprendizado intra-vida e a seleção inter-geracional:

| Fase | Características do Comportamento |
| :--- | :--- |
| **Gerações Iniciais** | Movimento caótico, mortes rápidas e alta frequência de colisões. Primeiros sinais de aprendizado: evitação de paredes. |
| **Gerações Intermediárias** | Navegação intencional. Agentes buscam a comida ativamente e demonstram redução de comportamentos cíclicos. Aumento do score médio. |
| **Gerações Avançadas** | Estratégias sofisticadas: Evitação de se prender em espaços pequenos e rotas otimizadas até a comida. Consistência em scores elevados. |

O sistema híbrido consolida um comportamento **robusto, adaptativo e altamente otimizado**, combinando a plasticidade do DQN com a eficiência de seleção do GA.

## Estrutura dos Arquivos

| Arquivo | Descrição |
| :--- | :--- |
| `cuda_agent.py` | Contém o loop principal, a lógica de multiprocessamento (`torch.multiprocessing`) e a orquestração dos agentes. |
| `model.py` | Define a Rede Neural (`Linear_QNet`) e a lógica de treinamento do DQN (`QTrainer`). |
| `snake_game.py` | Implementação do jogo clássico (utilizando Pygame). |
| `evo_syst.py` | Contém toda a lógica do Algoritmo Genético (Seleção, Crossover e Mutação/Imigração). |

## Como Executar

### Pré-requisitos
* Python 3.x
* PyTorch (com suporte CUDA é recomendado para melhor desempenho)
* Pygame

### Rodando o Treinamento

Execute o seguinte comando no terminal:

```bash
python cuda_agent.py
```
# Autores

* Eduardo Henrique Wenceslau Santana - N° USP: 15448734
* Pedro Henrique Vieira de Freitas - N° USP: 15652829
* Tiago Yuzo Yoshida Simão - N° USP: 15642202
* Vinicius de Oliveira Ribeiro - N° USP: 15498122
