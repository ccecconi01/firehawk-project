# Mapeamento de Estrutura para Projeto FireHawk AI

content = """# Mapeamento de Estrutura e Cobertura: Projeto FireHawk AI

Este documento apresenta o plano de transição do conteúdo existente para a nova estrutura de 9 secções exigida pelo **Guia de Orientação** [cite: 811-929]. O foco central é a mudança de paradigma da "previsão exata" para o "apoio à decisão através de tiers operacionais".

A análise da cobertura baseia-se na reestruturação exigida pelo Guia de Orientação, que foca a mudança do objetivo de "previsão exata de meios" para "classificação do nível expectável de resposta (apoio à decisão)".

Abaixo encontra-se o mapeamento detalhado do material proveniente do Relatório Provisório (RP), FireHawkReport (FHR) e PublishedArticle (PA) para as 9 secções da nova estrutura do documento final.

## Legenda de Decisões
- **(a) Reaproveitar 1:1**: Conteúdo factual, contextual ou técnico que permanece válido.
- **(b) Reescrita Ligeira**: Adaptação do texto para o novo foco de "apoio à decisão" e "níveis de resposta".
- **(c) Obsoleto/Descartar**: Conteúdo focado em regressão exata ou variáveis pós-evento (*leakage*).

---

## Tabela de Mapeamento: Secção x Fonte x Decisão

| Secção (Estrutura do Guia) | Fonte (Documentos Originais) | Decisão | Notas de Ajuste / Descarte |
| :--- | :--- | :---: | :--- |
| **1. Introdução e motivação** | PA (1. Introduction) [cite: 962-975]; FHR (1.1, 1.2, 2) [cite: 239-259, 273-324] | **(a), (b)** | **(a)** Dados de perdas florestais em 2024 e gastos estatais. **(b)** Reorientar o objetivo de "previsão de recursos" para "classificação do nível de resposta operacional". |
| **2. Enquadramento teórico** | RP (11. Porquê?) [cite: 202-207]; PA (1.3) [cite: 962-990]; FHR (2) [cite: 273-324] | **(a), (b)** | **(a)** Limitações dos dados operacionais e o papel da decisão humana. **(b)** Usar os conceitos de Orquestração/Coreografia para justificar a utilidade do apoio à decisão descentralizado. |
| **3. Estado da arte** | PA (2) [cite: 1027-1076]; FHR (1.3) [cite: 260-263] | **(a), (b)** | **(a)** Análise crítica dos sistemas FEBMON, SADO e projetos como FIRE-RES. **(b)** Focar na ausência de ferramentas de *dispatch prediction* integradas nestas plataformas. |
| **4. Descrição dos dados e auditoria** | RP (1, 3, 4) [cite: 3-7, 52-60, 81-83] | **(b), (c)** | **(b)** Organizar as features por grupo (temporais, meteo, etc.) conforme o Guia. **(c) Descartar** variáveis de leakage (`Area_Ardida`, `Duracao`) do modelo final preditivo. |
| **5. Metodologia** | RP (6.2, 5, 8) [cite: 123-130, 98-114, 144-162] | **(a), (b)** | **(a)** K-Means para definição dos Tiers (T0, T1, T2) e arquitetura *Two-Stage* para meios aéreos. **(b)** Justificar Random Forest como modelo central pela interpretabilidade. |
| **6. Resultados** | RP (4.2, 8, 9, 10) [cite: 84-97, 144-162, 173-187, 188-191] | **(b), (c)** | **(b)** Focar na accuracy dos Tiers (53.5%) e ranges (78.9% para veículos). **(c) Descartar** resultados de regressão contínua R²≈0 (usar apenas como baseline de falha). |
| **7. Discussão** | RP (11, 8) [cite: 192-201, 163-172] | **(a), (b)** | **(a)** Importância da feature `n_concurrent_fires`. **(b)** Discussão sobre o "bottleneck" ser a ausência de dados operacionais e não a má performance dos algoritmos. |
| **8. Protótipo e arquitetura** | FHR (3, 4) [cite: 325-364, 375-650, 656-782] | **(a), (b)** | **(a)** Diagramas UML e Style Guide. **(b)** Ajustar a interface para exibir "Nível de Resposta Estimado" em vez de números exatos. |
| **9. Conclusões e trabalho futuro** | RP (11) [cite: 192-212]; FHR (5) [cite: 783-798]; PA (4) [cite: 1092-1112] | **(a), (b)** | **(a)** Parcerias com Fogos.pt/VOST e uso de dados INE. **(b)** Confirmar que o sistema cumpre o papel de apoio à decisão tática apesar das limitações dos dados públicos. |

---
## Resumo da Estratégia de Decisão
A maior alteração exigida (Descarte de material obsoleto - c) encontra-se na "luta" que a equipa teve com a regressão contínua no Relatório Provisório. Todo esse percurso de erro-tentativa com variáveis viciadas (área ardida final) deve ser eliminado da narrativa principal. O documento final deverá ser linear: expõe o problema, descreve os dados limpos, justifica a criação de Tiers por clustering (K-Means), apresenta o Random Forest para classificação, e finalmente entrega os diagramas arquiteturais e UI do FireHawk atualizados para apresentar esse nível de alerta.


## Referências (Fontes Utilizadas)
1. **Relatório Provisório — Forest Fire Resource Forecast** (RP) [cite: 1-212]
2. **FireHawkReport — Informatics Engineering Project** (FHR) [cite: 213-810]
3. **Guia de orientação para o trabalho — Apoio à Decisão** [cite: 811-929]
4. **FireHawk: An Integrated and Adaptive Model... (Published Article)** (PA) [cite: 930-1145]
"""

with open("Mapeamento_Estrutura_FireHawk.md", "w", encoding="utf-8") as f:
    f.write(content)