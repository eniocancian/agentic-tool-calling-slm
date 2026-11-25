# 🤖 Agentic Tool-Calling SLM Router (`QLoRA` + `PEFT` + `Pydantic`)

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-PEFT_%2F_TRL-yellow)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Ajuste Fino Supervisionado (QLoRA 4-bit)** de um modelo compacto de linguagem (**SLM — Small Language Model**) de 3 bilhões de parâmetros (`Qwen/Qwen2.5-3B-Instruct`), especializado em **Function Calling / Tool Routing determinístico** para fluxos agênticos em produção.

---

## 🎯 Problema de Engenharia & Motivação

Em sistemas agênticos reais voltados a domínios críticos (telemetria industrial, engenharia térmica e risco financeiro), o maior gargalo raramente é apenas recuperar documentos via RAG — mas garantir que o modelo **selecione a ferramenta correta, extraia argumentos tipados e obedeça rigorosamente ao contrato JSON da API** sem quebrar o parser da aplicação.

Modelos compactos (`3B`) em modo *Zero-Shot* frequentemente falham em produção ao misturar prosa conversacional com JSON inválido. Este projeto demonstra como transformar um SLM local e de baixo custo em um roteador determinístico com **100% de conformidade estrutural** usando **QLoRA + Pydantic v2**.

---

## 📊 Resultados Principais de Benchmark (150 Amostras Estratificadas)

Avaliação comparativa rigorosa entre o modelo base (*Zero-Shot*) e os adaptadores LoRA treinados sobre o conjunto de validação estratificado (chamada única, chamada paralela, pedido de esclarecimento e recusa fora de escopo):

| Variante do Modelo | Validade Sintática JSON | Conformidade Schema Pydantic | Acurácia na Seleção de Ferramenta | Ganho Absoluto (Schema) |
| :--- | :---: | :---: | :---: | :---: |
| `Qwen/Qwen2.5-3B-Instruct` (Zero-Shot Base) | `2,0%` | `0,0%` | `0,0%` | — |
| `Qwen/Qwen2.5-3B-Instruct` + **QLoRA (`r=16, α=32`)** | **`100,0%`** | **`100,0%`** | **`100,0%`** | **`+100,0 pp`** |

### 📈 Dinâmica de Convergência e Eficiência Computacional
* **Hardware de Treinamento:** NVIDIA L4 (24 GB VRAM, precisão nativa `bfloat16`).
* **Eficiência Paramétrica:** Apenas **`29.933.568` parâmetros treináveis** (**`0,96%`** dos `3,12 bilhões` de parâmetros totais do modelo).
* **Generalização sem Overfitting:** Ao longo de **376 passos de otimização** (2 épocas / `1,56 milhão` de tokens processados), a *Validation Loss* caiu continuamente de `0,0889` (Step 100) para **`0,0772`** (Step 376), acompanhando a *Training Loss* (`0,0648`) e estabilizando a *Mean Token Accuracy* em **`97,33%`**.

![Curvas de Treinamento QLoRA](results/training_curves.png)

---

## 🧠 Arquitetura & Metodologia Aplicada

### 1. Engenharia e Curadoria de Dataset (`data/prepare_dataset.py`)
* **Composição Estratificada:** `3.000` exemplos de treino e `400` exemplos de validação combinando o dataset `glaiveai/glaive-function-calling-v2` com casos sintéticos de borda de domínio industrial e financeiro.
* **Cobertura de Comportamentos Agênticos:**
  * `call_tool` (única e paralela): Extração de múltiplos parâmetros tipados e decomposição de consultas compostas.
  * `clarify`: Solicitação de parâmetros obrigatórios ausentes antes de acionar APIs externas.
  * `no_tool`: Recusa segura e resposta direta para perguntas conceituais fora do catálogo de ferramentas.

### 2. Fine-Tuning QLoRA 4-bit (`notebooks/train_qlora_colab.py`)
* **Quantização:** 4-bit NormalFloat (`nf4`) com dupla quantização (`bnb_4bit_use_double_quant=True`) via `bitsandbytes`.
* **Configuração LoRA (`PEFT`):** Rank `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, aplicado em todas as projeções lineares de atenção e MLP (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Otimização (`TRL SFTTrainer`):** `paged_adamw_8bit`, scheduler `cosine`, `learning_rate=2e-4`, batch efetivo de `16` exemplos (`batch_size=4` × `gradient_accumulation_steps=4`).

### 3. Validação Determinística e Inferência (`src/`)
* **Contrato Estrito (`data/custom_tools_schema.py`):** Validação via **Pydantic v2** (`AgentDecision`, `ToolCall`) garantindo tipagem forte antes de qualquer execução externa.
* **Benchmark Automatizado (`src/evaluate.py`):** Pipeline reprodutível que mede sintaxe JSON, validação de schema Pydantic e exatidão do nome da ferramenta invocada.
* **Microsserviço de Inferência (`src/api_server.py`):** API REST em **FastAPI** pronta para servir o roteador treinado a orquestradores agênticos (LangGraph, CrewAI, etc.).

---

## 🔍 Análise Crítica de Erros Residuais (Stress-Test ao Vivo)

Além do benchmark quantitativo, o modelo foi submetido a testes de estresse com prompts inéditos em tempo real:
* ✅ **Decomposição Paralela (Aprovado):** Prompts compostos como *"Check the temperature and vibration of heat exchanger HX-402 over the last 45 minutes"* foram corretamente desmembrados em duas chamadas paralelas simultâneas a `get_equipment_telemetry`.
* ✅ **Roteamento Fora de Escopo (Aprovado):** Perguntas teóricas de engenharia térmica (*"Explain the difference between LMTD and NTU methods"*) foram roteadas para `"action": "no_tool"` sem alucinação de ferramentas.
* ⚠️ **Alucinação de Parâmetros Ausentes (Limitação Residual Mapeada):** Devido ao desbalanceamento natural da base (~91% de chamadas diretas vs. ~3% de exemplos `clarify`), diante de prompts subespecificados (*"Calculate the VaR exposure for my stock portfolio"*), o modelo tende a preencher valores padrão plausíveis (`["AAPL", "GOOGL", "MSFT"]`, `0.95`) em vez de pedir confirmação. Esse comportamento mapeia diretamente o próximo passo evolutivo do projeto: alinhamento de preferências via **DPO (Direct Preference Optimization)**.

---

## 📁 Estrutura do Repositório

* `data/custom_tools_schema.py` — Catálogo de ferramentas e contratos Pydantic v2 (`AgentDecision`)
* `data/prepare_dataset.py` — Curadoria estratificada (`Glaive v2` + casos sintéticos de borda)
* `notebooks/train_qlora_colab.py` — Pipeline completo de Fine-Tuning QLoRA 4-bit (`PEFT` + `TRL`)
* `results/benchmark_report.md` — Relatório técnico completo com logs e análise de erros
* `results/training_curves.png` — Curvas de Loss (Treino/Validação) e Token Accuracy
* `src/api_server.py` — Microsserviço FastAPI para inferência estruturada
* `src/evaluate.py` — Benchmark comparativo automatizado (Base vs. Fine-Tuned)
* `LICENSE` — Licença MIT
* `requirements.txt` — Dependências do projeto

---

## 🚀 Como Executar Localmente

1. **Clonar o repositório e instalar dependências:** `git clone https://github.com/eniocancian/agentic-tool-calling-slm.git && cd agentic-tool-calling-slm && pip install -r requirements.txt`
2. **Gerar o dataset estratificado (`train.jsonl` e `eval.jsonl`):** `python data/prepare_dataset.py`
3. **Executar o benchmark comparativo (Zero-Shot vs. Adaptadores LoRA):** `python src/evaluate.py`
4. **Subir o microsserviço de inferência FastAPI:** `uvicorn src.api_server:app --host 0.0.0.0 --port 8000`

---

## 👤 Autor

**Ênio Fernando Cancian Júnior** — Engenheiro Químico | Pós-graduando em Data Science, Analytics & IA (USP/Esalq)  
* 🔗 **GitHub:** [@eniocancian](https://github.com/eniocancian)
* 💼 **LinkedIn:** [Ênio Fernando Cancian Júnior](https://www.linkedin.com/in/eniocancian/)
