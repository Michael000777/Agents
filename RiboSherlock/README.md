# 🧬 RiboSherlock: A Quality Control Agent for Bulk RNA-seq

**RiboSherlock** is a smart, modular, and extensible **LLM-assisted agent** that performs automated **quality control (QC)** on **bulk RNA-seq datasets**. Inspired by the deductive brilliance of Sherlock Holmes, this agent investigates sequencing results to identify issues, explain them in natural language, and suggest next steps for your RNA-seq experiments.

---

## 🔍 Project Overview

Bulk RNA-seq experiments often produce large, complex datasets that require careful QC interpretation. RiboSherlock helps by:

- Parsing outputs from tools like **FastQC**, **MultiQC**, or **featureCounts**
- Identifying quality issues (e.g., low read quality, adapter contamination, GC bias)
- Offering plain-language interpretations of QC results
- Recommending data cleaning steps when needed
- Looping in the user when decisions are ambiguous

---

## 🛠️ Features

- 🧠 **LLM-Powered Reasoning**: Uses a large language model to detect, interpret, and explain quality control patterns
- 📊 **Tool-Agnostic**: Works with standard output formats like FastQC, MultiQC, featureCounts, etc.
- ⚙️ **Modular LangGraph Workflow**: Uses a state machine to guide multi-step QC analysis
- 📝 **Natural-Language Reports**: Communicates findings in a way that’s accessible to both bioinformaticians and wet-lab collaborators
- 🔁 **Interactive Feedback Loop**: Requests user input in ambiguous cases and adjusts reasoning accordingly

---

## 🧱 Architecture

RiboSherlock is built using:

- **LangGraph** – Workflow engine for multi-step LLM agents
- **LangChain / LCEL** – LLM interaction layer
- **OpenAI / Groq / HuggingFace** – LLM providers
- **BeautifulSoup / Pandas** – For parsing and summarizing QC files
- **(Optional)** ChromaDB or FAISS – For document/embedding search across samples

### 📌 Core Workflow Nodes

| Node               | Description                                          |
|--------------------|------------------------------------------------------|
| `load_inputs`       | Reads and parses FastQC/MultiQC summaries           |
| `detect_issues`     | Classifies common QC issues                         |
| `generate_summary`  | Produces plain-language reports                     |
| `ask_for_feedback`  | Queries the user when results are inconclusive      |
| `report_writer`     | Writes summaries to disk or displays them in terminal |

---

## 🧰 Installation

### Requirements

- Python 3.8+
- Virtual environment recommended (`venv`, `pipenv`, etc.)
- API key for LLM provider (OpenAI, Groq, or local models)

### Install

```bash
git clone https://github.com/yourname/ribosherlock.git
cd ribosherlock
pip install -r requirements.txt
