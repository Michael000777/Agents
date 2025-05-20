# 🧬 RiboSherlock: A Quality Control Agent for Bulk RNA-seq

**RiboSherlock** is a smart, modular, and extensible **LLM-assisted agent** that performs automated **quality control (QC)** on **bulk RNA-seq datasets**. Inspired by the deductive brilliance of Sherlock Holmes, this agent investigates sequencing results to identify issues, explain them in natural language, and suggest next steps for your RNA-seq experiments.

---

## 🔍 Project Overview

Bulk RNA-seq experiments often produce large, complex datasets that require careful QC interpretation. RiboSherlock helps by:

- Parsing outputs from reports like **count tables** and **QC report files** 
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
- **OpenAI / Groq / HuggingFace** – compatiible with LLM provider of choice.

### 📌 Core Workflow Nodes

| Node               | Description                                          |
|--------------------|------------------------------------------------------|
| `Supervisor`       | Controls the workflow and ensures quality deliveries |
| `grader`           | Ensures request stay on topic                        |
| `enhancer`         | Rewrites user prompt to be compatible and complete   |
| `coder`            | State-of-the-art coder for analytical tasks          |
| `Validator`        | Validates results from other nodes to ensure quality |

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

### Usage
python scripts/main.py
