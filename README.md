# input_engineering

Pre-processing Pipeline (normalization, abbreviation expansion, GEC, coreference resolution,
sentence decomposition) that cleans process descriptions before a process model gets generated.

## Setup

1. Clone the repo and open a terminal in the project root.
2. Create and activate a virtual environment:
   ```
   python3 -m venv input-engineering-venv
   source input-engineering-venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
4. Install Java 17+ (needed by LanguageTool for GEC). Check with `java -version`.
5. For the local llama runs: start Ollama (`ollama serve`) with `llama3.1:8b` pulled.

## Run (from the project root, venv active)

### Experiment 1: Impact of Linguistic Noise

Gemini (AutoBPMN):
```
python run.py inject-pet
python run.py generate-injected
python run.py evaluate-injected
```

Llama (local, one run per description):
```
python run.py generate-mermaid-injected
python run.py convert-mermaid --dataset pet_injected
python run.py evaluate-injected --model llama --n-runs 1
```

### Experiment 2: Effect of Input Engineering

Gemini (AutoBPMN):
```
python run.py build-pipeline-injected
python run.py generate-pipeline-injected
python run.py evaluate-pipeline-injected
```

Llama (local, one run per description):
```
python run.py generate-mermaid-pipeline-injected
python run.py convert-mermaid --dataset deterministic_pipeline
python run.py evaluate-pipeline-injected --model llama --n-runs 1
```

### Complementary Experiment: LLM-Generated Noise (Gemini only)

The LLM-generated descriptions are in `descriptions/pet_injected/llm_combined`.
```
python run.py generate-injected --categories clean llm_combined
python run.py evaluate-injected
python run.py build-pipeline-injected
python run.py generate-pipeline-injected --categories llm_combined
python run.py evaluate-pipeline-injected
```

Results are written to `results/`, generated XML to `cpee-models/generated/`.
