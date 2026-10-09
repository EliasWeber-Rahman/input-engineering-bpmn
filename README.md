# input_engineering

Pre-processing Pipeline (normalization, abbreviation expansion, GEC, coreference resolution,
sentence decomposition) that cleans process descriptions before a process model gets generated.

## Setup

1. Clone the repo and open a terminal in the project root.
2. Create and activate a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
4. Install Java 17+ (needed by LanguageTool for GEC). Check with `java -version`.
5. For the local llama runs: start Ollama (`ollama serve`) with `llama3.1:8b` pulled.

## Run (from the project root, venv active)

Baseline vs. pipeline configs:
```
python run.py generate
python run.py evaluate
```

RQ2 (noised PET descriptions):
```
python run.py inject-pet
python run.py generate-injected
python run.py evaluate-injected
```

RQ3 (pipeline on noised PET):
```
python run.py build-pipeline-injected
python run.py generate-pipeline-injected
python run.py evaluate-pipeline-injected
```

Local llama (instead of AutoBPMN): run `generate-mermaid-local`,
`generate-mermaid-injected` or `generate-mermaid-pipeline-injected`, then
`convert-mermaid`, then the matching evaluate command with `--model llama`.

Results are written to `results/`, generated XML to `cpee-models/generated/`.
