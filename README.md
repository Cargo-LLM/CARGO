# CARGO: A Framework for Confidence-Aware Routing of Large Language Models

This work proposes an ensemble-based selection mechanism that dynamically evaluates responses from multiple LLMs to identify the most suitable output for a given prompt. A ranking model trained on a diverse dataset enables adaptive model selection, improving response accuracy and relevance. We benchmark the ensemble strategy against individual models across multiple task categories to assess its effectiveness.

---

### 1. Prompt Response Collection

- **Folder**: `Prompt Response Collection`
  - **Script**: `response_collection.py`
    - Used to run a set of prompts on all LLMs via OpenRouter.
    - Collects their responses for further processing and analysis.
    - **Important**: Add your **OpenRouter** API key and specify the prompt dataset before running.

### 2. Full Dataset Ranking

- **Folder**: `Full Dataset Ranking`
  - **Dataset**: `Full_dataset_with_id.csv`
    - Contains the raw responses from all models **before** ranking.
  - **Script**: `rank_script_V3.py`
    - Ranks all the responses using the method described in the paper.
    - Generates the final ranked output in `ranked_responses_final.csv`
    - **Important**: Requires your **OpenRouter** API key as well.

### 3. Training

- **Folder**: `Training`
  - Contains training scripts and the trained classifier.
