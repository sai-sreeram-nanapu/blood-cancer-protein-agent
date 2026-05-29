# Blood Cancer Protein Sequence AI Agent

An educational AI-powered Streamlit web application that predicts whether a protein sequence is `Cancerous` or `Non-Cancerous` using sequence-derived features and scikit-learn models.

## Medical Disclaimer

This project is for education, research exploration, and portfolio demonstration only. It is not a medical diagnostic tool, has not been clinically validated, and must not be used for diagnosis, treatment planning, risk assessment, or any clinical decision.

## Features

- Paste FASTA or plain protein sequences.
- Upload `.txt`, `.fasta`, or `.fa` sequence files.
- Predict `Cancerous` or `Non-Cancerous` with confidence.
- Display class probabilities, sequence length, amino acid composition, and top extracted features.
- Check saved accuracy, precision, recall, F1-score, confusion matrix, and model comparison metrics from a dedicated tab.
- Search public metadata sources for protein sequence data.
- Download suitable FASTA, TXT, and CSV files when available.
- Clean, validate, deduplicate, and label sequence records.
- Merge newly collected public records with the existing processed training set before retraining.
- Train Logistic Regression, Random Forest, and SVC models.
- Select the best model by weighted F1-score.
- Save model and metrics locally.
- Fall back to synthetic demo data if public data, internet access, or optional credentials are unavailable.

## Architecture

```text
blood-cancer-protein-agent/
|-- app.py
|-- agent/
|   |-- __init__.py
|   |-- config.py
|   |-- dataset_search_agent.py
|   |-- dataset_downloader.py
|   |-- data_cleaner.py
|   |-- feature_extractor.py
|   |-- trainer.py
|   |-- evaluator.py
|   `-- predictor.py
|-- data/
|   |-- raw/
|   |   `-- .gitkeep
|   |-- processed/
|   |   |-- .gitkeep
|   |   `-- training_data.csv
|   `-- dataset_log.csv
|-- models/
|   `-- .gitkeep
|-- sample_data.csv
|-- requirements.txt
|-- README.md
|-- render.yaml
|-- .gitignore
`-- setup.sh
```

## How The Self-Training Agent Works

1. Loads environment variables with `python-dotenv`.
2. Searches targeted public UniProt records first and optional NCBI Protein records when Entrez is configured.
3. Rotates through later public result pages and skips source IDs already present in `data/processed/training_data.csv`.
4. Skips NCBI if `ENTREZ_EMAIL` is missing.
5. Skips Kaggle if `KAGGLE_USERNAME` or `KAGGLE_KEY` is missing.
6. Downloads only suitable small files and records every attempt in `data/dataset_log.csv`.
7. Parses FASTA/plain text, removes invalid sequences, deduplicates, and applies conservative labels from metadata.
8. Merges newly collected samples with existing processed samples, deduplicates identical sequences, and excludes sequences with conflicting inferred labels.
9. Extracts amino acid composition, dipeptide composition, sequence length, approximate molecular weight, and residue group ratios.
10. Trains Logistic Regression, Random Forest, and SVC.
11. Saves the best model to `models/best_model.joblib` and metrics to `models/metrics.json`.

## Dataset Sources

The dataset search agent can query:

- NCBI Protein through BioPython Entrez.
- UniProt REST API.
- Zenodo REST API.
- Kaggle API, only when credentials are configured.

Training uses a targeted public-data path first:

- Cancer-associated candidates: reviewed human UniProt records matching leukemia, lymphoma, myeloma, oncogene, tumor, or cancer terms, plus optional human NCBI records for leukemia/lymphoma protein queries.
- Non-cancer control candidates: reviewed human UniProt reference-proteome records filtered away from cancer, tumor, leukemia, lymphoma, myeloma, and oncogene terms.

These are authentic public protein sequences, but labels are still inferred from source query terms and metadata. Replace them with expert-curated biological labels before any serious research use.

## Environment Variables

Create a local `.env` file in the project root:

```bash
ENTREZ_EMAIL=your_email@example.com
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

All variables are optional for local demo use. Missing values are handled gracefully:

- Missing `ENTREZ_EMAIL`: NCBI is skipped.
- Missing Kaggle credentials: Kaggle is skipped.
- Missing internet or no usable public datasets: the app falls back to `sample_data.csv`.

Never commit `.env` or `kaggle.json`.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

You can also run:

```bash
bash setup.sh
```

## Train Or Retrain

From the Streamlit UI:

1. Open the `Train / Retrain Agent` tab.
2. Click `Search Datasets and Train Model`.
3. Review sample counts, source summary, model metrics, confusion matrix, and saved model path.
4. Open `Check Accuracy / Metrics` any time to review the last saved model metrics without retraining.

Each retraining run searches public sources again and trains a fresh model, but it trains on the cumulative processed dataset: previously saved samples plus newly collected valid samples from the current run. The fetcher rotates through later public result pages, skips source IDs already present in the tracked training CSV before download, skips duplicate protein sequences after cleaning, and reports how many fetched samples were actually added as new unique samples.

The repository tracks `data/processed/training_data.csv` as the free persistent baseline dataset. Every rebuild and deployment starts from that GitHub-tracked CSV and then optionally merges newly collected public data during the build.

On free Render deployments, runtime files are not permanent across rebuilds or service replacement. Cumulative training persists during the active deployment filesystem, but new data collected from the deployed app is only permanent after the updated processed CSV is committed back to GitHub.

From Python:

```bash
python -c "from agent.trainer import train_pipeline; print(train_pipeline(use_public_search=False))"
```

Use `use_public_search=True` to search and download from public sources.

## Prediction

If no model exists, the predictor automatically trains a local fallback model from available data. This keeps the demo working on a fresh deployment.

## Deploy On Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to Streamlit Community Cloud and create a new app from your repository.
3. Set the main file path to `app.py`.
4. Add secrets in Streamlit like this:

```toml
ENTREZ_EMAIL = "your_email@example.com"
KAGGLE_USERNAME = "your_kaggle_username"
KAGGLE_KEY = "your_kaggle_key"
```

5. Deploy the app. Streamlit will provide a public URL after the build succeeds.

The app works without these secrets, but NCBI and Kaggle are skipped when their values are missing.

## Deploy On Render

This repo includes `render.yaml`.

Render start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Steps:

1. Push the project to GitHub.
2. Create a new Render Blueprint from the repository or create a Python web service manually.
3. Confirm the build command is `pip install -r requirements.txt`.
4. For an authentic-data model during deployment, use the included Blueprint build command:

```bash
pip install -r requirements.txt && python -c "from agent.trainer import train_pipeline; train_pipeline(use_public_search=True)"
```

5. Confirm the start command is the Streamlit command above.
6. Add environment variables in the Render dashboard if you want NCBI/Kaggle access.
7. Deploy. Render will provide a public service URL after deployment.

## Security Notes

- Secrets are read only from environment variables or local `.env`.
- `.env` and `kaggle.json` are ignored by Git.
- The app never displays secret values.
- Kaggle is optional and skipped safely when credentials are absent.
- NCBI Entrez is optional and skipped safely when email is absent.

## Data Limitations

The included `sample_data.csv` is synthetic and exists only so the app can train and run as a portfolio demo. It is not biologically validated and must not be used for scientific or medical claims.

The included `data/processed/training_data.csv` is a public-source baseline collected from authentic public protein records. Labels are inferred from source queries and metadata, not expert clinical annotation.

Public search results can be noisy. Metadata-based labels are conservative, but they are not a substitute for curated expert labels.

## Replacing Demo Data With Validated Biological Data

For real research use, replace `sample_data.csv` or `data/processed/training_data.csv` with curated sequences:

```csv
sequence,label
MVALIDPROTEINSEQUENCE...,cancerous
MVALIDCONTROLSEQUENCE...,non_cancerous
```

Labels must be exactly:

- `cancerous`
- `non_cancerous`

Use experimentally validated sources, clear inclusion criteria, and independent train/test splits. Re-run training after replacing the data.

## Future Improvements

- Add curated benchmark datasets.
- Add protein language model embeddings.
- Add cross-validation and calibration curves.
- Add model explainability with SHAP or permutation importance.
- Add downloadable training reports.
- Add batch prediction for multiple FASTA records.
- Add CI tests and automated deployment checks.
