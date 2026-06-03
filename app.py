import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from agent.api_audit import load_api_audit
from agent.config import env_status
from agent.evaluator import load_metrics
from agent.predictor import get_model_status, predict_sequence
from agent.trainer import train_pipeline


logging.basicConfig(level=logging.INFO)

EXAMPLE_SEQUENCE = (
    "MMFTEDQGVDDRLLYDIVFKHFKRNKVEISNAIKKTFPFLEGLRDRDLITNKMFEDSQDSCRNLVPVQRVVYNVLSELEKTFNLPVLEALFSDVNMQEYPDLIHIYKGFENVIHDKLPLQ"
    "ESEEEEREERSGLQLSLEQGTGENSFRSLTWPPSGSPSHAGTTPPENGLSEHPCETEQINAKRKDTTSDKDDSLGSQQTNEQCAQKAEPTESCEQIAVQVNNGDAGREMPCPLPC"
    "DEESPEAELHNHGIQINSCSVRLVDIKKEKPFSNSKVECQAQARTHHNQASDIIVISSEDSEGSTDVDEPLEVFISAPRSEPVINNDNPLESNDEKEGQEATCSRPQIVPEPMDFRK"
    "LSTFRESFKKRVIGQDHDFSESSEEEAPAEASSGALRSKHGEKAPMTSRSTSTWRIPSRKRRFSSSDFSDLSNGEELQETCSSSLRRGSGKED"
)


def _disclaimer() -> None:
    st.warning(
        "Educational and portfolio demonstration only. This app is not a medical diagnostic tool, "
        "has not been clinically validated, and must not be used for clinical decisions."
    )


def _sidebar() -> None:
    status = get_model_status()
    env = env_status()
    st.sidebar.header("App Status")
    st.sidebar.write("Model:", "Available" if status["model_exists"] else "Not trained yet")
    st.sidebar.write("Best model:", status.get("best_model_name") or "Not available")
    st.sidebar.write("Last trained:", status.get("last_trained_timestamp") or "Not available")
    st.sidebar.write("Training data:", status.get("trained_from") or "Not available")
    st.sidebar.divider()
    st.sidebar.subheader("Environment")
    st.sidebar.write("NCBI Entrez:", "Configured" if env["ENTREZ_EMAIL"] else "Missing; skipped gracefully")
    st.sidebar.write(
        "Kaggle:",
        "Configured" if env["KAGGLE_USERNAME"] and env["KAGGLE_KEY"] else "Missing; skipped gracefully",
    )
    st.sidebar.caption("Secrets are never displayed by this app.")


def _read_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    return uploaded_file.read().decode("utf-8", errors="replace")


def _plot_probabilities(probabilities: dict) -> None:
    chart_df = pd.DataFrame(
        {
            "Class": ["Cancerous", "Non-Cancerous"],
            "Probability": [
                probabilities.get("cancerous", 0.0),
                probabilities.get("non_cancerous", 0.0),
            ],
        }
    )
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(chart_df["Class"], chart_df["Probability"], color=["#b73535", "#2d7d66"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Class probabilities")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.1%}", ha="center")
    st.pyplot(fig)


def _composition_table(composition: dict) -> pd.DataFrame:
    rows = [
        {"Amino Acid": key.replace("aa_", ""), "Fraction": value}
        for key, value in composition.items()
    ]
    return pd.DataFrame(rows).sort_values("Fraction", ascending=False)


def predict_tab() -> None:
    st.subheader("Predict Sequence")
    st.write("Paste a FASTA/plain protein sequence or upload a `.txt`, `.fasta`, or `.fa` file.")

    if "sequence_input" not in st.session_state:
        st.session_state.sequence_input = ""

    if st.button("Use example sequence"):
        st.session_state.sequence_input = EXAMPLE_SEQUENCE

    pasted = st.text_area(
        "Protein sequence",
        key="sequence_input",
        height=180,
        placeholder="Paste FASTA or plain amino acid sequence here...",
    )
    uploaded = st.file_uploader("Upload sequence file", type=["txt", "fasta", "fa"])
    uploaded_text = _read_uploaded_file(uploaded)
    sequence_text = uploaded_text or pasted

    if st.button("Predict", type="primary"):
        if not sequence_text.strip():
            st.error("Please paste or upload a protein sequence first.")
            return
        with st.spinner("Cleaning sequence, extracting features, and running prediction..."):
            result = predict_sequence(sequence_text)
        if not result.get("ok"):
            st.error(result.get("error", "Prediction failed."))
            return

        st.success(f"Prediction: {result['display_label']}")
        col1, col2 = st.columns(2)
        col1.metric("Confidence", f"{result['confidence']:.1%}")
        col2.metric("Sequence length", f"{result['sequence_length']} residues")

        _plot_probabilities(result["probabilities"])
        st.warning(result["medical_warning"])

        comp_df = _composition_table(result["amino_acid_composition"])
        st.write("Amino acid composition")
        st.dataframe(comp_df, width="stretch", hide_index=True)

        st.write("Top extracted features")
        feature_df = pd.DataFrame(
            [{"Feature": key, "Value": value} for key, value in result["top_features"].items()]
        )
        st.dataframe(feature_df, width="stretch", hide_index=True)

        frequent_kmers = result.get("frequent_kmers", {})
        if frequent_kmers:
            st.write("Frequent k-mer markers")
            rows = []
            for k_label, kmers in frequent_kmers.items():
                for kmer, count in kmers.items():
                    rows.append({"k-mer size": k_label.replace("k", ""), "k-mer": kmer, "Count": count})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _show_training_results(metrics: dict) -> None:
    st.success("Training completed and best model saved.")
    _show_metrics_report(metrics)


def _show_metrics_report(metrics: dict) -> None:
    if not metrics:
        st.info("No saved metrics found yet. Train the model first, then return here to check accuracy and other metrics.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total samples", metrics.get("total_samples", 0))
    col2.metric("Cancerous", metrics.get("cancerous_samples", 0))
    col3.metric("Non-cancerous", metrics.get("non_cancerous_samples", 0))

    col4, col5, col6, col7 = st.columns(4)
    col4.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    col5.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    col6.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    col7.metric("F1-score", f"{metrics.get('f1_score', 0):.3f}")

    st.write("Best model:", metrics.get("best_model_name", "Unknown"))
    st.write("Saved model path:", metrics.get("saved_model_path", "Unknown"))
    st.write("Last trained:", metrics.get("last_trained_timestamp", "Unknown"))
    st.write("Training data mode:", metrics.get("trained_from", "Unknown"))
    st.write("Feature count:", metrics.get("feature_count", "Unknown"))

    if metrics.get("data_quality_note"):
        st.info(metrics["data_quality_note"])
    if metrics.get("training_data_persistence_note"):
        st.info(metrics["training_data_persistence_note"])
    if metrics.get("api_audit_note"):
        st.info(metrics["api_audit_note"])

    if metrics.get("cumulative_training_enabled"):
        st.write("Cumulative training:", "Enabled")
        cumulative_cols = st.columns(5)
        cumulative_cols[0].metric("Previous samples", metrics.get("previous_training_samples", 0))
        cumulative_cols[1].metric("Fetched valid samples", metrics.get("new_training_samples_collected", 0))
        cumulative_cols[2].metric("New unique added", metrics.get("new_unique_training_samples_added", 0))
        cumulative_cols[3].metric("Cumulative samples", metrics.get("cumulative_training_samples", 0))
        cumulative_cols[4].metric("Duplicates skipped", metrics.get("duplicate_sequences_removed", 0))
        conflicts = metrics.get("conflicting_sequences_removed", 0)
        if conflicts:
            st.warning(f"{conflicts} sequences had conflicting inferred labels and were excluded from training.")

    st.write("Dataset source summary")
    source_summary = metrics.get("source_summary", {})
    st.dataframe(
        pd.DataFrame([{"Source": key, "Samples": value} for key, value in source_summary.items()]),
        width="stretch",
        hide_index=True,
    )

    api_cols = st.columns(4)
    api_cols[0].metric("API calls logged", metrics.get("api_calls_logged", 0))
    api_cols[1].metric("API success", metrics.get("api_successful_calls", 0))
    api_cols[2].metric("API failed", metrics.get("api_failed_calls", 0))
    api_cols[3].metric("API skipped", metrics.get("api_skipped_calls", 0))

    api_sources = metrics.get("api_calls_by_source", {})
    if api_sources:
        st.write("API calls by source")
        st.dataframe(
            pd.DataFrame([{"Source": key, "Calls": value} for key, value in api_sources.items()]),
            width="stretch",
            hide_index=True,
        )

    st.write("Confusion matrix")
    matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    labels = metrics.get("confusion_matrix_labels", ["cancerous", "non_cancerous"])
    st.dataframe(pd.DataFrame(matrix, index=labels, columns=labels), width="stretch")

    warnings = metrics.get("warnings", [])
    for warning in warnings:
        st.warning(warning)

    all_model_metrics = metrics.get("all_model_metrics", {})
    if all_model_metrics:
        st.write("Model comparison")
        rows = []
        for model_name, model_metrics in all_model_metrics.items():
            rows.append(
                {
                    "Model": model_name,
                    "Accuracy": model_metrics.get("accuracy", 0),
                    "Precision": model_metrics.get("precision", 0),
                    "Recall": model_metrics.get("recall", 0),
                    "F1-score": model_metrics.get("f1_score", 0),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    markers = metrics.get("kmer_marker_summary", [])
    if markers:
        st.write("Top enriched 3-mer markers")
        st.caption("These are exploratory sequence motifs from the current labeled training set, not validated biomarkers.")
        marker_df = pd.DataFrame(markers)
        st.dataframe(marker_df, width="stretch", hide_index=True)

    _show_api_audit_section(metrics)


def _show_api_audit_section(metrics=None) -> None:
    rows = []
    if metrics:
        rows = metrics.get("recent_api_calls", [])
    if not rows:
        rows = load_api_audit(limit=75)

    st.write("Recent API Calls")
    st.caption(
        "These requests are made by the Streamlit server, so they usually do not appear as UniProt/NCBI calls "
        "in your browser Network tab."
    )
    if not rows:
        st.info("No public API calls have been logged yet. Click Train to generate an API audit log.")
        return

    audit_df = pd.DataFrame(rows)
    preferred_columns = [
        "timestamp_utc",
        "stage",
        "source",
        "method",
        "status",
        "status_code",
        "result_count",
        "bytes_received",
        "duration_ms",
        "endpoint",
        "query",
        "message",
    ]
    visible_columns = [column for column in preferred_columns if column in audit_df.columns]
    st.dataframe(audit_df[visible_columns].tail(75), width="stretch", hide_index=True)


def train_tab() -> None:
    st.subheader("Train / Retrain Agent")
    st.write("Search public sequence sources, clean and label records, extract features, train models, and save the best model.")
    st.info(
        "Training first tries authentic public UniProt records and optional NCBI records. "
        "Each run merges newly collected valid sequences with the existing processed training set before retraining. "
        "If internet access or optional credentials are unavailable, it falls back to the synthetic demo dataset."
    )
    st.caption(
        "The committed data/processed/training_data.csv file is used as the free persistent baseline on every rebuild. "
        "Every Train click rotates through later public result pages, skips already-seen source IDs before download, "
        "and skips duplicate protein sequences after cleaning. "
        "On Render Free, new runtime training data can still reset unless the updated processed CSV is committed back to GitHub."
    )

    if st.button("Search Datasets and Train Model", type="primary"):
        progress = st.progress(0)
        status_box = st.empty()
        steps = [
            "Loading environment variables",
            "Searching datasets",
            "Downloading sequences",
            "Cleaning sequences",
            "Labeling data",
            "Extracting features",
            "Training models",
            "Evaluating models",
            "Saving best model",
        ]
        seen_steps = set()

        def callback(step: str, message: str) -> None:
            seen_steps.add(step)
            progress.progress(min(len(seen_steps) / len(steps), 1.0))
            status_box.info(f"{step}: {message}")

        try:
            metrics = train_pipeline(use_public_search=True, progress_callback=callback)
            progress.progress(1.0)
            _show_training_results(metrics)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Training failed: {exc}")
            st.info("The prediction tab can still auto-train from local demo data if no model is available.")


def metrics_tab() -> None:
    st.subheader("Check Accuracy / Metrics")
    st.write("View the currently saved model accuracy, precision, recall, F1-score, confusion matrix, and data source summary.")

    if st.button("Refresh Metrics", type="primary"):
        st.session_state["metrics_refreshed"] = True

    metrics = load_metrics()
    _show_metrics_report(metrics)


def main() -> None:
    st.set_page_config(page_title="Blood Cancer Protein Sequence AI Agent", page_icon="B", layout="wide")
    _sidebar()
    st.title("Blood Cancer Protein Sequence AI Agent")
    st.write(
        "An AI-powered research demo that classifies protein sequences as Cancerous or Non-Cancerous "
        "using sequence-derived features and scikit-learn models."
    )
    _disclaimer()

    predict, train, metrics = st.tabs(["Predict Sequence", "Train / Retrain Agent", "Check Accuracy / Metrics"])
    with predict:
        predict_tab()
    with train:
        train_tab()
    with metrics:
        metrics_tab()


if __name__ == "__main__":
    main()
