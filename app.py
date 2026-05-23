import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import tempfile
import fitz
from importlib.util import find_spec

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

st.set_page_config(page_title="MMJ Classifier", layout="wide")

st.title("Klasifikasi Meaningful, Mindful, Joyful")
st.caption("PDF/CSV -> Auto Label -> Highlight -> SVM -> fastText -> IndoBERT -> Macro F1")

LABEL_COLORS = {
    "Meaningful": "#DFF5E1",
    "Mindful": "#E1ECFF",
    "Joyful": "#FFF2CC",
    "Unknown": "#F2F2F2"
}

def is_module_available(module_name):
    return find_spec(module_name) is not None

def is_streamlit_cloud_environment():
    home_dir = os.path.expanduser("~")
    return (
        os.getenv("USER") == "appuser"
        or home_dir == "/home/appuser"
        or os.path.isdir("/home/appuser/.streamlit")
    )

def get_missing_indobert_dependencies():
    required_modules = ("torch", "datasets", "transformers", "accelerate")
    return [module for module in required_modules if not is_module_available(module)]

RUNNING_ON_STREAMLIT_CLOUD = is_streamlit_cloud_environment()

if RUNNING_ON_STREAMLIT_CLOUD:
    st.info(
        "Mode Streamlit Community Cloud aktif. Untuk menjaga app tetap ringan, "
        "fitur training berat seperti fastText dan IndoBERT disembunyikan. "
        "Training yang tersedia di Cloud hanya Linear SVM."
    )

def patch_fasttext_numpy_compat():
    try:
        numpy_major = int(np.__version__.split(".", 1)[0])
        if numpy_major < 2:
            return False

        import fasttext.FastText as fasttext_module
    except Exception:
        return False

    if getattr(fasttext_module._FastText.predict, "__name__", "") == "_predict_numpy_compat":
        return True

    def _predict_numpy_compat(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
        def check(entry):
            if entry.find("\n") != -1:
                raise ValueError("predict processes one line at a time (remove '\\n')")
            entry += "\n"
            return entry

        if type(text) == list:
            text = [check(entry) for entry in text]
            all_labels, all_probs = self.f.multilinePredict(
                text, k, threshold, on_unicode_error
            )
            return all_labels, all_probs

        text = check(text)
        predictions = self.f.predict(text, k, threshold, on_unicode_error)
        if predictions:
            probs, labels = zip(*predictions)
        else:
            probs, labels = ([], ())

        return labels, np.asarray(probs)

    fasttext_module._FastText.predict = _predict_numpy_compat
    return True

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z\u00C0-\u00FF0-9\s]", "", text)
    return text.strip()

def evaluate_model(y_true, y_pred, model_name):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

def generate_results_summary(results_df):
    if results_df.empty:
        return "Belum ada hasil evaluasi model yang bisa disimpulkan."

    sorted_df = results_df.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
    best_row = sorted_df.iloc[0]

    best_model = best_row["Model"]
    best_f1 = float(best_row["Macro F1"])
    best_accuracy = float(best_row["Accuracy"])
    best_precision = float(best_row["Macro Precision"])
    best_recall = float(best_row["Macro Recall"])

    summary_lines = [
        f"Model dengan performa terbaik adalah **{best_model}**.",
        (
            f"Nilainya terdiri dari **Macro F1 {best_f1:.3f}**, "
            f"**Accuracy {best_accuracy:.3f}**, "
            f"**Precision {best_precision:.3f}**, dan "
            f"**Recall {best_recall:.3f}**."
        ),
    ]

    if len(sorted_df) > 1:
        second_row = sorted_df.iloc[1]
        second_model = second_row["Model"]
        f1_gap = best_f1 - float(second_row["Macro F1"])

        if f1_gap >= 0.05:
            summary_lines.append(
                f"Dibandingkan **{second_model}**, selisih Macro F1 sebesar **{f1_gap:.3f}**, jadi model terbaik terlihat unggul cukup jelas."
            )
        else:
            summary_lines.append(
                f"Dibandingkan **{second_model}**, selisih Macro F1 sebesar **{f1_gap:.3f}**, jadi performa keduanya masih cukup berdekatan."
            )

    summary_lines.append(
        "Macro F1 dipakai sebagai acuan utama karena metrik ini membantu melihat keseimbangan performa model pada semua label, bukan hanya akurasi total."
    )

    return "\n\n".join(summary_lines)

def extract_pdf(uploaded_pdf):
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    rows = []
    progress = st.progress(0)
    total_pages = len(doc)

    for i, page in enumerate(doc):
        text = page.get_text("text")
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]

        for j, para in enumerate(paragraphs):
            rows.append({
                "page": i + 1,
                "paragraph_id": j + 1,
                "text": para
            })

        progress.progress((i + 1) / total_pages)

    doc.close()
    return pd.DataFrame(rows)

def get_keyword_map():
    return {
        "Meaningful": [
            "tujuan pembelajaran", "capaian pembelajaran", "kompetensi",
            "memahami", "menjelaskan", "mengidentifikasi", "menganalisis",
            "menerapkan", "menyimpulkan", "konsep", "materi", "pengetahuan",
            "keterampilan", "wawasan", "pengertian", "definisi",
            "fungsi", "manfaat", "tujuan", "prinsip", "prosedur",
            "langkah", "contoh", "penerapan", "implementasi",
            "peserta didik mampu", "setelah mempelajari",
            "diharapkan dapat", "membekali", "menguasai",
            "referensi", "sumber belajar", "pembelajaran",
            "pemahaman", "informasi", "isi buku", "kata kunci",
            "peta materi", "soft skill", "hard skill"
        ],

        "Mindful": [
            "refleksi", "berpikir kritis", "berpikir", "problem solving",
            "pemecahan masalah", "analisis", "evaluasi", "mengevaluasi",
            "membandingkan", "menentukan", "mengapa", "bagaimana",
            "apa alasan", "diskusikan", "berdiskusi", "pertanyaan",
            "tantangan", "strategi", "solusi", "menilai",
            "menghubungkan", "meninjau kembali", "mengukur kemampuan",
            "uji kompetensi", "menjawab pertanyaan", "berikan pendapat",
            "setuju atau tidak setuju", "identifikasi", "observasi",
            "menalar", "memecahkan", "menemukan masalah",
            "menyelesaikan masalah", "menentukan pilihan",
            "membuat keputusan", "mempertimbangkan", "meninjau",
            "kesimpulan", "kritik", "saran", "alasan"
        ],

        "Joyful": [
            "aktivitas", "aktivitas belajar", "kerja tim", "kelompok",
            "praktik", "praktik mandiri", "literasi mandiri",
            "presentasikan", "berbagi informasi", "bermain",
            "menyenangkan", "seru", "menarik", "eksplorasi",
            "kreativitas", "kolaborasi", "interaktif", "mencoba",
            "buatlah", "carilah", "lengkapi tabel", "diskusi kelompok",
            "teman", "pengalaman", "permainan", "gim",
            "menghibur", "motivasi", "rasa ingin tahu",
            "kegiatan", "proyek", "latihan", "simulasi",
            "mandiri", "tim", "bersama", "berkelompok",
            "menciptakan", "mengembangkan ide", "praktikkan",
            "ayo", "coba", "pindai aku", "pengayaan"
        ]
    }

def auto_label_rule(text):
    t = clean_text(text)
    keyword_map = get_keyword_map()
    scores = {}

    for label, keywords in keyword_map.items():
        score = 0
        for kw in keywords:
            if kw in t:
                score += 2 if len(kw.split()) > 1 else 1
        scores[label] = score

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score == 0:
        return "Unknown"

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_scores) > 1:
        top_label, top_score = sorted_scores[0]
        second_label, second_score = sorted_scores[1]

        if top_score == second_score:
            if any(k in t for k in ["refleksi", "uji kompetensi", "mengapa", "bagaimana", "analisis", "evaluasi"]):
                return "Mindful"
            if any(k in t for k in ["aktivitas", "kerja tim", "praktik", "presentasikan", "kelompok", "bermain"]):
                return "Joyful"
            if any(k in t for k in ["tujuan pembelajaran", "memahami", "konsep", "materi", "kompetensi"]):
                return "Meaningful"

    return best_label

def auto_label_with_score(text):
    t = clean_text(text)
    keyword_map = get_keyword_map()

    raw_scores = {}
    max_possible = {}

    for label, keywords in keyword_map.items():
        score = 0
        possible = 0

        for kw in keywords:
            weight = 2 if len(kw.split()) > 1 else 1
            possible += weight

            if kw in t:
                score += weight

        raw_scores[label] = score
        max_possible[label] = possible

    label = auto_label_rule(text)

    if label == "Unknown":
        return pd.Series(["Unknown", 0.0])

    top_score = raw_scores[label]
    total_score = sum(raw_scores.values())

    confidence = top_score / total_score if total_score > 0 else 0
    confidence = round(float(confidence), 3)

    return pd.Series([label, confidence])

def highlight_html(text, label, confidence=None):
    color = LABEL_COLORS.get(label, "#F2F2F2")
    conf_text = f" | Confidence: {confidence}" if confidence is not None else ""

    return f"""
    <div style="
        background:{color};
        padding:12px;
        border-radius:10px;
        margin-bottom:10px;
        border:1px solid #ddd;">
        <b>{label}</b>{conf_text}<br>
        {text}
    </div>
    """

def train_indobert(train_df, test_df):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from sklearn.preprocessing import LabelEncoder

    model_name = "indobenchmark/indobert-base-p1"
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    st.info(f"IndoBERT berjalan dengan: {device_name}")

    le = LabelEncoder()

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["label_id"] = le.fit_transform(train_df["label"])
    test_df["label_id"] = le.transform(test_df["label"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = Dataset.from_pandas(train_df[["clean_text", "label_id"]])
    test_ds = Dataset.from_pandas(test_df[["clean_text", "label_id"]])

    def tokenize(batch):
        return tokenizer(
            batch["clean_text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    train_ds = train_ds.rename_column("label_id", "labels")
    test_ds = test_ds.rename_column("label_id", "labels")

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(le.classes_)
    )

    args = TrainingArguments(
        output_dir="./indobert-mmj",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds
    )

    trainer.train()

    pred_output = trainer.predict(test_ds)
    pred_ids = np.argmax(pred_output.predictions, axis=1)
    preds = le.inverse_transform(pred_ids)

    return preds

mode = st.radio("Pilih sumber data:", ["Upload CSV", "Upload PDF Buku"])

df = None

if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

if mode == "Upload PDF Buku":
    uploaded_pdf = st.file_uploader("Upload PDF ratusan halaman", type=["pdf"])
    if uploaded_pdf:
        st.info("Membaca PDF per halaman...")
        df = extract_pdf(uploaded_pdf)
        st.success(f"Berhasil ekstrak {df.shape[0]} paragraf dari {df['page'].nunique()} halaman")

if df is not None:
    st.subheader("Preview Data")
    st.dataframe(df.head())

    if "text" not in df.columns:
        st.error("Data wajib memiliki kolom `text`.")
        st.stop()

    df = df.dropna(subset=["text"])
    df["clean_text"] = df["text"].apply(clean_text)

    st.subheader("Auto Labeling Rule-Based Awal")

    if st.button("Buat Label Otomatis"):
        df[["label", "confidence"]] = df["text"].apply(auto_label_with_score)
        st.session_state["df_labeled"] = df

    if "df_labeled" in st.session_state:
        df = st.session_state["df_labeled"]

    if "label" in df.columns:
        cols = ["text", "label"]
        if "confidence" in df.columns:
            cols.append("confidence")

        st.dataframe(df[cols].head(30))

        st.download_button(
            "Download Dataset Berlabel",
            df.to_csv(index=False).encode("utf-8"),
            "dataset_mmj_auto_label.csv",
            "text/csv"
        )

        st.subheader("Highlight Kalimat / Paragraf")

        label_filter = st.selectbox(
            "Filter label",
            ["Semua", "Meaningful", "Mindful", "Joyful", "Unknown"]
        )

        show_df = df if label_filter == "Semua" else df[df["label"] == label_filter]

        for _, row in show_df.head(50).iterrows():
            confidence = row["confidence"] if "confidence" in row else None
            st.markdown(
                highlight_html(row["text"], row["label"], confidence),
                unsafe_allow_html=True
            )

        df_trainable = df[df["label"].isin(["Meaningful", "Mindful", "Joyful"])].copy()

        if df_trainable["label"].nunique() < 2:
            st.warning("Minimal butuh 2 label valid untuk training.")
            st.stop()

        st.subheader("Distribusi Label")
        st.bar_chart(df_trainable["label"].value_counts())

        test_size = st.slider("Test size", 0.1, 0.4, 0.2)
        fasttext_available = is_module_available("fasttext")
        missing_indobert_dependencies = get_missing_indobert_dependencies()
        run_fasttext = False
        run_bert = False
        available_models = ["Linear SVM"]

        if RUNNING_ON_STREAMLIT_CLOUD:
            st.caption(
                "Mode Cloud aktif: opsi model berat disembunyikan untuk menjaga "
                "pemakaian resource tetap aman."
            )
        else:
            if fasttext_available:
                available_models.append("fastText")
                run_fasttext = st.checkbox("Jalankan fastText")
            else:
                st.caption(
                    "fastText dilewati di sesi ini karena dependency opsional "
                    "`fasttext-wheel` tidak dipasang."
                )

            if missing_indobert_dependencies:
                st.caption(
                    "IndoBERT dinonaktifkan di sesi ini karena dependency opsional "
                    "belum dipasang: " + ", ".join(missing_indobert_dependencies)
                )
            else:
                available_models.append("IndoBERT")
                run_bert = st.checkbox("Jalankan IndoBERT GPU-ready")

        st.caption("Model yang tersedia di sesi ini: " + ", ".join(available_models))

        X_train, X_test, y_train, y_test = train_test_split(
            df_trainable["clean_text"],
            df_trainable["label"],
            test_size=test_size,
            random_state=42,
            stratify=df_trainable["label"]
        )

        train_df = pd.DataFrame({"clean_text": X_train, "label": y_train})
        test_df = pd.DataFrame({"clean_text": X_test, "label": y_test})

        if st.button("Jalankan Training"):
            results = {}

            with st.spinner("Training Linear SVM..."):
                svm_model = Pipeline([
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
                    ("svm", LinearSVC())
                ])

                svm_model.fit(X_train, y_train)
                svm_pred = svm_model.predict(X_test)

                results["Linear SVM"] = {
                    "metrics": evaluate_model(y_test, svm_pred, "Linear SVM"),
                    "report": classification_report(y_test, svm_pred, zero_division=0)
                }

            if run_fasttext:
                try:
                    import fasttext

                    with st.spinner("Training fastText..."):
                        fasttext_patched = patch_fasttext_numpy_compat()

                        train_ft = pd.DataFrame({
                            "label": "__label__" + y_train.astype(str),
                            "text": X_train
                        })

                        train_ft["fasttext"] = train_ft["label"] + " " + train_ft["text"]

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                            temp_train_path = temp_file.name

                        train_ft["fasttext"].to_csv(temp_train_path, index=False, header=False)

                        try:
                            ft_model = fasttext.train_supervised(
                                input=temp_train_path,
                                epoch=25,
                                lr=0.5,
                                wordNgrams=2,
                                dim=100
                            )
                        finally:
                            try:
                                os.remove(temp_train_path)
                            except OSError:
                                pass

                        ft_preds = []

                        for t in X_test:
                            pred = ft_model.predict(t)[0][0]
                            ft_preds.append(pred.replace("__label__", ""))

                        results["fastText"] = {
                            "metrics": evaluate_model(y_test, ft_preds, "fastText"),
                            "report": classification_report(y_test, ft_preds, zero_division=0)
                        }

                        if fasttext_patched:
                            st.caption("fastText memakai patch kompatibilitas untuk NumPy 2.x.")

                except Exception as e:
                    st.warning(f"fastText gagal dijalankan: {e}")

            if run_bert:
                try:
                    with st.spinner("Training IndoBERT..."):
                        bert_preds = train_indobert(train_df, test_df)

                    results["IndoBERT"] = {
                        "metrics": evaluate_model(y_test, bert_preds, "IndoBERT"),
                        "report": classification_report(y_test, bert_preds, zero_division=0)
                    }

                except Exception as e:
                    st.warning(f"IndoBERT gagal dijalankan: {e}")

            results_df = pd.DataFrame([v["metrics"] for v in results.values()])
            results_df = results_df.sort_values(by="Macro F1", ascending=False)

            st.subheader("Perbandingan Model - Macro F1 Utama")
            st.dataframe(results_df)

            st.bar_chart(results_df.set_index("Model")["Macro F1"])

            st.subheader("Kesimpulan Hasil")
            st.info(generate_results_summary(results_df))

            for model_name, output in results.items():
                st.subheader(f"Classification Report - {model_name}")
                st.code(output["report"])

    else:
        st.info("Klik **Buat Label Otomatis** untuk membuat label awal.")
