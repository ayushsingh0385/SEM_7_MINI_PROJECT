"""One panel per content type: input widgets, then results."""
from __future__ import annotations

import io

import streamlit as st

from mmdetect.registry import ModalityInfo

from . import components as C
from .theme import step


# ---------------------------------------------------------------------------
# cached loaders - Streamlit reruns the whole script on every interaction, so
# models must be cached or they would reload on each click.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_text_detector():
    from mmdetect.text.infer import get_text_detector

    det = get_text_detector()
    det.load()
    return det


@st.cache_resource(show_spinner=False)
def _load_image_detector():
    from mmdetect.image.infer import get_image_detector

    det = get_image_detector()
    det.load()
    return det


@st.cache_data(show_spinner=False)
def _sample_texts(n: int = 6):
    """A few held-out test documents, for demoing without hunting for input."""
    try:
        from mmdetect.config import TEXT_TEST_PARQUET
        import pandas as pd

        if not TEXT_TEST_PARQUET.exists():
            return []
        df = pd.read_parquet(TEXT_TEST_PARQUET).sample(n, random_state=7)
        return [
            {"label": int(r.label), "text": str(r.text)[:4000]}
            for r in df.itertuples()
        ]
    except Exception:
        return []


def _model_missing(info: ModalityInfo) -> None:
    st.error(
        f"The {info.label.lower()} model has not been trained yet, so there is "
        "nothing to run.",
        icon=":material/error:",
    )
    script = "train_text.py" if info.key == "text" else "train_image.py"
    prep = "prepare_text_data.py" if info.key == "text" else "prepare_image_data.py"
    st.code(f"python scripts/{prep}\npython scripts/{script}", language="bash")


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
def text_panel(info: ModalityInfo) -> None:
    if not info.is_trained:
        _model_missing(info)
        return

    step(2, "Provide the text")

    tab_upload, tab_paste, tab_sample = st.tabs(
        ["Upload a file", "Paste text", "Try an example"]
    )

    content = ""
    source_name = "input.txt"

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a .txt, .md or .csv file",
            type=info.extensions,
            key="text_upload",
            help="The file is read in memory and never leaves this machine.",
        )
        if uploaded is not None:
            raw = uploaded.read()
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                content = raw.decode("latin-1", errors="replace")
                st.caption("File was not valid UTF-8; decoded as Latin-1.")
            source_name = uploaded.name
            st.success(f"Loaded **{uploaded.name}** ({len(content):,} characters)")
            with st.expander("Preview"):
                st.text(content[:1500] + ("..." if len(content) > 1500 else ""))

    with tab_paste:
        pasted = st.text_area(
            "Paste the text to analyse",
            height=230,
            placeholder="Paste an essay, article or answer here...",
            key="text_paste",
        )
        if pasted.strip():
            content = pasted
            source_name = "pasted.txt"

    with tab_sample:
        samples = _sample_texts()
        if not samples:
            st.caption("Examples need the prepared dataset. Run "
                       "`python scripts/prepare_text_data.py` first.")
        else:
            st.caption(
                "Held-out documents the model never saw during training. "
                "The true label is hidden until you analyse it."
            )
            idx = st.selectbox(
                "Pick an example",
                range(len(samples)),
                format_func=lambda i: f"Example {i + 1}  -  "
                                      f"{samples[i]['text'][:70].strip()}...",
                key="text_sample_idx",
            )
            if st.button("Use this example", key="use_text_sample"):
                st.session_state["_text_sample"] = samples[idx]
            chosen = st.session_state.get("_text_sample")
            if chosen:
                content = chosen["text"]
                source_name = "example.txt"
                st.info(f"Example loaded ({len(content):,} characters).")

    if not content.strip():
        st.caption("Provide some text above to enable analysis.")
        return

    step(3, "Analyse")
    if not st.button("Analyse text", type="primary", key="run_text"):
        return

    detector = _load_text_detector()
    with st.spinner("Scoring writing style..."):
        result = detector.predict(content)

    st.divider()
    C.verdict_card(result)
    C.probability_bar(result.ai_probability)
    C.score_row(result)

    if result.meta.get("too_short"):
        C.note(
            f"This text is only {result.meta['words']} words. Style statistics "
            f"get unreliable below ~{result.meta['min_reliable_words']} words, so "
            "this verdict is weaker than the number suggests."
        )

    st.markdown("#### What drove this verdict")
    C.text_evidence(result)

    with st.expander("How this model performs on held-out data"):
        C.model_metrics_panel(detector.training_metrics(), "text")

    C.download_result(result, f"{source_name}.result.json")


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
def image_panel(info: ModalityInfo) -> None:
    if not info.is_trained:
        _model_missing(info)
        return

    step(2, "Upload an image")
    uploaded = st.file_uploader(
        "Upload a PNG, JPG, BMP or WebP image",
        type=info.extensions,
        key="image_upload",
        help="The image is processed in memory and never leaves this machine.",
    )

    if uploaded is None:
        st.caption("Upload an image above to enable analysis.")
        return

    from PIL import Image

    data = uploaded.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        st.error(f"Could not read that image: {exc}", icon=":material/error:")
        return

    col_img, col_meta = st.columns([2, 1])
    with col_img:
        st.image(data, caption=uploaded.name, width="stretch")
    with col_meta:
        st.markdown("**File**")
        st.write(f"`{uploaded.name}`")
        st.write(f"{img.width} x {img.height} px")
        st.write(f"{len(data) / 1024:.0f} KB - {img.format or 'unknown'} format")

    step(3, "Analyse")
    if not st.button("Analyse image", type="primary", key="run_image"):
        return

    detector = _load_image_detector()
    with st.spinner("Scoring generator fingerprints..."):
        result = detector.predict(io.BytesIO(data))

    st.divider()
    C.verdict_card(result)
    C.probability_bar(result.ai_probability)
    C.score_row(result)

    if result.meta.get("far_from_training_resolution"):
        C.note(
            "This image is much larger than the 32x32 images the model was "
            "trained on. The upload is analysed at several scales to compensate, "
            "but accuracy on high-resolution photos is lower than the headline "
            "test-set number. See the limitations note below."
        )
    if result.meta.get("upscaled_from_small"):
        C.note("This image is smaller than 32x32 in one dimension, which is "
               "below what the model was built to handle.")

    st.markdown("#### What drove this verdict")
    C.image_evidence(result)

    with st.expander("How this model performs on held-out data"):
        C.model_metrics_panel(detector.training_metrics(), "image")
        st.caption(
            "Those numbers are measured on CIFAKE's own 32x32 test split. "
            "Real-world photos from other generators are a harder, different "
            "problem and the model is weaker there."
        )

    C.download_result(result, f"{uploaded.name}.result.json")


# ---------------------------------------------------------------------------
def render(info: ModalityInfo) -> None:
    """Dispatch to the right panel for the selected modality."""
    if not info.is_ready:
        C.planned_panel(info)
        return
    if info.key == "text":
        text_panel(info)
    elif info.key == "image":
        image_panel(info)
    else:  # pragma: no cover - registry and panels are kept in sync
        C.planned_panel(info)
