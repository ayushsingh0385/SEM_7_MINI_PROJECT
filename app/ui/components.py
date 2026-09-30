"""Reusable result widgets shared by every modality panel."""
from __future__ import annotations

from typing import Any

import streamlit as st

from .theme import VERDICT_COLORS


def verdict_card(result) -> None:
    """The headline verdict, colour-coded by outcome."""
    accent, tint = VERDICT_COLORS.get(result.verdict, ("#868e96", "rgba(134,142,150,.12)"))
    pct = result.ai_probability * 100

    if result.verdict == "Uncertain":
        sub = (
            "The model cannot separate this one confidently. Treat it as "
            "inconclusive rather than as evidence either way."
        )
    elif result.is_ai:
        sub = f"The model attributes {pct:.1f}% probability to this being AI-generated."
    else:
        sub = f"The model attributes {100 - pct:.1f}% probability to this being human-made."

    st.markdown(
        f"""
        <div class="mm-verdict" style="--mm-accent:{accent}; --mm-tint:{tint};">
            <div class="tag">Verdict</div>
            <div class="headline">{result.verdict}</div>
            <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_bar(ai_probability: float) -> None:
    """A single bar running from 'human' on the left to 'AI' on the right."""
    pct = max(0.0, min(1.0, ai_probability)) * 100
    st.markdown(
        f"""
        <div class="mm-bar-wrap">
            <div class="mm-bar">
                <div class="mm-bar-fill" style="width:{pct:.2f}%;"></div>
                <div class="mm-bar-mid"></div>
            </div>
            <div class="mm-bar-legend">
                <span>Human / Real &nbsp;{100 - pct:.1f}%</span>
                <span>decision threshold</span>
                <span>{pct:.1f}%&nbsp; AI-generated</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_row(result) -> None:
    """Three headline numbers under the verdict."""
    c1, c2, c3 = st.columns(3)
    c1.metric("AI probability", f"{result.ai_probability * 100:.1f}%")
    c2.metric("Human probability", f"{result.human_probability * 100:.1f}%")
    c3.metric(
        "Confidence",
        f"{result.confidence * 100:.0f}%",
        help="How far the score sits from an even 50/50 split.",
    )


def note(text: str) -> None:
    st.markdown(f'<div class="mm-note">{text}</div>', unsafe_allow_html=True)


def text_evidence(result) -> None:
    """Which writing-style features pushed the verdict, and in which direction."""
    signals = result.explanation.get("signals")
    if not signals:
        st.caption("No per-feature breakdown available for this model.")
        return

    st.markdown(
        "Each row is a writing-style measurement and the direction it pushed "
        "the decision. These come from the model's own linear weights, not a "
        "separate estimate."
    )
    rows = []
    for s in signals:
        cls = "to-ai" if s["points_to"] == "AI" else "to-human"
        arrow = "&#9650;" if s["points_to"] == "AI" else "&#9660;"
        rows.append(
            f'<div class="mm-signal"><span>{s["label"]}'
            f'<span style="opacity:.6"> &nbsp;=&nbsp;{s["value"]:.3g}</span></span>'
            f'<span class="{cls}">{arrow} {s["points_to"]}</span></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)

    raw = result.explanation.get("raw_features", {})
    if raw:
        with st.expander(f"All {len(raw)} stylometric measurements"):
            st.dataframe(
                {"feature": list(raw.keys()),
                 "value": [round(v, 4) for v in raw.values()]},
                width="stretch", hide_index=True,
            )


def image_evidence(result) -> None:
    """How the multiple views of the image voted."""
    exp = result.explanation
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Whole-image view",
        f"{exp['global_probability'] * 100:.1f}%",
        help="The full image resized to 32x32, the closest match to how the "
             "model was trained.",
    )
    if exp.get("tile_probability") is not None:
        c2.metric(
            f"Tile view ({exp['n_tiles']} tiles)",
            f"{exp['tile_probability'] * 100:.1f}%",
            help="The image cut into 32x32 tiles and averaged.",
        )
        c3.metric(
            "View agreement",
            "Agree" if exp["views_agree"] else "Split",
            help="Whether the tiles produced similar scores. A split vote means "
                 "the result is shaky.",
        )
    else:
        c2.metric("Tile view", "n/a", help="Image too small to tile.")
        c3.metric("View agreement", "n/a")

    probs = exp.get("tile_probabilities") or []
    if len(probs) > 1:
        st.caption("Per-tile AI probability")
        st.bar_chart(
            {"AI probability": probs},
            height=170, width="stretch",
        )

    if not exp.get("views_agree", True):
        note(
            "The tiles disagreed with each other. Different parts of this image "
            "look different to the model, so treat the overall score with caution."
        )


def model_metrics_panel(metrics: dict[str, Any] | None, modality: str) -> None:
    """Held-out performance recorded when the model was trained."""
    if not metrics:
        st.caption("No saved evaluation metrics found for this model.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c2.metric("F1 score", f"{metrics['f1']:.4f}")
    c3.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    c4.metric(
        "False-positive rate",
        f"{metrics['false_positive_rate'] * 100:.2f}%",
        help="How often genuine human content is wrongly flagged as AI. "
             "This is the error that matters most in practice.",
    )
    st.caption(
        f"Measured on {metrics.get('n_test', 0):,} held-out samples the model "
        "never saw during training."
    )

    cm = metrics.get("confusion_matrix")
    if cm:
        with st.expander("Confusion matrix"):
            st.dataframe(
                {
                    "": ["Actually human/real", "Actually AI"],
                    "Predicted human/real": [cm["true_human"]["pred_human"],
                                             cm["true_ai"]["pred_human"]],
                    "Predicted AI": [cm["true_human"]["pred_ai"],
                                     cm["true_ai"]["pred_ai"]],
                },
                width="stretch", hide_index=True,
            )


def planned_panel(info) -> None:
    """Placeholder shown for modalities that are not built yet."""
    st.markdown(
        f"""
        <div class="mm-planned">
            <div class="big">{info.icon}</div>
            <h3 style="margin:.2rem 0;">{info.label} detection arrives in week {info.week}</h3>
            <p style="opacity:.8; margin:.4rem 0 0 0;">{info.description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if info.planned_approach:
        st.markdown(f"**Planned approach** &mdash; {info.planned_approach}")
    st.info(
        f"Weeks 1-3 are complete: the Streamlit interface, the text detector and "
        f"the image detector. {info.label} support is next on the plan and will "
        f"appear here behind the same upload flow, with no changes to the rest "
        f"of the app.",
        icon=":material/schedule:",
    )


def download_result(result, filename: str) -> None:
    """Offer the full result as JSON, handy for the project report."""
    import json

    st.download_button(
        "Download this result as JSON",
        data=json.dumps(result.as_dict(), indent=2),
        file_name=filename,
        mime="application/json",
    )
