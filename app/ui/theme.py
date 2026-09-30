"""Styling for the Streamlit app.

Colours are chosen to stay legible in both Streamlit themes: backgrounds are
semi-transparent tints layered over whatever the theme provides, and text
colours are inherited rather than hard-coded, so nothing turns invisible when
the user switches between light and dark mode.
"""
from __future__ import annotations

import streamlit as st

# Verdict palette. Each entry is (accent, tint) where the tint is translucent.
VERDICT_COLORS = {
    "AI-generated": ("#e03131", "rgba(224, 49, 49, 0.12)"),
    "Human / Real": ("#2f9e44", "rgba(47, 158, 68, 0.12)"),
    "Uncertain":    ("#f08c00", "rgba(240, 140, 0, 0.14)"),
}

CSS = """
<style>
/* ---------- layout ---------- */
.block-container { padding-top: 2.2rem; max-width: 1100px; }

/* ---------- hero ---------- */
.mm-hero {
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(99,102,241,0.16), rgba(168,85,247,0.10));
    border: 1px solid rgba(128,128,128,0.22);
    margin-bottom: 1.4rem;
}
.mm-hero h1 { margin: 0 0 .3rem 0; font-size: 1.9rem; line-height: 1.2; }
.mm-hero p  { margin: 0; opacity: .8; font-size: .97rem; }

/* ---------- step headings ---------- */
.mm-step {
    display: flex; align-items: center; gap: .55rem;
    font-weight: 600; font-size: 1.05rem;
    margin: 1.3rem 0 .6rem 0;
}
.mm-step .n {
    display: inline-flex; align-items: center; justify-content: center;
    width: 1.6rem; height: 1.6rem; border-radius: 50%;
    background: rgba(99,102,241,.22);
    font-size: .85rem; font-weight: 700; flex: 0 0 auto;
}

/* ---------- verdict card ---------- */
.mm-verdict {
    border-radius: 14px; padding: 1.3rem 1.5rem; margin: .4rem 0 1rem 0;
    border: 1px solid var(--mm-accent); background: var(--mm-tint);
}
.mm-verdict .tag {
    font-size: .74rem; letter-spacing: .1em; text-transform: uppercase;
    opacity: .75; font-weight: 600;
}
.mm-verdict .headline {
    font-size: 1.85rem; font-weight: 700; margin: .25rem 0 .1rem 0;
    color: var(--mm-accent);
}
.mm-verdict .sub { font-size: .93rem; opacity: .82; }

/* ---------- probability bar ---------- */
.mm-bar-wrap { margin: .9rem 0 .2rem 0; }
.mm-bar {
    position: relative; height: 26px; border-radius: 13px; overflow: hidden;
    background: rgba(47,158,68,.20);
}
.mm-bar-fill {
    height: 100%; border-radius: 13px 0 0 13px;
    background: linear-gradient(90deg, rgba(224,49,49,.55), rgba(224,49,49,.9));
}
.mm-bar-mid {
    position: absolute; top: 0; bottom: 0; left: 50%; width: 2px;
    background: rgba(128,128,128,.85);
}
.mm-bar-legend {
    display: flex; justify-content: space-between;
    font-size: .78rem; opacity: .75; margin-top: .3rem;
}

/* ---------- misc ---------- */
.mm-note {
    border-left: 3px solid rgba(240,140,0,.75);
    background: rgba(240,140,0,.09);
    padding: .65rem .9rem; border-radius: 0 8px 8px 0;
    font-size: .88rem; margin: .6rem 0;
}
.mm-planned {
    border: 1px dashed rgba(128,128,128,.45); border-radius: 12px;
    padding: 1.6rem; text-align: center; opacity: .92;
}
.mm-planned .big { font-size: 2.6rem; margin-bottom: .4rem; }
.mm-chip {
    display: inline-block; padding: .16rem .6rem; border-radius: 999px;
    font-size: .74rem; font-weight: 600; margin-left: .4rem;
    border: 1px solid rgba(128,128,128,.35);
}
.mm-chip.ok   { background: rgba(47,158,68,.16);  border-color: rgba(47,158,68,.5); }
.mm-chip.todo { background: rgba(128,128,128,.12); }

/* signal rows in the evidence panel */
.mm-signal {
    display: flex; justify-content: space-between; align-items: center;
    padding: .35rem .1rem; border-bottom: 1px solid rgba(128,128,128,.16);
    font-size: .88rem;
}
.mm-signal:last-child { border-bottom: none; }
.mm-signal .to-ai    { color: #e03131; font-weight: 600; }
.mm-signal .to-human { color: #2f9e44; font-weight: 600; }
</style>
"""


def inject() -> None:
    """Add the stylesheet to the page (call once, right after set_page_config)."""
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="mm-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def step(number: int, text: str) -> None:
    st.markdown(
        f'<div class="mm-step"><span class="n">{number}</span>{text}</div>',
        unsafe_allow_html=True,
    )
