"""
ModerationKit — Local Nudity Detector

Modern Gradio UI for the two-model moderation stack:
  - NudeNet 640m  : body-part detector with bounding boxes
  - Falconsai     : whole-image NSFW classifier

Run locally:        python app.py
Public share link:  python app.py --share
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from analyzer import CombinedAnalyzer, VerdictPolicy
from detector import (
    EXPLICIT_CLASSES,
    NEUTRAL_CLASSES,
    SUGGESTIVE_CLASSES,
    Detector,
    annotate,
)
from nsfw_classifier import NSFWClassifier


# Brand assets — read once at module load so we can inline-embed in the UI
_LOGO_DIR = Path(__file__).resolve().parent / "Logo"
LOGO_ICON_PATH = _LOGO_DIR / "icon.svg"
LOGO_FULL_PATH = _LOGO_DIR / "ModerationKit_Shield_Logo.svg"
try:
    LOGO_ICON_SVG = LOGO_ICON_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    LOGO_ICON_SVG = ""  # fall back gracefully if logo missing


ALL_CLASSES = sorted(EXPLICIT_CLASSES | SUGGESTIVE_CLASSES | NEUTRAL_CLASSES)
DEFAULT_CLASSES = sorted(EXPLICIT_CLASSES | SUGGESTIVE_CLASSES)

# Production-locked defaults (matches server.py)
DEFAULT_RESOLUTION = 1024
DEFAULT_THRESHOLD = 0.15

VERDICT_META = {
    "ALLOW":  {"color1": "#10b981", "color2": "#059669", "icon": "✓",  "label": "ALLOW"},
    "REVIEW": {"color1": "#f59e0b", "color2": "#d97706", "icon": "!",  "label": "REVIEW"},
    "BLOCK":  {"color1": "#ef4444", "color2": "#dc2626", "icon": "✕",  "label": "BLOCK"},
}

_analyzer_cache: dict[int, CombinedAnalyzer] = {}
_shared_classifier: NSFWClassifier | None = None


def _get_classifier() -> NSFWClassifier:
    global _shared_classifier
    if _shared_classifier is None:
        _shared_classifier = NSFWClassifier()
    return _shared_classifier


def get_analyzer(resolution: int) -> CombinedAnalyzer:
    if resolution not in _analyzer_cache:
        det = Detector(
            inference_resolution=resolution,
            score_threshold=0.05,
            nms_threshold=0.05,
        )
        _analyzer_cache[resolution] = CombinedAnalyzer(
            detector=det,
            classifier=_get_classifier(),
        )
    return _analyzer_cache[resolution]


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------
def _verdict_html(verdict: str, reason: str, elapsed_ms: int) -> str:
    meta = VERDICT_META.get(verdict, VERDICT_META["ALLOW"])
    return f"""
<div class="mk-verdict-banner" style="background:linear-gradient(135deg,{meta['color1']},{meta['color2']})">
  <div class="mk-verdict-icon">{meta['icon']}</div>
  <div class="mk-verdict-text">
    <div class="mk-verdict-label">{meta['label']}</div>
    <div class="mk-verdict-reason">{reason}</div>
  </div>
  <div class="mk-verdict-time">{elapsed_ms}ms</div>
</div>
"""


def _gauge_bar(value: float, label: str, color: str) -> str:
    """Horizontal progress bar with percentage label."""
    pct = max(0.0, min(1.0, value)) * 100
    return f"""
<div class="mk-gauge">
  <div class="mk-gauge-header">
    <span class="mk-gauge-label">{label}</span>
    <span class="mk-gauge-value">{pct:.1f}%</span>
  </div>
  <div class="mk-gauge-track">
    <div class="mk-gauge-fill" style="width:{pct}%;background:{color}"></div>
  </div>
</div>
"""


def _stats_html(nsfw_score, top_explicit, n_dets: int, kept_after_filter: int) -> str:
    nsfw_pct = nsfw_score.nsfw * 100
    safe_pct = nsfw_score.normal * 100

    if top_explicit:
        explicit_card = f"""
<div class="mk-stat-card mk-stat-danger">
  <div class="mk-stat-label">Top explicit detection</div>
  <div class="mk-stat-value-sm">{top_explicit.class_name}</div>
  <div class="mk-stat-sub">Score: <b>{top_explicit.score:.3f}</b></div>
</div>
"""
    else:
        explicit_card = """
<div class="mk-stat-card mk-stat-success">
  <div class="mk-stat-label">Top explicit detection</div>
  <div class="mk-stat-value-sm">None</div>
  <div class="mk-stat-sub">No explicit body parts found</div>
</div>
"""

    return f"""
<div class="mk-stats-grid">
  <div class="mk-stat-card">
    <div class="mk-stat-label">Whole-image classifier</div>
    {_gauge_bar(nsfw_score.nsfw, "NSFW", "#ef4444")}
    {_gauge_bar(nsfw_score.normal, "Safe", "#10b981")}
  </div>
  {explicit_card}
  <div class="mk-stat-card">
    <div class="mk-stat-label">Detector output</div>
    <div class="mk-stat-value">{n_dets}</div>
    <div class="mk-stat-sub">{kept_after_filter} shown after filtering</div>
  </div>
</div>
"""


def _conflicts_html(notes: list[str]) -> str:
    if not notes:
        return ""
    items = "".join(f"<li>{n}</li>" for n in notes)
    return f"""
<div class="mk-card mk-conflicts">
  <div class="mk-card-title">Pipeline notes</div>
  <ul class="mk-conflict-list">{items}</ul>
</div>
"""


def _model_badges_html() -> str:
    return """
<div class="mk-badges">
  <div class="mk-badge"><b>Detector:</b> NudeNet 640m</div>
  <div class="mk-badge"><b>Classifier:</b> Falconsai/nsfw_image_detection</div>
  <div class="mk-badge mk-badge-locked">Locked: 1024px @ 0.15 threshold</div>
</div>
"""


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------
def analyze(
    image: Image.Image | None,
    detector_threshold: float,
    selected_classes: list[str],
    blur_explicit: bool,
    resolution: int,
    nsfw_block_threshold: float,
    nsfw_review_threshold: float,
    explicit_block_threshold: float,
    explicit_review_threshold: float,
    male_score_boost: float,
):
    if image is None:
        empty_verdict = """
<div class="mk-empty">
  <div class="mk-empty-icon">↑</div>
  <div class="mk-empty-text">Drop an image above to analyze</div>
</div>
"""
        return None, [], empty_verdict, "", ""

    import time
    t0 = time.time()
    bgr = _pil_to_bgr(image)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, bgr)
        tmp_path = tmp.name

    try:
        analyzer = get_analyzer(int(resolution))
        analyzer.policy = VerdictPolicy(
            nsfw_block_threshold=nsfw_block_threshold,
            nsfw_review_threshold=nsfw_review_threshold,
            explicit_block_threshold=explicit_block_threshold,
            explicit_review_threshold=explicit_review_threshold,
            male_score_boost=male_score_boost,
        )
        result = analyzer.analyze(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    elapsed_ms = int((time.time() - t0) * 1000)

    user_kept = [
        d for d in result.kept_detections
        if d.score >= detector_threshold
        and (not selected_classes or d.class_name in selected_classes)
    ]
    annotated = annotate(bgr, user_kept, blur=blur_explicit)
    annotated_pil = _bgr_to_pil(annotated)

    rows = [
        [d.class_name, d.category, f"{d.score:.3f}", f"{d.box[2]}×{d.box[3]}"]
        for d in sorted(user_kept, key=lambda d: d.score, reverse=True)
    ]

    explicit_dets = [d for d in result.detection_result.detections
                     if d.class_name in EXPLICIT_CLASSES]
    top_explicit = max(explicit_dets, key=lambda d: d.score, default=None)

    verdict_html = _verdict_html(result.verdict, result.reason, elapsed_ms)
    stats_html = _stats_html(
        result.nsfw_score, top_explicit,
        len(result.kept_detections), len(user_kept),
    )
    conflicts_html = _conflicts_html(result.conflicts)

    return annotated_pil, rows, verdict_html, stats_html, conflicts_html


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
  --mk-bg: #fefaf6;
  --mk-card: #ffffff;
  --mk-border: #f1e8df;
  --mk-text: #1c1917;
  --mk-text-soft: #57534e;
  --mk-text-faint: #a8a29e;
  --mk-primary: #ea580c;
  --mk-primary-dark: #c2410c;
  --mk-accent: #FF4500;
  --mk-accent-soft: #FF6B35;
  --mk-shadow: 0 1px 3px rgba(204, 34, 0, 0.04), 0 1px 2px rgba(0,0,0,0.05);
  --mk-shadow-lg: 0 10px 25px -5px rgba(204, 34, 0, 0.08), 0 4px 6px -2px rgba(0,0,0,0.04);
  --mk-radius: 14px;
  --mk-radius-sm: 8px;
}

.dark, .gradio-container.dark {
  --mk-bg: #1c1413;
  --mk-card: #2d201c;
  --mk-border: #44322d;
  --mk-text: #fef3ec;
  --mk-text-soft: #d6c4b8;
  --mk-text-faint: #8a766c;
  --mk-shadow: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.4);
  --mk-shadow-lg: 0 10px 25px -5px rgba(0,0,0,0.5), 0 4px 6px -2px rgba(0,0,0,0.3);
}

.gradio-container {
  max-width: 1400px !important;
  margin: 0 auto !important;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, system-ui, sans-serif !important;
  background: var(--mk-bg) !important;
}

/* Header */
.mk-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 24px 0 8px;
  border-bottom: 1px solid var(--mk-border);
  margin-bottom: 24px;
}
.mk-header-title {
  font-size: 28px;
  font-weight: 800;
  letter-spacing: -0.025em;
  color: var(--mk-text);
  margin: 0;
}
.mk-header-title .mk-brand-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 48px; height: 48px;
  vertical-align: middle;
  margin-right: 14px;
  filter: drop-shadow(0 4px 12px rgba(255, 69, 0, 0.25));
}
.mk-header-title .mk-brand-mark svg {
  width: 100%; height: 100%;
}
.mk-header-subtitle {
  color: var(--mk-text-soft);
  font-size: 15px;
  margin-top: 6px;
}

/* Badges */
.mk-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 16px 0 24px;
}
.mk-badge {
  font-size: 12px;
  padding: 6px 12px;
  background: var(--mk-card);
  border: 1px solid var(--mk-border);
  color: var(--mk-text-soft);
  border-radius: 999px;
}
.mk-badge b { color: var(--mk-text); font-weight: 600; }
.mk-badge-locked {
  background: linear-gradient(135deg, rgba(255, 107, 53, 0.1), rgba(255, 69, 0, 0.1));
  border-color: rgba(255, 69, 0, 0.3);
  color: var(--mk-primary-dark);
  font-weight: 600;
}

/* Cards */
.mk-card {
  background: var(--mk-card);
  border: 1px solid var(--mk-border);
  border-radius: var(--mk-radius);
  padding: 24px;
  box-shadow: var(--mk-shadow);
  margin-bottom: 16px;
}
.mk-card-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--mk-text-soft);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin: 0 0 16px 0;
}

/* Verdict banner */
.mk-verdict-banner {
  border-radius: var(--mk-radius);
  padding: 28px 32px;
  display: flex;
  align-items: center;
  gap: 24px;
  color: white;
  box-shadow: var(--mk-shadow-lg);
  margin-bottom: 16px;
}
.mk-verdict-icon {
  font-size: 48px;
  font-weight: 900;
  width: 72px; height: 72px;
  background: rgba(255,255,255,0.18);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.mk-verdict-text { flex: 1; }
.mk-verdict-label {
  font-size: 32px;
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.mk-verdict-reason {
  font-size: 14px;
  opacity: 0.92;
  margin-top: 4px;
}
.mk-verdict-time {
  font-size: 13px;
  opacity: 0.85;
  font-variant-numeric: tabular-nums;
  background: rgba(255,255,255,0.18);
  padding: 6px 12px;
  border-radius: 8px;
  font-weight: 600;
}

/* Stats grid */
.mk-stats-grid {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr;
  gap: 16px;
  margin-bottom: 16px;
}
@media (max-width: 900px) {
  .mk-stats-grid { grid-template-columns: 1fr; }
}
.mk-stat-card {
  background: var(--mk-card);
  border: 1px solid var(--mk-border);
  border-radius: var(--mk-radius);
  padding: 20px;
  box-shadow: var(--mk-shadow);
}
.mk-stat-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--mk-text-faint);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 12px;
}
.mk-stat-value {
  font-size: 32px;
  font-weight: 800;
  color: var(--mk-text);
  line-height: 1;
  font-variant-numeric: tabular-nums;
}
.mk-stat-value-sm {
  font-size: 16px;
  font-weight: 700;
  color: var(--mk-text);
  font-family: ui-monospace, 'SF Mono', Menlo, monospace;
  word-break: break-all;
}
.mk-stat-sub {
  font-size: 13px;
  color: var(--mk-text-soft);
  margin-top: 6px;
}
.mk-stat-danger { border-left: 3px solid #ef4444; }
.mk-stat-success { border-left: 3px solid #10b981; }

/* Gauge */
.mk-gauge { margin-top: 8px; }
.mk-gauge + .mk-gauge { margin-top: 14px; }
.mk-gauge-header {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  margin-bottom: 6px;
}
.mk-gauge-label { color: var(--mk-text-soft); font-weight: 500; }
.mk-gauge-value { color: var(--mk-text); font-weight: 700; font-variant-numeric: tabular-nums; }
.mk-gauge-track {
  height: 8px;
  background: var(--mk-border);
  border-radius: 999px;
  overflow: hidden;
}
.mk-gauge-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 0.4s ease;
}

/* Conflicts */
.mk-conflicts { background: linear-gradient(135deg, rgba(99,102,241,0.04), rgba(236,72,153,0.04)); }
.mk-conflict-list {
  margin: 0;
  padding-left: 20px;
  color: var(--mk-text-soft);
  font-size: 14px;
}
.mk-conflict-list li { margin: 4px 0; }

/* Empty state */
.mk-empty {
  text-align: center;
  padding: 80px 20px;
  background: var(--mk-card);
  border: 2px dashed var(--mk-border);
  border-radius: var(--mk-radius);
  color: var(--mk-text-faint);
}
.mk-empty-icon {
  font-size: 64px;
  font-weight: 200;
  line-height: 1;
  margin-bottom: 12px;
}
.mk-empty-text {
  font-size: 15px;
  font-weight: 500;
}

/* Tighten Gradio's default spacing */
.gradio-container .gr-block { background: transparent !important; border: none !important; padding: 0 !important; }
.gradio-container .gr-form { background: transparent !important; border: none !important; }
.gradio-container button.primary {
  background: linear-gradient(135deg, var(--mk-accent-soft), var(--mk-accent)) !important;
  border: none !important;
  border-radius: var(--mk-radius-sm) !important;
  font-weight: 600 !important;
  color: white !important;
  box-shadow: 0 2px 8px rgba(255, 69, 0, 0.25) !important;
  transition: transform 0.1s, box-shadow 0.2s !important;
}
.gradio-container button.primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(255, 69, 0, 0.35) !important;
}

/* Footer */
.mk-footer {
  text-align: center;
  color: var(--mk-text-faint);
  font-size: 13px;
  padding: 24px 0;
  margin-top: 32px;
  border-top: 1px solid var(--mk-border);
}
.mk-footer a { color: var(--mk-primary); text-decoration: none; }
.mk-footer a:hover { text-decoration: underline; }

/* Image cards */
.mk-image-cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 16px;
}
@media (max-width: 900px) { .mk-image-cards { grid-template-columns: 1fr; } }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ModerationKit — Local Nudity Detector", css=CUSTOM_CSS) as demo:
        # Header
        gr.HTML(f"""
<div class="mk-header">
  <div>
    <div class="mk-header-title"><span class="mk-brand-mark">{LOGO_ICON_SVG}</span>ModerationKit</div>
    <div class="mk-header-subtitle">Two-model offline nudity detection with bias-compensated verdict logic.</div>
  </div>
</div>
{_model_badges_html()}""")

        # Main two-column layout
        with gr.Row():
            # ---- LEFT: input ----
            with gr.Column(scale=5):
                gr.Markdown("### Input")
                image_in = gr.Image(
                    label=None,
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=420,
                    show_label=False,
                )
                with gr.Row():
                    run_btn = gr.Button("Analyze image", variant="primary", size="lg", scale=2)
                    blur = gr.Checkbox(value=False, label="Blur explicit regions", scale=1)

                with gr.Accordion("Display filters", open=False):
                    threshold = gr.Slider(
                        0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.05,
                        label="Display threshold",
                        info="Hide detections below this score (display only — does not change the verdict).",
                    )
                    classes = gr.CheckboxGroup(
                        choices=ALL_CLASSES,
                        value=DEFAULT_CLASSES,
                        label="Classes to show",
                    )

                with gr.Accordion("Verdict policy (advanced)", open=False):
                    gr.Markdown(
                        "These thresholds determine the final verdict. "
                        "Defaults match `server.py` production settings."
                    )
                    male_boost = gr.Slider(
                        1.0, 4.0, value=2.5, step=0.1,
                        label="Male anatomy score boost",
                        info="Compensates for NudeNet's female-skewed training data.",
                    )
                    nsfw_block = gr.Slider(
                        0.0, 1.0, value=0.70, step=0.05,
                        label="Whole-image NSFW: BLOCK threshold",
                    )
                    nsfw_review = gr.Slider(
                        0.0, 1.0, value=0.30, step=0.05,
                        label="Whole-image NSFW: REVIEW threshold",
                    )
                    explicit_block = gr.Slider(
                        0.0, 1.0, value=0.50, step=0.05,
                        label="Detector explicit: BLOCK threshold",
                    )
                    explicit_review = gr.Slider(
                        0.0, 1.0, value=0.20, step=0.05,
                        label="Detector explicit: REVIEW threshold",
                    )

                with gr.Accordion("Inference settings", open=False):
                    resolution = gr.Radio(
                        choices=[320, 640, 1024, 1280],
                        value=DEFAULT_RESOLUTION,
                        label="Inference resolution",
                        info="1024 = production-locked. Higher = catches harder cases but slower.",
                    )

            # ---- RIGHT: results ----
            with gr.Column(scale=7):
                gr.Markdown("### Result")
                verdict_out = gr.HTML(_verdict_html("ALLOW", "Awaiting upload", 0))
                stats_out = gr.HTML()
                gr.Markdown("### Annotated")
                image_out = gr.Image(label=None, type="pil", height=420, show_label=False)
                with gr.Accordion("Detection table", open=True):
                    table = gr.Dataframe(
                        headers=["class", "category", "score", "size"],
                        interactive=False,
                        wrap=True,
                    )
                conflicts_out = gr.HTML()

        # Footer
        gr.HTML("""
<div class="mk-footer">
  <p>
    <b>ModerationKit</b> · 100% offline · No images leave your machine
    &nbsp;·&nbsp;
    <a href="https://github.com" target="_blank">GitHub</a>
    &nbsp;·&nbsp;
    <a href="/docs" target="_blank">API docs (when running server.py)</a>
  </p>
  <p style="margin-top:8px;font-size:11px;">
    For 18+ moderation use only. <b>This tool does not detect CSAM</b> — use NCMEC PhotoDNA for that.
  </p>
</div>
""")

        # Wiring
        inputs = [
            image_in, threshold, classes, blur, resolution,
            nsfw_block, nsfw_review, explicit_block, explicit_review,
            male_boost,
        ]
        outputs = [image_out, table, verdict_out, stats_out, conflicts_out]

        triggers = [
            run_btn.click, image_in.change,
            threshold.change, resolution.change, classes.change, blur.change,
            nsfw_block.change, nsfw_review.change,
            explicit_block.change, explicit_review.change,
            male_boost.change,
        ]
        for t in triggers:
            t(analyze, inputs=inputs, outputs=outputs)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true",
                        help="Create a public gradio.live URL")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    # Ensure the larger NudeNet model is present. install.bat does this on
    # local boxes, but Hugging Face Spaces instances start from a clean slate,
    # so we trigger the download here on first cold start.
    print("Checking detection model...")
    try:
        from download_models import download as _download_detector_model
        _download_detector_model()
    except Exception as exc:
        print(f"WARNING: detector model auto-download failed ({exc}). "
              "Falling back to bundled smaller model.")

    print("Pre-loading models (one-time)...")
    _get_classifier()
    print("Models ready.")

    demo = build_ui()
    launch_kwargs = dict(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
    # Set the browser tab favicon to the shield icon when available
    if LOGO_ICON_PATH.is_file():
        launch_kwargs["favicon_path"] = str(LOGO_ICON_PATH)
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
