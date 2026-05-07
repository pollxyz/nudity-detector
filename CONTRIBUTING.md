# Contributing to ModerationKit

Thanks for your interest. This document covers how to set up a dev environment, what kinds of contributions are most welcome, and the testing checklist for PRs.

## Setup for development

```bash
git clone https://github.com/YOUR_USERNAME/moderationkit.git
cd moderationkit

# Windows
install.bat

# macOS / Linux
chmod +x install.sh && ./install.sh

# Verify everything works
.venv/Scripts/python.exe -c "from analyzer import CombinedAnalyzer; print('ok')"   # Windows
.venv/bin/python -c "from analyzer import CombinedAnalyzer; print('ok')"           # macOS/Linux
```

## What we welcome

In rough priority order:

1. **Better detection models.** If you find a public open-source model that handles the male-anatomy bias better than NudeNet 640m + boost, swap-in PRs are welcome. The `Detector` class abstracts the underlying ONNX session, so a new detector that returns the same `Detection` objects drops in cleanly.
2. **CSAM-detection integrations.** Wrappers for PhotoDNA, Safer, etc. that drop in alongside the existing analyzer. **Don't try to train a CSAM detector** — work with the established services.
3. **Additional integration examples.** We have Node/Express, Next.js, Django, Laravel. PRs for Rails, Go, Rust, Elixir, Spring Boot, .NET welcome.
4. **Threshold tuning data.** If you have a labeled set of profile pictures from your platform and ran sweeps to find optimal thresholds for your demographic, share the recommendations (anonymized).
5. **Better cross-class conflict rules.** The current rules in `analyzer.py` are heuristic. If you've found edge cases that break them, submit a test case + fix.
6. **UI / accessibility improvements** in `app.py` — keyboard shortcuts, screen-reader labels, mobile responsiveness.

## What we don't want

- **Datasets containing minors.** Period. Reject any PR that ships, references, or trains on such data.
- **Deepfake detection.** Out of scope.
- **Content classification beyond NSFW** (violence, hate symbols, etc.) — different problem space, different models, won't merge here.
- **API key auth, billing, multi-tenancy.** This is a self-hosted single-tenant tool by design.

## Testing checklist for PRs

Before submitting:

1. **Smoke-test the analyzer** — feed it a known-safe image, a known-explicit image (your test set, not committed), confirm verdicts are sensible
2. **Smoke-test the API** — `curl http://127.0.0.1:8000/health`, then `POST /check` with a sample image
3. **Smoke-test the UI** — start `app.py`, upload an image, confirm the verdict + annotations + stats render
4. **Don't commit large files** — model files, test images, or anything > 1 MB. The `.gitignore` already excludes `*.onnx` and the venv.
5. **Run the import smoke** — `python -c "import detector, analyzer, nsfw_classifier, app, server"` should print nothing.

## Code style

- Python 3.10+ syntax (`from __future__ import annotations`, union types with `|`, dataclasses)
- Type hints on public functions
- No comments that just restate what the code does — only WHY
- Keep modules focused: `detector.py` is detector-only, `analyzer.py` is composition, `app.py` is UI, `server.py` is HTTP

## Commit messages

Conventional commits encouraged but not required. Just be descriptive:
- `detector: support dynamic ONNX input shape`
- `analyzer: tune male_score_boost default to 2.5x`
- `docs: add Rails integration example`

## License

By submitting a PR you agree your contribution will be licensed under MIT (the project license).
