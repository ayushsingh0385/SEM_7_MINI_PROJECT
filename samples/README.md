# Demo files

Ready-made uploads for trying the app without hunting for input.

Every file here comes from the **held-out test split** of its dataset, so no
model in this project was trained on any of them. The filename gives the true
label, which makes them useful for a live demo: state the expected answer before
you press Analyse.

| Folder | Files | Source |
|--------|-------|--------|
| `images/` | `real_01..03.png`, `ai_01..03.png` | CIFAKE test split. 32×32 PNGs — small on purpose, because that is the resolution the model was trained at. |
| `text/` | `human_01..02.txt`, `ai_01..02.txt` | HC3 test split, after the same detokenisation normalisation the training data receives. |

Regenerate them after rebuilding a dataset:

```bash
python scripts/prepare_image_data.py
python scripts/prepare_text_data.py
```

then re-run the short snippet in the project README, or simply use the
**Try an example** tab in the text panel, which samples the test split live.

## A note on the image samples

These are 32×32, matching the training distribution, so the detector is at its
strongest on them. A 4K photograph from the web is a much harder case and the
model is noticeably weaker there — that gap is the domain shift documented in
the main README, not a bug in these files.
