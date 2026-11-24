# Presentation Build Sources

Open the [PowerPoint](../../01_presentation/final_presentation_project_a_project_b.pptx) or [PDF](../../01_presentation/final_presentation_project_a_project_b.pdf) directly. The speaker notes are also available as [Markdown](../../01_presentation/speaker_notes.md) and [JSON](speaker_notes.json).

With `@oai/artifact-tool` available to Node.js, run this from the repository root to rebuild the editable deck and notes from the included figures.

```powershell
node Evidence/presentation_source/deck.mjs runs/presentation
```

`build_figures.py` records how the results plots were made. Its authoring-machine paths identify read-only source inputs, and the point-level Project A inputs remain outside the public repository. The [included figures](../../Figures/) can be used to rebuild the slides without those inputs.

Project A and B runtime checks are independent of the presentation tooling. The default checks use the `niulaoshi` Python environment.
