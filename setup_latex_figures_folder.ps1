# setup_latex_figures_folder.ps1
#
# Creates a single clean folder (paper_figures_final) and copies every
# finished, paper-ready figure into it from wherever it currently lives
# (figures/ and results/figures/), so you never have to hunt across
# folders when writing LaTeX \includegraphics paths.
#
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File setup_latex_figures_folder.ps1

$dest = ".\paper_figures_final"
New-Item -ItemType Directory -Path $dest -Force | Out-Null

# ── Already-confirmed-final figures (Stage 1, all-38-model views) ─────────
$files = @(
    ".\figures\all_models_accuracy.png",          # Figure 3
    ".\figures\accuracy_vs_speed.png",             # Figure 6
    ".\figures\iou_comparison_healthy_vs_multilesion.png",  # Section 3.3.1 figure
    ".\figures\annotation_collage.png",
    ".\figures\annotation_collage_multi.png",
    ".\figures\ablation_study_yield_prediction.png",

    # ConvNeXtLarge-specific (featured model) — copy what already exists;
    # Grad-CAM, per-class ROC, and reliability diagram still need generating
    ".\results\figures\ConvNeXtLarge_confusion_matrix.png",
    ".\results\figures\ConvNeXtLarge_training_history.png"
)

Write-Host "Copying confirmed figures into $dest ..."
foreach ($f in $files) {
    if (Test-Path $f) {
        Copy-Item $f -Destination $dest -Force
        Write-Host "  Copied: $f"
    } else {
        Write-Host "  MISSING (not copied): $f"
    }
}

Write-Host "`nDone. Contents of ${dest}:"
Get-ChildItem $dest | Select-Object Name, Length

Write-Host "`nSTILL TO ADD once generated:"
Write-Host "  - ConvNeXtLarge Grad-CAM heatmaps (Figure 16)"
Write-Host "  - ConvNeXtLarge per-class ROC curves (Figure 5)"
Write-Host "  - ConvNeXtLarge reliability diagram (Figure 4)"
Write-Host "  - Updated per-class confusion breakdown for new Table 6"