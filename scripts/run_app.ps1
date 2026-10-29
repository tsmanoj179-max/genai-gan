# Run Streamlit Web Application
Write-Host "Starting Generative AI Synthesis Studio..." -ForegroundColor Cyan

# Check if conda environment is active, otherwise try activating it
if ($env:CONDA_DEFAULT_ENV -ne "gan-image-project") {
    Write-Host "Attempting to activate 'gan-image-project' conda environment..." -ForegroundColor Yellow
    try {
        conda activate gan-image-project 2>$null
    } catch {
        # continue
    }
}

# Run Streamlit
streamlit run src/app.py
