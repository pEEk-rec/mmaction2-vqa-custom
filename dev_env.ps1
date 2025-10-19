# MMAction2 Development Environment Setup
# This script sets PYTHONPATH and activates venv

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  MMAction2 Dev Environment" -ForegroundColor Green  
Write-Host "========================================" -ForegroundColor Cyan

# Set PYTHONPATH to project root
$env:PYTHONPATH = "D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1"

# Activate virtual environment  
& "D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\.venv\Scripts\Activate.ps1"

# Verify setup
Write-Host "`n✅ Checking import location..." -ForegroundColor Yellow
python -c "import mmaction; import os; src = os.path.dirname(mmaction.__file__); print(f'Importing from: {src}')"

Write-Host "`n📝 Source directory: mmaction\" -ForegroundColor Cyan
Write-Host "   Edit files there - changes take effect immediately!" -ForegroundColor Green
Write-Host "`n========================================`n" -ForegroundColor Cyan
