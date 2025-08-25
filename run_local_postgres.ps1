param([string]$DBURL="")
if (-not $DBURL) {
  Write-Host "Informe a DATABASE_URL como parâmetro ou edite este arquivo."
  exit 1
}
cd $PSScriptRoot
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:DATABASE_URL = $DBURL
python -m streamlit run .\streamlit_app.py --server.port 8502