cd $PSScriptRoot
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run .\streamlit_app.py --server.port 8502