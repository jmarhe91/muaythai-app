MUAYTHAI — APP STREAMLIT (ONLINE-READY)

COMO RODAR (SQLite local):
  1) Abra o PowerShell na pasta do projeto.
  2) Crie/ative a venv e instale requisitos:
     py -m venv .venv
     .\.venv\Scripts\Activate.ps1
     python -m pip install -r requirements.txt
  3) Rode:
     python -m streamlit run .\streamlit_app.py --server.port 8502
  (ou apenas execute:  .\run_local_sqlite.ps1)

COMO RODAR (Postgres/Neon):
  1) Ative a venv e instale requisitos (como acima).
  2) Rode:
     $env:DATABASE_URL = "postgresql+psycopg://USUARIO:SENHA@HOST/DB?sslmode=require&channel_binding=require"
     python -m streamlit run .\streamlit_app.py --server.port 8502
  (ou use: .\run_local_postgres.ps1 -DBURL "postgresql+psycopg://...")

DEPLOY NO STREAMLIT CLOUD:
  - Suba streamlit_app.py e requirements.txt para o GitHub.
  - Em Advanced settings -> Secrets:
    DATABASE_URL = "postgresql+psycopg://USUARIO:SENHA@HOST/DB?sslmode=require&channel_binding=require"
  - Deploy.

OBS:
  - Se quiser usar logo, salve um arquivo "logo.png" na pasta do app e troque o page_icon, se desejar.
  - Colunas e páginas seguem as regras solicitadas (repasse, extras, relatórios, filtros etc.).