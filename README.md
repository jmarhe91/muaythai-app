# App de Gestão da Turma de Muay Thai

**Tecnologias**: Streamlit (UI), SQLite + SQLModel (banco), Pandas/Altair (relatórios).  
**Como rodar localmente**:

1. Instale o Python 3.10+.
2. No terminal, dentro desta pasta:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # ou: source .venv/bin/activate  # Linux/Mac

   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```
3. Abra o endereço indicado (ex.: http://localhost:8501).

## Recursos do MVP
- **Cadastro de alunos** (nome, nascimento, idade calculada, início, tempo de treino calculado, graduação e data, ativo, mensalidade).
- **Baixa de pagamentos** com mês de referência, forma, observações e **repasse automático** (% configurável) + ajuste.
- **Relatórios mensais**: receita bruta, repasse, líquida; detalhamento e por aluno.
- **Importar/Exportar**: importa CSV de alunos; exporta CSV de alunos e pagamentos.
- **Configurações**: % padrão de repasse e símbolo de moeda.

## Próximos passos sugeridos
- Autenticação simples (streamlit-auth-component) para proteger o app on-line.
- Ajustes de usabilidade (edição/arquivamento de alunos; parcelamentos; faturas).
- Hospedar de graça no **Streamlit Community Cloud** (conecta com seu GitHub).
- Backups automáticos do SQLite (Google Drive) e logs de auditoria.
