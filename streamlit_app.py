# -*- coding: utf-8 -*-
import os
from datetime import date, datetime
from typing import Any, Dict, Optional, List

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, MetaData, Table, select, insert, update, delete, and_, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoSuchTableError, SQLAlchemyError

# ==============================================================
# Config da página
# ==============================================================
st.set_page_config(
    page_title="Gestão da Turma de Muay Thai",
    page_icon="🥊",
    layout="wide",
)

# ==============================================================
# Conexão e reflexão do banco
# ==============================================================
DB_URL = (
    st.secrets.get("DATABASE_URL")
    or os.getenv("DATABASE_URL")
    or f"sqlite:///{os.path.join(os.path.dirname(__file__), 'muaythai.db')}"
)

@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    # pool_pre_ping ajuda Neon/PG a reconectar após hibernação
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    return engine

engine = get_engine()
metadata = MetaData()

# nomes de tabelas (ajuste se usar outros nomes!)
T_STUDENT = "student"
T_GRADUATION = "graduation_history"
T_PAYMENT = "payment"
T_EXTRA = "extra_repasse"
T_SETTINGS = "settings"
T_COACH = "coach"
T_SLOT = "train_slot"

def reflect_table(name: str) -> Optional[Table]:
    """Reflete tabela existente. Retorna None se não existir."""
    try:
        tbl = Table(name, metadata, autoload_with=engine)
        return tbl
    except NoSuchTableError:
        return None

# ==============================================================
# Helpers de UI / datas / formatos
# ==============================================================
BIRTH_MIN = date(1900, 1, 1)
BIRTH_MAX = date.today()
TRAIN_MIN = date(1990, 1, 1)
TRAIN_MAX = date.today()

def fmt_date(d: Optional[date]) -> str:
    if not d:
        return "—"
    if isinstance(d, str):
        try:
            d = datetime.fromisoformat(d).date()
        except Exception:
            return d
    return d.strftime("%d/%m/%Y")

def parse_date(v: Any) -> Optional[date]:
    if v is None or v == "":
        return None
    if isinstance(v, date):
        return v
    try:
        # tenta ISO ou dd/mm/YYYY
        if isinstance(v, str) and "/" in v:
            return datetime.strptime(v, "%d/%m/%Y").date()
        return datetime.fromisoformat(str(v)).date()
    except Exception:
        return None

def idade_atual(dn: Optional[date]) -> str:
    if not dn:
        return "—"
    dn = parse_date(dn)
    if not dn:
        return "—"
    today = date.today()
    years = today.year - dn.year - ((today.month, today.day) < (dn.month, dn.day))
    return f"{years} anos"

def tempo_treino_fmt(dt_inicio: Optional[date]) -> str:
    if not dt_inicio:
        return "—"
    di = parse_date(dt_inicio)
    if not di:
        return "—"
    today = date.today()
    months = (today.year - di.year) * 12 + (today.month - di.month)
    if today.day < di.day:
        months -= 1
    if months < 0:
        months = 0
    anos = months // 12
    meses = months % 12
    if anos > 0:
        return f"{anos} anos e {meses} meses" if meses else f"{anos} anos"
    return f"{meses} meses"

def money(v: Any) -> str:
    try:
        f = float(v or 0)
    except Exception:
        return "R$ 0,00"
    s = f"{f:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

# ==============================================================
# LOGIN simples (admin e operador) via secrets
# ==============================================================
def _do_login():
    st.sidebar.subheader("🔐 Login")
    u = st.sidebar.text_input("Usuário", placeholder="admin ou operador")
    p = st.sidebar.text_input("Senha", type="password")
    ok = st.sidebar.button("Entrar", type="primary", use_container_width=True)

    if ok:
        users_section = st.secrets.get("users", {})
        users = {
            "admin":    {"pw": users_section.get("admin", ""), "role": "admin"},
            "operador": {"pw": users_section.get("operador", ""), "role": "operador"},
        }
        if u in users and p == users[u]["pw"] and p != "":
            st.session_state["user"] = u
            st.session_state["role"] = users[u]["role"]
            st.rerun()
        else:
            st.sidebar.error("Usuário ou senha inválidos.")
    st.stop()

if "role" not in st.session_state:
    _do_login()

with st.sidebar:
    st.caption(f"Usuário: **{st.session_state['user']}** · Perfil: **{st.session_state['role']}**")
    if st.button("Sair", use_container_width=True):
        for k in ("user", "role"):
            st.session_state.pop(k, None)
        st.rerun()

def require_admin():
    if st.session_state["role"] != "admin":
        st.warning("Somente o administrador pode acessar esta seção.")
        st.stop()

# ==============================================================
# Acesso ao banco (genérico)
# ==============================================================

def table_has_column(tbl: Table, colname: str) -> bool:
    return colname in tbl.c if tbl is not None else False

def fetch_all_students() -> pd.DataFrame:
    tbl = reflect_table(T_STUDENT)
    if not tbl:
        return pd.DataFrame()
    cols = [tbl.c.get(k) for k in tbl.c.keys()]
    stmt = select(*cols).order_by(tbl.c.get("name", list(tbl.c.values())[0]))
    with engine.begin() as conn:
        rows = conn.execute(stmt).mappings().all()
    df = pd.DataFrame(rows)
    # Campos de exibição derivados
    if "birth_date" in df:
        df["Idade"] = df["birth_date"].apply(idade_atual)
    else:
        df["Idade"] = "—"
    if "start_date" in df:
        df["Tempo de treino"] = df["start_date"].apply(tempo_treino_fmt)
    else:
        df["Tempo de treino"] = "—"

    # graduação atual (pega a mais recente se existir tabela de histórico)
    gh = reflect_table(T_GRADUATION)
    if gh is not None and not df.empty and "id" in df:
        with engine.begin() as conn:
            grads = conn.execute(
                text(f"""
                    SELECT DISTINCT ON (student_id)
                           student_id, grade, date
                    FROM {T_GRADUATION}
                    ORDER BY student_id, date DESC, id DESC
                """)
            ).mappings().all()
        gmap = {r["student_id"]: (r["grade"], r["date"]) for r in grads}
        df["Graduação"] = df["id"].map(lambda i: gmap.get(i, ("Branca", None))[0])
        df["Data Graduação"] = df["id"].map(lambda i: fmt_date(gmap.get(i, (None, None))[1]))
    else:
        df["Graduação"] = "Branca"
        df["Data Graduação"] = "—"
    return df

def insert_student(values: Dict[str, Any]) -> int:
    tbl = reflect_table(T_STUDENT)
    if not tbl:
        raise RuntimeError("Tabela 'student' não encontrada no banco.")
    data = {k: v for k, v in values.items() if table_has_column(tbl, k)}
    stmt = insert(tbl).values(**data).returning(tbl.c.id)
    with engine.begin() as conn:
        new_id = conn.execute(stmt).scalar_one()
    return int(new_id)

def update_student(stu_id: int, values: Dict[str, Any]) -> int:
    tbl = reflect_table(T_STUDENT)
    if not tbl:
        raise RuntimeError("Tabela 'student' não encontrada.")
    data = {k: v for k, v in values.items() if table_has_column(tbl, k)}
    stmt = update(tbl).where(tbl.c.id == stu_id).values(**data)
    with engine.begin() as conn:
        res = conn.execute(stmt)
        return res.rowcount or 0

def delete_student(stu_id: int) -> int:
    tbl = reflect_table(T_STUDENT)
    if not tbl:
        return 0
    stmt = delete(tbl).where(tbl.c.id == stu_id)
    with engine.begin() as conn:
        res = conn.execute(stmt)
        return res.rowcount or 0

def add_graduation(student_id: int, grade: str, grad_date: date, notes: Optional[str] = None):
    gh = reflect_table(T_GRADUATION)
    if not gh:
        return
    payload = {}
    if table_has_column(gh, "student_id"): payload["student_id"] = student_id
    if table_has_column(gh, "grade"): payload["grade"] = grade
    if table_has_column(gh, "date"): payload["date"] = grad_date
    if table_has_column(gh, "notes"): payload["notes"] = notes
    if not payload:
        return
    stmt = insert(gh).values(**payload)
    with engine.begin() as conn:
        conn.execute(stmt)

# ==============================================================
# UI - Sidebar navegação conforme perfil
# ==============================================================
ALL_PAGES = [
    "Alunos",
    "Graduações",
    "Receber Pagamento",
    "Extras (Repasse)",
    "Relatórios",
    "Importar / Exportar",
    "Configurações",
]
PAGES = ["Alunos", "Relatórios"] if st.session_state["role"] == "operador" else ALL_PAGES

st.sidebar.markdown("### Navegação")
page = st.sidebar.radio("Ir para:", PAGES, index=0, label_visibility="collapsed")

st.title("🥊 Gestão da Turma de Muay Thai")

# ==============================================================
# Página: Alunos
# ==============================================================
if page == "Alunos":
    st.subheader("Cadastro e Edição de Alunos")

    # ----- Lista de alunos
    df_students = fetch_all_students()
    if df_students.empty:
        st.info("Nenhum aluno encontrado.")
    else:
        show_cols = []
        for c in ["id","name","birth_date","start_date","monthly_fee","active","Graduação","Data Graduação","Idade","Tempo de treino"]:
            if c in df_students.columns:
                show_cols.append(c)
        df_show = df_students.copy()
        if "birth_date" in df_show:
            df_show["birth_date"] = df_show["birth_date"].apply(fmt_date)
        if "start_date" in df_show:
            df_show["start_date"] = df_show["start_date"].apply(fmt_date)
        df_show = df_show.rename(columns={
            "id":"ID","name":"Nome","birth_date":"Nascimento","start_date":"Início","monthly_fee":"Mensalidade (R$)","active":"Ativo?"
        })
        st.dataframe(df_show[show_cols].rename(columns={"name":"Nome"}), use_container_width=True, hide_index=True)

    st.divider()
    colA, colB = st.columns([1,1])

    # ----- Cadastrar novo aluno (form)
    with colA:
        st.markdown("### ➕ Cadastrar novo aluno")
        with st.form("form_novo_aluno", clear_on_submit=False):
            n_name = st.text_input("Nome *", placeholder="Digite o nome")
            n_birth = st.date_input("Data de nascimento", value=date(2000,1,1), min_value=BIRTH_MIN, max_value=BIRTH_MAX, format="DD/MM/YYYY")
            n_start = st.date_input("Início do treino", value=date.today(), min_value=TRAIN_MIN, max_value=TRAIN_MAX, format="DD/MM/YYYY")
            n_fee = st.number_input("Mensalidade (R$)", min_value=0.0, step=10.0, format="%.2f")
            n_active = st.checkbox("Ativo?", value=True)
            # IDs de professor/horário são opcionais; usa se existirem colunas
            n_coach_id = st.text_input("ID do Professor (opcional)", value="")
            n_slot_id  = st.text_input("ID do Horário (opcional)", value="")
            submitted = st.form_submit_button("Salvar", type="primary", use_container_width=True)

        if submitted:
            try:
                payload = {
                    "name": n_name.strip(),
                    "birth_date": n_birth,
                    "start_date": n_start,
                    "monthly_fee": float(n_fee or 0),
                    "active": True if n_active else False,
                }
                if n_coach_id.strip():
                    payload["coach_id"] = int(n_coach_id)
                if n_slot_id.strip():
                    payload["train_slot_id"] = int(n_slot_id)

                new_id = insert_student(payload)

                # graduação inicial "Branca" na data de início do treino
                add_graduation(new_id, "Branca", n_start, None)

                st.success(f"Aluno cadastrado (ID {new_id}).")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao salvar: {e}")

    # ----- Editar aluno (form)
    with colB:
        st.markdown("### ✏️ Editar aluno")
        if df_students.empty:
            st.info("Cadastre um aluno para habilitar a edição.")
        else:
            ids = df_students["id"].tolist()
            sid = st.selectbox("Selecionar aluno (ID)", ids, format_func=lambda i: f"ID {i} — {df_students.loc[df_students['id']==i,'name'].values[0]}")
            if sid:
                # aluno atual
                row = df_students[df_students["id"]==sid].iloc[0]
                e_name = row["name"]
                e_birth = parse_date(row.get("birth_date"))
                e_start = parse_date(row.get("start_date"))
                e_fee = float(row.get("monthly_fee", 0.0) or 0.0)
                e_active = bool(row.get("active", True))

                with st.form(f"form_editar_{sid}"):
                    col1, col2 = st.columns([2,1])
                    with col1:
                        f_name = st.text_input("Nome *", value=e_name)
                        f_birth = st.date_input("Data de nascimento", value=(e_birth or date(2000,1,1)), min_value=BIRTH_MIN, max_value=BIRTH_MAX, format="DD/MM/YYYY")
                        f_start = st.date_input("Início do treino", value=(e_start or date.today()), min_value=TRAIN_MIN, max_value=TRAIN_MAX, format="DD/MM/YYYY")
                        f_active = st.checkbox("Ativo?", value=e_active)
                    with col2:
                        f_fee = st.number_input("Mensalidade (R$)", value=e_fee, min_value=0.0, step=10.0, format="%.2f")
                        # campos opcionais, só aparecem se a tabela tiver
                        coach_id = st.text_input("ID do Professor (opcional)", value=str(row.get("coach_id","") or ""))
                        slot_id  = st.text_input("ID do Horário (opcional)", value=str(row.get("train_slot_id","") or ""))

                    b_save = st.form_submit_button("Atualizar", type="primary")
                    b_del  = st.form_submit_button("Excluir", type="secondary")

                if b_save:
                    try:
                        payload = {
                            "name": f_name.strip(),
                            "birth_date": f_birth,
                            "start_date": f_start,
                            "monthly_fee": float(f_fee or 0),
                            "active": True if f_active else False,
                        }
                        if coach_id.strip():
                            payload["coach_id"] = int(coach_id)
                        if slot_id.strip():
                            payload["train_slot_id"] = int(slot_id)
                        n = update_student(int(sid), payload)
                        st.success("Registro atualizado." if n else "Nada para atualizar.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro: {e}")

                if b_del:
                    require_admin()
                    try:
                        n = delete_student(int(sid))
                        st.success("Aluno excluído." if n else "Aluno não encontrado.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro: {e}")

# ==============================================================
# Página: Graduações (placeholder leve)
# ==============================================================
elif page == "Graduações":
    require_admin()
    st.info("Tela de Graduações — em breve nesta versão simplificada. (Seu banco não é alterado.)")

# ==============================================================
# Página: Receber Pagamento (placeholder leve)
# ==============================================================
elif page == "Receber Pagamento":
    require_admin()
    st.info("Tela de Recebimento — em breve nesta versão simplificada. (Use sua versão completa se preferir.)")

# ==============================================================
# Página: Extras (Repasse) (placeholder leve)
# ==============================================================
elif page == "Extras (Repasse)":
    require_admin()
    st.info("Tela de Extras/Repasse — em breve nesta versão simplificada.")

# ==============================================================
# Página: Relatórios (resumo simples para não travar)
# ==============================================================
elif page == "Relatórios":
    st.subheader("Relatórios (resumo simples)")
    # Mostra apenas um snapshot de alunos com idade/tempo/graduação
    df_students = fetch_all_students()
    if df_students.empty:
        st.info("Sem alunos para apresentar.")
    else:
        show_cols = ["id","name","Idade","Tempo de treino","Graduação","Data Graduação"]
        show_cols = [c for c in show_cols if c in df_students.columns]
        df = df_students[show_cols].rename(columns={"id":"ID","name":"Aluno"})
        st.dataframe(df, use_container_width=True, hide_index=True)

# ==============================================================
# Página: Importar / Exportar (placeholder)
# ==============================================================
elif page == "Importar / Exportar":
    require_admin()
    st.info("Ferramentas de importação/exportação — em breve.")

# ==============================================================
# Página: Configurações (placeholder)
# ==============================================================
elif page == "Configurações":
    require_admin()
    st.info("Configurações gerais — em breve.")
