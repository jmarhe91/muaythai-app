# streamlit_app.py
# -------------------------------------------------------------
# JAT - Gestão de alunos
# -------------------------------------------------------------
# Requisitos:
#   pip install streamlit sqlalchemy psycopg[binary] pandas python-dateutil
# -------------------------------------------------------------

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any, Set

import pandas as pd
import streamlit as st
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Date, Boolean, Text,
    Numeric, ForeignKey, select, insert, update, delete, func, Index, inspect
)
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError

# ============================================================
# Aparência / Branding
# ============================================================
st.set_page_config(page_title="JAT - Gestão de alunos", page_icon="🥊", layout="wide")
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=120)
st.title("JAT - Gestão de alunos")

# ============================================================
# Login (admin / operador)
# ============================================================
def load_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

ADMIN_PWD = load_secret("ADMIN_PWD") or load_secret("admin") or "admin"
OPERATOR_PWD = load_secret("OPERATOR_PWD") or load_secret("operador") or "operador"

def require_login() -> str:
    role = st.session_state.get("role")
    if role in ("admin", "operador"):
        return role

    with st.form("login_form"):
        st.subheader("Login")
        col1, col2 = st.columns(2)
        with col1:
            perfil = st.selectbox("Perfil", ["operador", "admin"])
        with col2:
            pwd = st.text_input("Senha", type="password")
        ok = st.form_submit_button("🔐 Entrar")
        if ok:
            if perfil == "admin" and pwd == ADMIN_PWD:
                st.session_state["role"] = "admin"
                st.success("Login como administrador.")
                st.rerun()
            elif perfil == "operador" and pwd == OPERATOR_PWD:
                st.session_state["role"] = "operador"
                st.success("Login como operador.")
                st.rerun()
            else:
                st.error("Credenciais inválidas.")
        st.stop()
    return "operador"

# ============================================================
# Datas - limites centralizados
# ============================================================
TODAY = date.today()
BIRTH_MIN, BIRTH_MAX = date(1930, 1, 1), TODAY
START_MIN, START_MAX = date(2000, 1, 1), TODAY
GRADE_MIN, GRADE_MAX = START_MIN, TODAY

def first_day_same_month_years_ago(d: date, years: int) -> date:
    return date(d.year - years, d.month, 1)

PAY_MIN, PAY_MAX = first_day_same_month_years_ago(TODAY, 1), TODAY + timedelta(days=1)

def clamp_date(d: Optional[date], min_d: date, max_d: date) -> date:
    if d is None:
        return min_d
    if d < min_d:
        return min_d
    if d > max_d:
        return max_d
    return d

def this_month_ref(d: date = TODAY) -> str:
    return d.strftime("%Y-%m")

# ============================================================
# Banco (Postgres -> SQLite fallback)
# ============================================================
DB_URL = load_secret("DATABASE_URL") or os.getenv("DATABASE_URL") or f"sqlite:///muaythai.db"

@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    return create_engine(DB_URL, pool_pre_ping=True, future=True)

engine = get_engine()
metadata = MetaData()

# ============================================================
# Esquema (sem mudanças estruturais)
# ============================================================
settings = Table(
    "settings", metadata,
    Column("id", Integer, primary_key=True),
    Column("master_percent", Numeric(5, 4), nullable=False, server_default="0.60"),
)

coach = Table(
    "coach", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(120), nullable=False, unique=True),
    Column("full_pass", Boolean, nullable=False, server_default="false"),
)

train_slot = Table(
    "train_slot", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(120), nullable=False, unique=True),
)

student = Table(
    "student", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(200), nullable=False),
    Column("birth_date", Date, nullable=True),
    Column("start_date", Date, nullable=True),
    Column("monthly_fee", Numeric(12, 2), nullable=False, server_default="0"),
    Column("active", Boolean, nullable=False, server_default="true"),
    Column("coach_id", Integer, ForeignKey("coach.id"), nullable=True),
    Column("train_slot_id", Integer, ForeignKey("train_slot.id"), nullable=True),
    Column("master_percent_override", Numeric(5, 4), nullable=True),
    Column("grade", String(50), nullable=True),
    Column("grade_date", Date, nullable=True),
)
Index("ix_student_coach_id", student.c.coach_id)
Index("ix_student_train_slot_id", student.c.train_slot_id)

graduation = Table(
    "graduation", metadata,
    Column("id", Integer, primary_key=True),
    Column("student_id", Integer, ForeignKey("student.id"), nullable=False),
    Column("grade", String(50), nullable=False),
    Column("grade_date", Date, nullable=False),
    Column("notes", String(255), nullable=True),
)
Index("ix_graduation_student_id", graduation.c.student_id)

payment = Table(
    "payment", metadata,
    Column("id", Integer, primary_key=True),
    Column("student_id", Integer, ForeignKey("student.id"), nullable=False),
    Column("paid_date", Date, nullable=False),
    Column("month_ref", String(7), nullable=False),  # 'AAAA-MM'
    Column("amount", Numeric(12, 2), nullable=False),
    Column("method", String(20), nullable=True),
    Column("notes", String(255), nullable=True),
    Column("master_percent_used", Numeric(5, 4), nullable=False, server_default="0"),
    Column("master_adjustment", Numeric(12, 2), nullable=False, server_default="0"),
    Column("master_amount", Numeric(12, 2), nullable=False, server_default="0"),
)
Index("ix_payment_student_id", payment.c.student_id)
Index("ix_payment_month_ref", payment.c.month_ref)

extra_repasse = Table(
    "extra_repasse", metadata,
    Column("id", Integer, primary_key=True),
    Column("date", Date, nullable=False, server_default=func.current_date()),
    Column("month_ref", String(7), nullable=False),
    Column("description", Text, nullable=False),
    Column("amount", Numeric(12, 2), nullable=False),  # pode ser negativo
    Column("is_recurring", Boolean, nullable=False, server_default="false"),
    Column("student_id", Integer, ForeignKey("student.id"), nullable=True),
    Column("created_at", Date, nullable=False, server_default=func.current_date()),
)
Index("ix_extra_repasse_month_ref", extra_repasse.c.month_ref)
Index("ix_extra_repasse_student_id", extra_repasse.c.student_id)

BELTS = [
    "Branca", "Amarelo", "Amarelo e Branca", "Verde", "Verde e Branca",
    "Azul", "Azul e Branca", "Marrom", "Marrom e Branca", "Vermelha",
    "Vermelha e Branca", "Preta",
]

# ============================================================
# Bootstrapping (autocura)
# ============================================================
REQUIRED_TABLES = {"settings","coach","train_slot","student","graduation","payment","extra_repasse"}

def bootstrap_db_if_needed() -> None:
    try:
        insp = inspect(engine)
        existing = set(insp.get_table_names())
        if not REQUIRED_TABLES.issubset(existing):
            metadata.create_all(engine)
        with engine.begin() as conn:
            row = conn.execute(select(settings.c.id).where(settings.c.id == 1)).first()
            if not row:
                conn.execute(insert(settings).values(id=1, master_percent=0.60))
    except SQLAlchemyError:
        pass

bootstrap_db_if_needed()

# ============================================================
# Helpers / cache
# ============================================================
def fmt_money(v: Any) -> str:
    try:
        x = float(v or 0)
    except Exception:
        x = 0.0
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_pct_decimal_to_ui(d: Optional[float]) -> int:
    if d is None:
        return 0
    return int(round(float(d) * 100))

def pct_ui_to_decimal(p: Optional[int]) -> Optional[float]:
    if p is None:
        return None
    return round(p / 100.0, 4)

@st.cache_data(show_spinner=False)
def fetch_settings() -> Dict[str, Any]:
    with engine.begin() as conn:
        row = conn.execute(select(settings)).mappings().first()
        return dict(row) if row else {"id": 1, "master_percent": 0.60}

@st.cache_data(show_spinner=False)
def fetch_coaches() -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(select(coach).order_by(coach.c.name)).mappings().all()
        return [dict(r) for r in rows]

@st.cache_data(show_spinner=False)
def fetch_slots() -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(select(train_slot).order_by(train_slot.c.name)).mappings().all()
        return [dict(r) for r in rows]

@st.cache_data(show_spinner=False)
def fetch_students_df() -> pd.DataFrame:
    """Busca alunos; se faltar tabela/coluna, tenta bootstrap e repete uma vez."""
    try:
        with engine.begin() as conn:
            rows = conn.execute(select(student).order_by(student.c.name)).mappings().all()
            df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[c.name for c in student.c])
        return df
    except ProgrammingError:
        bootstrap_db_if_needed()
        with engine.begin() as conn:
            rows = conn.execute(select(student).order_by(student.c.name)).mappings().all()
            df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[c.name for c in student.c])
        return df

@st.cache_data(show_spinner=False)
def get_paid_ids_for_month(month_ref: str) -> Set[int]:
    with engine.begin() as conn:
        rows = conn.execute(select(payment.c.student_id).where(payment.c.month_ref==month_ref)).all()
    return {int(r[0]) for r in rows}

def invalidate_all_cache():
    fetch_settings.clear()
    fetch_coaches.clear()
    fetch_slots.clear()
    fetch_students_df.clear()
    get_paid_ids_for_month.clear()

def compute_master_percent_for_student(conn: Connection, stu_row: Dict[str, Any]) -> float:
    percent = None
    cid = stu_row.get("coach_id")
    if cid:
        c = conn.execute(select(coach).where(coach.c.id == cid)).mappings().first()
        if c and bool(c["full_pass"]):
            return 1.0
    if stu_row.get("master_percent_override") is not None:
        percent = float(stu_row["master_percent_override"])
    if percent is None:
        cfg = conn.execute(select(settings)).mappings().first()
        percent = float(cfg["master_percent"]) if cfg and cfg.get("master_percent") is not None else 0.6
    return percent

# ============================================================
# Páginas
# ============================================================
def page_alunos(role: str):
    st.header("Alunos")

    # Mês para o status de pagamento (garante consistência com relatórios/recebimento)
    month_for_status = st.text_input("Mês p/ status (AAAA-MM)", value=this_month_ref())

    df = fetch_students_df()
    if df.empty:
        st.info("Nenhum aluno cadastrado.")
    else:
        paid_ids = get_paid_ids_for_month(month_for_status)

        view = df.copy()
        view["Mensalidade"] = view["monthly_fee"].apply(fmt_money)
        view["Ativo"] = view["active"].map({True: "Sim", False: "Não"})
        view["Repasse % (aluno)"] = view["master_percent_override"].apply(
            lambda v: f"{fmt_pct_decimal_to_ui(float(v))*1}%" if v is not None else "-"
        )
        view["Nasc."] = pd.to_datetime(view["birth_date"]).dt.strftime("%d/%m/%Y").fillna("")
        view["Início"] = pd.to_datetime(view["start_date"]).dt.strftime("%d/%m/%Y").fillna("")
        view["Graduação"] = view["grade"].fillna("-")
        view["Data grad."] = pd.to_datetime(view["grade_date"]).dt.strftime("%d/%m/%Y").fillna("")
        view["Status"] = view["id"].apply(lambda i: "🟢 Pago" if int(i) in paid_ids else "🔴 Pendente")

        cols = ["name","Status","Nasc.","Início","Mensalidade","Ativo","Graduação","Data grad.","Repasse % (aluno)"]
        st.dataframe(view[cols].rename(columns={"name":"Nome"}),
                     use_container_width=True, hide_index=True)

    # ---- Cadastro de novo aluno ----
    with st.expander("➕ Cadastrar novo aluno", expanded=False):
        with st.form("form_new_student"):
            col1, col2, col3 = st.columns(3)
            with col1:
                n_name = st.text_input("Nome*", max_chars=200)
                n_birth = st.date_input(
                    "Data de nascimento", value=clamp_date(date(2000,1,1), BIRTH_MIN, BIRTH_MAX),
                    min_value=BIRTH_MIN, max_value=BIRTH_MAX, format="DD/MM/YYYY",
                )
            with col2:
                n_start = st.date_input(
                    "Início do treino", value=clamp_date(TODAY, START_MIN, START_MAX),
                    min_value=START_MIN, max_value=START_MAX, format="DD/MM/YYYY",
                )
                n_fee = st.number_input("Mensalidade (R$)", min_value=0.0, step=10.0, value=0.0)
            with col3:
                coaches = fetch_coaches()
                slots = fetch_slots()
                opt_coach = ["(nenhum)"] + [c["name"] for c in coaches]
                opt_slot = ["(nenhum)"] + [s["name"] for s in slots]
                n_coach_name = st.selectbox("Professor responsável", opt_coach)
                n_slot_name = st.selectbox("Horário de treino", opt_slot)
                n_override_pct = st.number_input(
                    "Repasse do aluno (%) (0 = usar padrão)", min_value=0, max_value=100, value=0, step=5
                )
                n_active = st.checkbox("Ativo?", value=True)

            sub = st.form_submit_button("💾 Salvar aluno")

        if sub:
            if not n_name.strip():
                st.error("Informe o nome.")
            else:
                try:
                    with engine.begin() as conn:
                        coach_id = None
                        slot_id = None
                        if n_coach_name != "(nenhum)":
                            r = conn.execute(select(coach.c.id).where(coach.c.name == n_coach_name)).first()
                            coach_id = r[0] if r else None
                        if n_slot_name != "(nenhum)":
                            r = conn.execute(select(train_slot.c.id).where(train_slot.c.name == n_slot_name)).first()
                            slot_id = r[0] if r else None

                        res = conn.execute(insert(student).values(
                            name=n_name.strip(),
                            birth_date=n_birth,
                            start_date=n_start,
                            monthly_fee=float(n_fee or 0),
                            active=bool(n_active),
                            coach_id=coach_id,
                            train_slot_id=slot_id,
                            master_percent_override=pct_ui_to_decimal(n_override_pct) if n_override_pct else None,
                        ))
                        stu_id = res.inserted_primary_key[0]
                        # faixa branca automática com data do início
                        conn.execute(insert(graduation).values(student_id=stu_id, grade="Branca", grade_date=(n_start or TODAY)))
                        conn.execute(update(student).where(student.c.id == stu_id).values(grade="Branca", grade_date=(n_start or TODAY)))

                    invalidate_all_cache()
                    st.success("✅ Aluno cadastrado com sucesso.")
                    st.rerun()
                except SQLAlchemyError as e:
                    st.error(f"Erro ao salvar: {e}")

    # ---- Edição ----
    with st.expander("✏️ Editar aluno", expanded=False):
        df = fetch_students_df()
        if df.empty:
            st.info("Sem alunos.")
        else:
            sid = st.selectbox(
                "Selecionar aluno",
                options=df["id"].tolist(),
                format_func=lambda i: df.loc[df["id"]==i, "name"].values[0],
            )
            row = df.loc[df["id"]==sid].iloc[0].to_dict()
            with st.form("form_edit_student"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    e_name = st.text_input("Nome*", value=row["name"])
                    e_birth = st.date_input(
                        "Data de nascimento",
                        value=clamp_date(row.get("birth_date"), BIRTH_MIN, BIRTH_MAX),
                        min_value=BIRTH_MIN, max_value=BIRTH_MAX, format="DD/MM/YYYY",
                    )
                with col2:
                    e_start = st.date_input(
                        "Início do treino",
                        value=clamp_date(row.get("start_date"), START_MIN, START_MAX),
                        min_value=START_MIN, max_value=START_MAX, format="DD/MM/YYYY",
                    )
                    e_fee = st.number_input("Mensalidade (R$)", min_value=0.0, step=5.0, value=float(row.get("monthly_fee") or 0))
                with col3:
                    coaches = fetch_coaches()
                    slots = fetch_slots()
                    opt_coach = ["(nenhum)"] + [c["name"] for c in coaches]
                    opt_slot = ["(nenhum)"] + [s["name"] for s in slots]
                    current_coach_name = next((c["name"] for c in coaches if c["id"] == row.get("coach_id")), "(nenhum)")
                    current_slot_name = next((s["name"] for s in slots if s["id"] == row.get("train_slot_id")), "(nenhum)")
                    e_coach_name = st.selectbox("Professor responsável", opt_coach, index=opt_coach.index(current_coach_name))
                    e_slot_name = st.selectbox("Horário de treino", opt_slot, index=opt_slot.index(current_slot_name))
                    e_override_pct = st.number_input(
                        "Repasse do aluno (%) (0 = usar padrão)",
                        min_value=0, max_value=100,
                        value=fmt_pct_decimal_to_ui(row.get("master_percent_override")) if row.get("master_percent_override") is not None else 0,
                        step=5,
                    )
                    e_active = st.checkbox("Ativo?", value=bool(row.get("active")))

                ok = st.form_submit_button("💾 Salvar alterações")
            if ok:
                try:
                    with engine.begin() as conn:
                        coach_id = None
                        slot_id = None
                        if e_coach_name != "(nenhum)":
                            r = conn.execute(select(coach.c.id).where(coach.c.name == e_coach_name)).first()
                            coach_id = r[0] if r else None
                        if e_slot_name != "(nenhum)":
                            r = conn.execute(select(train_slot.c.id).where(train_slot.c.name == e_slot_name)).first()
                            slot_id = r[0] if r else None
                        conn.execute(update(student).where(student.c.id == sid).values(
                            name=e_name.strip(),
                            birth_date=e_birth,
                            start_date=e_start,
                            monthly_fee=float(e_fee or 0),
                            active=bool(e_active),
                            coach_id=coach_id,
                            train_slot_id=slot_id,
                            master_percent_override=pct_ui_to_decimal(e_override_pct) if e_override_pct else None,
                        ))
                    invalidate_all_cache()
                    st.success("✅ Dados atualizados.")
                    st.rerun()
                except SQLAlchemyError as e:
                    st.error(f"Erro ao atualizar: {e}")

    # ---- Exclusão ----
    with st.expander("🗑️ Excluir aluno", expanded=False):
        df = fetch_students_df()
        if df.empty:
            st.info("Sem alunos.")
        else:
            sid_del = st.selectbox(
                "Escolha o aluno",
                options=df["id"].tolist(),
                format_func=lambda i: df.loc[df["id"]==i, "name"].values[0],
            )
            st.warning("Esta ação é permanente. Pagamentos e graduações vinculadas serão removidos (via regras de FK).")
            colx, coly = st.columns([1,3])
            with colx:
                confirm = st.checkbox("Confirmo a exclusão")
            with coly:
                if st.button("🗑️ Excluir aluno (definitivo)", disabled=not confirm):
                    try:
                        with engine.begin() as conn:
                            conn.execute(delete(student).where(student.c.id==sid_del))
                        invalidate_all_cache()
                        st.success("Aluno excluído.")
                        st.rerun()
                    except SQLAlchemyError as e:
                        st.error(f"Erro ao excluir: {e}")

def page_graduacoes(role: str):
    st.header("Graduações")
    df = fetch_students_df()
    if df.empty:
        st.info("Cadastre alunos primeiro.")
        return
    sid = st.selectbox(
        "Aluno",
        options=df["id"].tolist(),
        format_func=lambda i: df.loc[df["id"]==i, "name"].values[0]
    )
    stu = df.loc[df["id"]==sid].iloc[0].to_dict()

    with engine.begin() as conn:
        rows = conn.execute(select(graduation).where(graduation.c.student_id==sid)
                            .order_by(graduation.c.grade_date.desc(), graduation.c.id.desc())).mappings().all()
    hist = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[c.name for c in graduation.c])
    if not hist.empty:
        hist["Data"] = pd.to_datetime(hist["grade_date"]).dt.strftime("%d/%m/%Y")
        st.subheader("Histórico")
        st.dataframe(hist[["grade","Data","notes"]].rename(columns={"grade":"Graduação","notes":"Observações"}),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Sem graduações registradas para este aluno.")

    with st.form("form_add_grade"):
        col1, col2 = st.columns([2,1])
        with col1:
            g_grade = st.selectbox("Graduação", BELTS, index=BELTS.index("Branca"))
        with col2:
            min_g = max(GRADE_MIN, stu.get("start_date") or GRADE_MIN)
            g_date = st.date_input(
                "Data da graduação",
                value=clamp_date(TODAY, min_g, GRADE_MAX),
                min_value=min_g, max_value=GRADE_MAX, format="DD/MM/YYYY",
            )
        g_notes = st.text_input("Observações (opcional)")
        ok = st.form_submit_button("💾 Salvar graduação")
    if ok:
        try:
            with engine.begin() as conn:
                conn.execute(insert(graduation).values(student_id=sid, grade=g_grade, grade_date=g_date, notes=g_notes or None))
                # atualiza grade atual do aluno
                last = conn.execute(
                    select(graduation.c.grade, graduation.c.grade_date)
                    .where(graduation.c.student_id == sid)
                    .order_by(graduation.c.grade_date.desc(), graduation.c.id.desc())
                    .limit(1)
                ).mappings().first()
                if last:
                    conn.execute(update(student)
                                 .where(student.c.id == sid)
                                 .values(grade=last["grade"], grade_date=last["grade_date"]))
            invalidate_all_cache()
            st.success("✅ Graduação registrada.")
            st.rerun()
        except SQLAlchemyError as e:
            st.error(f"Erro ao salvar graduação: {e}")

def page_receber(role: str):
    st.header("Receber Pagamento")
    df = fetch_students_df()
    if df.empty:
        st.info("Cadastre alunos primeiro.")
        return

    df_active = df[df["active"]==True].copy()
    coaches = fetch_coaches()

    with st.form("form_receive"):
        col1, col2, col3 = st.columns(3)
        with col1:
            filtro_coach = st.selectbox("Filtrar por professor", ["(todos)"] + [c["name"] for c in coaches])
        with col2:
            paid_date = st.date_input(
                "Data do pagamento",
                value=clamp_date(TODAY, PAY_MIN, PAY_MAX),
                min_value=PAY_MIN, max_value=PAY_MAX, format="DD/MM/YYYY",
            )
        with col3:
            month_ref = st.text_input("Mês de referência (AAAA-MM)", value=this_month_ref())

        if filtro_coach != "(todos)":
            cid = next((c["id"] for c in coaches if c["name"]==filtro_coach), None)
            df_active = df_active[df_active["coach_id"]==cid]

        # Filtro de status pago/pendente
        status_filter = st.radio("Status", ["Pendentes","Pagos","Todos"], horizontal=True, index=0)

        paid_ids = get_paid_ids_for_month(month_ref)
        if status_filter == "Pendentes":
            df_list = df_active[~df_active["id"].isin(paid_ids)].copy()
        elif status_filter == "Pagos":
            df_list = df_active[df_active["id"].isin(paid_ids)].copy()
        else:
            df_list = df_active.copy()

        # Seleção só de pendentes (para não pagar 2x)
        selectable_ids = df_list[~df_list["id"].isin(paid_ids)]["id"].tolist()
        sel = st.multiselect(
            "Selecione os alunos que pagaram (somente pendentes aparecem aqui)",
            options=selectable_ids,
            format_func=lambda i: df_active.loc[df_active["id"]==i, "name"].values[0] + \
                                  f" (R$ {float(df_active.loc[df_active['id']==i,'monthly_fee'].values[0]):.2f})"
        )
        method = st.selectbox("Forma", ["Dinheiro","PIX","Cartão","Transferência","Outro"])
        notes = st.text_input("Observações (opcional)")

        ok = st.form_submit_button("✅ Confirmar recebimento")

    if ok:
        if not sel:
            st.warning("Selecione pelo menos um aluno.")
        else:
            already = []
            inserted = 0
            try:
                with engine.begin() as conn:
                    for sid in sel:
                        exists = conn.execute(
                            select(payment.c.id).where(
                                (payment.c.student_id==sid) &
                                (payment.c.month_ref==month_ref)
                            )
                        ).first()
                        if exists:
                            already.append(sid)
                            continue
                        stu_row = df[df["id"]==sid].iloc[0].to_dict()
                        amount = float(stu_row.get("monthly_fee") or 0)
                        percent = compute_master_percent_for_student(conn, stu_row)
                        master_amt = round(amount * percent, 2)
                        conn.execute(insert(payment).values(
                            student_id=sid, paid_date=paid_date, month_ref=month_ref,
                            amount=amount, method=method, notes=notes or None,
                            master_percent_used=percent, master_adjustment=0, master_amount=master_amt
                        ))
                        inserted += 1
                if inserted:
                    st.success(f"✅ {inserted} pagamento(s) registrado(s).")
                if already:
                    nomes = ", ".join(df.loc[df["id"].isin(already), "name"].tolist())
                    st.warning(f"⚠️ Já havia pagamento no mês para: {nomes}. Não foram duplicados.")
                if inserted or already:
                    invalidate_all_cache()
                    st.rerun()
            except SQLAlchemyError as e:
                st.error(f"Erro: {e}")

    # Lista do mês (visualização / exclusão)
    st.subheader("Pagamentos do mês")
    month = st.text_input("Mês (AAAA-MM)", value=this_month_ref(), key="list_pay_month")
    with engine.begin() as conn:
        q = select(payment, student.c.name.label("aluno")).join(student, student.c.id==payment.c.student_id).where(payment.c.month_ref==month)
        if filtro_coach != "(todos)":
            cid = next((c["id"] for c in coaches if c["name"]==filtro_coach), None)
            if cid:
                q = q.where(student.c.coach_id==cid)
        rows = conn.execute(q.order_by(payment.c.paid_date.desc(), payment.c.id.desc())).mappings().all()
    dfp = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[c.name for c in payment.c] + ["aluno"])
    if dfp.empty:
        st.info("Nenhum pagamento no período.")
    else:
        dfp["Data"] = pd.to_datetime(dfp["paid_date"]).dt.strftime("%d/%m/%Y")
        dfp["Valor (R$)"] = dfp["amount"].apply(fmt_money)
        dfp["Repasse (R$)"] = dfp["master_amount"].apply(fmt_money)
        show = dfp[["aluno","Data","month_ref","Valor (R$)","method","notes","Repasse (R$)"]].rename(
            columns={"aluno":"Aluno","month_ref":"Ref.","method":"Forma","notes":"Obs."}
        )
        st.dataframe(show, use_container_width=True, hide_index=True)

        c1, c2 = st.columns([1,1])
        with c1:
            ids_del = st.multiselect("Excluir pagamentos", options=dfp["id"].tolist(), format_func=lambda i: f"ID {i}")
            if st.button("🗑️ Excluir selecionados"):
                try:
                    with engine.begin() as conn:
                        if ids_del:
                            conn.execute(delete(payment).where(payment.c.id.in_(ids_del)))
                    invalidate_all_cache()
                    st.success("Excluído(s).")
                    st.rerun()
                except SQLAlchemyError as e:
                    st.error(f"Erro ao excluir: {e}")
        with c2:
            if st.button('🧹 Excluir TODOS deste mês'):
                try:
                    with engine.begin() as conn:
                        conn.execute(delete(payment).where(payment.c.month_ref==month))
                    invalidate_all_cache()
                    st.success("Pagamentos do mês excluídos.")
                    st.rerun()
                except SQLAlchemyError as e:
                    st.error(f"Erro ao excluir: {e}")

def page_extras(role: str):
    st.header("Extras (Repasse)")
    with st.form("form_extra"):
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            dt = st.date_input(
                "Data", value=clamp_date(TODAY, START_MIN, GRADE_MAX),
                min_value=START_MIN, max_value=GRADE_MAX, format="DD/MM/YYYY",
            )
        with col2:
            month_ref = st.text_input("Mês de referência (AAAA-MM)", value=this_month_ref())
        with col3:
            desc = st.text_input("Descrição")
        col4, col5, col6 = st.columns([1,1,2])
        with col4:
            val = st.number_input("Valor do extra (R$) (negativo para desconto)", step=10.0, value=0.0)
        with col5:
            is_rec = st.checkbox("Recorrente mês a mês?", value=False)
        with col6:
            df = fetch_students_df()
            opt = ["(sem aluno vinculado)"] + [f"{n} (ID {int(i)})" for i,n in zip(df["id"], df["name"])]
            pick = st.selectbox("Vincular a um aluno (opcional)", opt)
        ok = st.form_submit_button("➕ Adicionar extra")
    if ok:
        try:
            with engine.begin() as conn:
                sid = None
                if pick != "(sem aluno vinculado)":
                    sid = int(pick.split("ID")[-1].strip(" )"))
                conn.execute(insert(extra_repasse).values(
                    date=dt, month_ref=month_ref, description=desc.strip(),
                    amount=float(val), is_recurring=bool(is_rec), student_id=sid
                ))
            st.success("✅ Extra adicionado.")
            st.rerun()
        except SQLAlchemyError as e:
            st.error(f"Erro ao salvar: {e}")

    st.subheader("Lista de extras por mês")
    m = st.text_input("Mês (AAAA-MM)", value=this_month_ref(), key="list_extra_month")
    with engine.begin() as conn:
        rows = conn.execute(select(extra_repasse).where(extra_repasse.c.month_ref==m)
                            .order_by(extra_repasse.c.date.desc(), extra_repasse.c.id.desc())).mappings().all()
    dfe = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[c.name for c in extra_repasse.c])
    if dfe.empty:
        st.info("Nenhum extra para o mês.")
    else:
        dfe["Data"] = pd.to_datetime(dfe["date"]).dt.strftime("%d/%m/%Y")
        dfe["Valor (R$)"] = dfe["amount"].apply(fmt_money)
        st.dataframe(dfe[["Data","month_ref","description","Valor (R$)","is_recurring","student_id"]]
                     .rename(columns={"month_ref":"Ref.","description":"Descrição","is_recurring":"Recorrente?","student_id":"Aluno ID"}),
                     use_container_width=True, hide_index=True)

        ids_del = st.multiselect("Excluir extras", options=dfe["id"].tolist(), format_func=lambda i: f"ID {i}")
        if st.button("🗑️ Excluir selecionados"):
            try:
                with engine.begin() as conn:
                    if ids_del:
                        conn.execute(delete(extra_repasse).where(extra_repasse.c.id.in_(ids_del)))
                st.success("Excluído(s).")
                st.rerun()
            except SQLAlchemyError as e:
                st.error(f"Erro ao excluir: {e}")

def page_relatorios(role: str):
    st.header("Relatórios")
    coaches = fetch_coaches()
    cfilter = st.selectbox("Professor (opcional)", ["(todos)"] + [c["name"] for c in coaches])
    month = st.text_input("Mês (AAAA-MM)", value=this_month_ref())

    with engine.begin() as conn:
        q = (select(payment, student.c.name.label("aluno"),
                    student.c.birth_date, student.c.start_date, student.c.grade)
             .join(student, student.c.id==payment.c.student_id)
             .where(payment.c.month_ref==month))
        if cfilter != "(todos)":
            cid = next((c["id"] for c in coaches if c["name"]==cfilter), None)
            if cid:
                q = q.where(student.c.coach_id==cid)
        p_rows = conn.execute(q.order_by(student.c.name)).mappings().all()
        e_rows = conn.execute(select(extra_repasse).where(extra_repasse.c.month_ref==month)
                              .order_by(extra_repasse.c.date)).mappings().all()

    pag = pd.DataFrame(p_rows) if p_rows else pd.DataFrame(columns=[c.name for c in payment.c] + ["aluno","birth_date","start_date","grade"])
    ext = pd.DataFrame(e_rows) if e_rows else pd.DataFrame(columns=[c.name for c in extra_repasse.c])

    st.subheader("Mensalidades (alunos)")
    if pag.empty:
        st.info("Sem pagamentos no mês.")
        total_rep = 0.0
    else:
        def idade(dt: Optional[date]) -> str:
            if not dt:
                return "-"
            today = TODAY
            y = today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day))
            return f"{y} anos"

        def tempo_treino(sd: Optional[date]) -> str:
            if not sd:
                return "-"
            total_months = (TODAY.year - sd.year) * 12 + (TODAY.month - sd.month)
            if total_months < 12:
                return f"{total_months} meses"
            anos = total_months // 12
            meses = total_months % 12
            return f"{anos} anos e {meses} meses" if meses else f"{anos} anos"

        pag["Idade"] = pag["birth_date"].apply(idade)
        pag["Tempo de treino"] = pag["start_date"].apply(tempo_treino)
        pag["Valor (R$)"] = pag["amount"].apply(fmt_money)
        pag["Repasse (R$)"] = pag["master_amount"].apply(fmt_money)
        show = pag[["aluno","Idade","Tempo de treino","grade","paid_date","month_ref","Valor (R$)","method","notes","Repasse (R$)"]].rename(
            columns={"aluno":"Aluno","grade":"Graduação","paid_date":"Data","month_ref":"Ref.","method":"Forma","notes":"Obs."}
        )
        show["Data"] = pd.to_datetime(show["Data"]).dt.strftime("%d/%m/%Y")
        st.dataframe(show, use_container_width=True, hide_index=True)
        total_rep = float(pag["master_amount"].astype(float).sum())
        st.metric("Total de repasse (mensalidades)", fmt_money(total_rep))

    st.subheader("Extras (detalhado)")
    if ext.empty:
        st.info("Sem extras no mês.")
        total_ext = 0.0
    else:
        ext["Data"] = pd.to_datetime(ext["date"]).dt.strftime("%d/%m/%Y")
        ext["Valor (R$)"] = ext["amount"].apply(fmt_money)
        st.dataframe(ext[["date","description","amount","is_recurring","student_id","month_ref"]]
                     .rename(columns={"date":"Data","description":"Descrição","amount":"Valor (R$)","is_recurring":"Recorrente?","student_id":"Aluno ID","month_ref":"Ref."}),
                     use_container_width=True, hide_index=True)
        total_ext = float(ext["amount"].astype(float).sum())
        st.metric("Total (extras)", fmt_money(total_ext))

    st.subheader("Total geral (repasse + extras)")
    total_geral = total_rep + total_ext
    st.metric("Total geral (a repassar)", fmt_money(total_geral))

def page_config(role: str):
    st.header("Configurações")
    if role != "admin":
        st.warning("Apenas administradores podem acessar.")
        return

    cfg = fetch_settings()
    with st.form("form_cfg"):
        pct_ui = st.number_input("Percentual padrão de repasse (%)", min_value=0, max_value=100,
                                 value=fmt_pct_decimal_to_ui(cfg.get("master_percent")), step=5)
        ok = st.form_submit_button("💾 Salvar configuração")
    if ok:
        try:
            with engine.begin() as conn:
                conn.execute(update(settings).where(settings.c.id==1)
                             .values(master_percent=pct_ui_to_decimal(pct_ui)))
            invalidate_all_cache()
            st.success("✅ Configuração atualizada.")
        except SQLAlchemyError as e:
            st.error(f"Erro ao salvar configurações: {e}")

    st.subheader("Professores")
    with st.form("form_coach"):
        col1, col2 = st.columns([3,1])
        with col1:
            c_name = st.text_input("Nome do professor")
        with col2:
            c_full = st.checkbox("Repasse 100% (full pass)?", value=False)
        ok1 = st.form_submit_button("➕ Adicionar professor")
    if ok1 and c_name.strip():
        try:
            with engine.begin() as conn:
                conn.execute(insert(coach).values(name=c_name.strip(), full_pass=bool(c_full)))
            invalidate_all_cache()
            st.success("✅ Professor adicionado.")
            st.rerun()
        except SQLAlchemyError as e:
            st.error(f"Erro: {e}")

    dfc = pd.DataFrame(fetch_coaches())
    if not dfc.empty:
        dfc["Full pass?"] = dfc["full_pass"].map({True:"Sim", False:"Não"})
        st.dataframe(dfc[["name","Full pass?"]].rename(columns={"name":"Nome"}),
                     use_container_width=True, hide_index=True)

    st.subheader("Horários (treinos)")
    with st.form("form_slot"):
        s_name = st.text_input("Descrição do horário (ex.: Seg/Qua 19h)")
        ok2 = st.form_submit_button("➕ Adicionar horário")
    if ok2 and s_name.strip():
        try:
            with engine.begin() as conn:
                conn.execute(insert(train_slot).values(name=s_name.strip()))
            invalidate_all_cache()
            st.success("✅ Horário adicionado.")
            st.rerun()
        except SQLAlchemyError as e:
            st.error(f"Erro: {e}")

    dfs = pd.DataFrame(fetch_slots())
    if not dfs.empty:
        st.dataframe(dfs.rename(columns={"name":"Horário"}),
                     use_container_width=True, hide_index=True)

# ============================================================
# Main
# ============================================================
def main():
    role = require_login()
    with st.sidebar:
        st.markdown("### Navegação")
        page = st.radio(
            "Ir para:",
            ["Alunos","Graduações","Receber Pagamento","Extras (Repasse)","Relatórios","Configurações"],
            index=0,
            label_visibility="collapsed"
        )
        if st.button("🚪 Sair"):
            st.session_state.clear()
            st.rerun()

    menu = {
        "Alunos": page_alunos,
        "Graduações": page_graduacoes,
        "Receber Pagamento": page_receber,
        "Extras (Repasse)": page_extras,
        "Relatórios": page_relatorios,
        "Configurações": page_config,
    }
    menu[page](role)

if __name__ == "__main__":
    main()
