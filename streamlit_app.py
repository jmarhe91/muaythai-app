# streamlit_app.py
# ============================================================
# JAT - Gestão de alunos
# ============================================================
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Optional, Any, List, Dict, Set

import pandas as pd
import streamlit as st
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Date, Boolean, Text,
    Numeric, ForeignKey, select, insert, update, delete, func, Index, inspect, text,
)
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError

# -------------------- Aparência --------------------
icon = "logo.png" if os.path.exists("logo.png") else "🥊"
st.set_page_config(page_title="JAT - Gestão de alunos", page_icon=icon, layout="wide")
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=120)
st.title("JAT - Gestão de alunos")

# -------------------- Login ------------------------
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
                st.session_state["role"] = "admin"; st.success("Login como administrador."); st.rerun()
            elif perfil == "operador" and pwd == OPERATOR_PWD:
                st.session_state["role"] = "operador"; st.success("Login como operador."); st.rerun()
            else:
                st.error("Credenciais inválidas.")
        st.stop()
    return "operador"

# -------------------- Datas ------------------------
TODAY = date.today()
# faixas amplas para evitar erros de range
BIRTH_MIN, BIRTH_MAX = date(1930,1,1), date(2100,12,31)
START_MIN, START_MAX = date(2000,1,1), date(2100,12,31)
GRADE_MIN, GRADE_MAX = START_MIN, date(2100,12,31)

def first_day_same_month_years_ago(d: date, years: int) -> date:
    y = max(1900, d.year - years)
    return date(y, d.month, 1)

PAY_MIN, PAY_MAX = first_day_same_month_years_ago(TODAY, 50), date(2100,12,31)

def clamp_date(d: Optional[date], min_d: date, max_d: date) -> date:
    if d is None: return min_d
    if d < min_d: return min_d
    if d > max_d: return max_d
    return d

def this_month_ref(d: date = TODAY) -> str:
    return d.strftime("%Y-%m")

# -------------------- Banco ------------------------
DB_URL = load_secret("DATABASE_URL") or os.getenv("DATABASE_URL") or "sqlite:///muaythai.db"

@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    return create_engine(DB_URL, pool_pre_ping=True, future=True)
engine = get_engine()
metadata = MetaData()

# -------------------- Esquema ----------------------
settings = Table(
    "settings", metadata,
    Column("id", Integer, primary_key=True),
    Column("master_percent", Numeric(5,4), nullable=False, server_default="0.60"),
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
    Column("monthly_fee", Numeric(12,2), nullable=False, server_default="0"),
    Column("active", Boolean, nullable=False, server_default="true"),
    Column("coach_id", Integer, ForeignKey("coach.id"), nullable=True),
    Column("train_slot_id", Integer, ForeignKey("train_slot.id"), nullable=True),
    Column("master_percent_override", Numeric(5,4), nullable=True),
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
    Column("month_ref", String(7), nullable=False),
    Column("amount", Numeric(12,2), nullable=False),
    Column("method", String(20), nullable=True),
    Column("notes", String(255), nullable=True),
    Column("master_percent_used", Numeric(5,4), nullable=False, server_default="0"),
    Column("master_adjustment", Numeric(12,2), nullable=False, server_default="0"),
    Column("master_amount", Numeric(12,2), nullable=False, server_default="0"),
)
Index("ix_payment_student_id", payment.c.student_id)
Index("ix_payment_month_ref", payment.c.month_ref)

extra_repasse = Table(
    "extra_repasse", metadata,
    Column("id", Integer, primary_key=True),
    Column("date", Date, nullable=False, server_default=func.current_date()),
    Column("month_ref", String(7), nullable=False),
    Column("description", Text, nullable=False),
    Column("amount", Numeric(12,2), nullable=False),  # pode ser negativo
    Column("is_recurring", Boolean, nullable=False, server_default="false"),
    Column("student_id", Integer, ForeignKey("student.id"), nullable=True),
    Column("coach_id", Integer, ForeignKey("coach.id"), nullable=True),
    Column("created_at", Date, nullable=False, server_default=func.current_date()),
)
Index("ix_extra_repasse_month_ref", extra_repasse.c.month_ref)
Index("ix_extra_repasse_student_id", extra_repasse.c.student_id)

BELTS = [
    "Branca","Amarelo","Amarelo e Branca","Verde","Verde e Branca",
    "Azul","Azul e Branca","Marrom","Marrom e Branca","Vermelha",
    "Vermelha e Branca","Preta",
]

# -------------------- Bootstrap/Migrações ------------
REQUIRED_TABLES = {"settings","coach","train_slot","student","graduation","payment","extra_repasse"}

def bootstrap_db_if_needed() -> None:
    try:
        insp = inspect(engine)
        existing = set(insp.get_table_names())
        if not REQUIRED_TABLES.issubset(existing):
            metadata.create_all(engine)
        with engine.begin() as conn:
            row = conn.execute(select(settings.c.id).where(settings.c.id==1)).first()
            if not row:
                conn.execute(insert(settings).values(id=1, master_percent=0.60))
    except SQLAlchemyError:
        pass

def ensure_extra_has_coach():
    """Garante coluna/índices em extra_repasse (idempotente)."""
    try:
        insp = inspect(engine)
        cols = [c["name"] for c in insp.get_columns("extra_repasse")]
        if "coach_id" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE extra_repasse ADD COLUMN coach_id INTEGER NULL"))

        with engine.begin() as conn:
            # índices úteis para os filtros
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_extra_repasse_coach_id ON extra_repasse (coach_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_extra_repasse_month_ref ON extra_repasse (month_ref)"))
            # índice para recorrência + mês (melhora muito as consultas)
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_extra_repasse_rec_mon ON extra_repasse (is_recurring, month_ref)"))
    except Exception:
        pass

bootstrap_db_if_needed()
ensure_extra_has_coach()

# -------------------- Helpers/Cache -------------------
def fmt_money(v: Any) -> str:
    try:
        x = float(v or 0)
    except Exception:
        x = 0.0
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_pct_decimal_to_ui(d: Optional[float]) -> int:
    if d is None: return 0
    return int(round(float(d) * 100))

def pct_ui_to_decimal(p: Optional[int]) -> Optional[float]:
    if p is None: return None
    return round(p / 100.0, 4)

@st.cache_data(show_spinner=False)
def fetch_settings() -> Dict[str, Any]:
    with engine.begin() as conn:
        row = conn.execute(select(settings)).mappings().first()
        return dict(row) if row else {"id":1,"master_percent":0.60}

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
    try:
        with engine.begin() as conn:
            rows = conn.execute(select(student).order_by(student.c.name)).mappings().all()
            return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[c.name for c in student.c])
    except ProgrammingError:
        bootstrap_db_if_needed()
        with engine.begin() as conn:
            rows = conn.execute(select(student).order_by(student.c.name)).mappings().all()
            return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[c.name for c in student.c])

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
        c = conn.execute(select(coach).where(coach.c.id==cid)).mappings().first()
        if c and bool(c["full_pass"]):
            return 1.0
    if stu_row.get("master_percent_override") is not None:
        percent = float(stu_row["master_percent_override"])
    if percent is None:
        cfg = conn.execute(select(settings)).mappings().first()
        percent = float(cfg["master_percent"]) if cfg and cfg.get("master_percent") is not None else 0.6
    return percent

# ===== Atualização em massa de mensalidades =====
def update_monthly_fee_bulk(engine: Engine, student_ids: List[int], new_fee: float) -> int:
    """Atualiza mensalidade para todos os IDs informados. Retorna quantidade alterada."""
    if not student_ids:
        return 0
    with engine.begin() as conn:
        res = conn.execute(
            update(student).where(student.c.id.in_(student_ids)).values(monthly_fee=float(new_fee))
        )
        return res.rowcount or 0
# ======================================================

# -------------------- Páginas ------------------------
def page_alunos(role: str):
    st.header("Alunos")
    month_for_status = st.text_input("Mês p/ status (AAAA-MM)", value=this_month_ref())

    df = fetch_students_df()
    if df.empty:
        st.info("Nenhum aluno cadastrado.")
    else:
        paid_ids = get_paid_ids_for_month(month_for_status)
        view = df.copy()
        view["Mensalidade"] = view["monthly_fee"].apply(fmt_money)
        view["Ativo"] = view["active"].map({True:"Sim", False:"Não"})
        view["Repasse % (aluno)"] = view["master_percent_override"].apply(
            lambda v: f"{fmt_pct_decimal_to_ui(float(v))}%" if v is not None else "-"
        )
        view["Nasc."] = pd.to_datetime(view["birth_date"]).dt.strftime("%d/%m/%Y").fillna("")
        view["Início"] = pd.to_datetime(view["start_date"]).dt.strftime("%d/%m/%Y").fillna("")
        view["Graduação"] = view["grade"].fillna("-")
        view["Data grad."] = pd.to_datetime(view["grade_date"]).dt.strftime("%d/%m/%Y").fillna("")
        view["Status"] = view["id"].apply(lambda i: "🟢 Pago" if int(i) in paid_ids else "🔴 Pendente")

        cols = ["name","Status","Nasc.","Início","Mensalidade","Ativo","Graduação","Data grad.","Repasse % (aluno)"]
        st.dataframe(view[cols].rename(columns={"name":"Nome"}), use_container_width=True, hide_index=True)

    # ===== Alterar mensalidades em massa =====
    with st.expander("💰 Alterar mensalidades (em massa)", expanded=False):
        df_all = fetch_students_df()
        if df_all.empty:
            st.info("Sem alunos para alterar.")
        else:
            coaches = fetch_coaches()
            with st.form("form_bulk_fee"):
                c1, c2, c3 = st.columns([1,1,2])
                with c1:
                    only_active = st.checkbox("Somente ativos", value=True)
                with c2:
                    coach_filter = st.selectbox(
                        "Professor (opcional)",
                        ["(todos)"] + [c["name"] for c in coaches]
                    )
                with c3:
                    new_fee = st.number_input("Novo valor (R$)", min_value=0.0, step=10.0, format="%.2f")

                df_sel = df_all.copy()
                if only_active:
                    df_sel = df_sel[df_sel["active"] == True]
                if coach_filter != "(todos)":
                    cid = next((c["id"] for c in coaches if c["name"] == coach_filter), None)
                    if cid is not None:
                        df_sel = df_sel[df_sel["coach_id"] == cid]

                options_map = {
                    f"{row['name']} — {fmt_money(row['monthly_fee'])}": int(row["id"])
                    for _, row in df_sel.sort_values("name").iterrows()
                }

                sel = st.multiselect(
                    "Selecione os alunos",
                    options=list(options_map.keys()),
                    placeholder="Digite para filtrar por nome…"
                )

                # Sempre habilitado; validação após o clique
                submit_bulk = st.form_submit_button("✅ Confirmar alteração", type="primary")

            if submit_bulk:
                if not sel:
                    st.warning("Selecione ao menos um aluno.")
                else:
                    ids = [options_map[s] for s in sel]
                    n = update_monthly_fee_bulk(engine, ids, new_fee)
                    invalidate_all_cache()
                    st.success(f"Mensalidade atualizada para **{n}** aluno(s) → {fmt_money(new_fee)}")
                    df_ok = fetch_students_df()
                    df_ok = df_ok[df_ok["id"].isin(ids)][["name","monthly_fee"]].rename(
                        columns={"name": "Aluno", "monthly_fee": "Nova mensalidade (R$)"}
                    )
                    if not df_ok.empty:
                        df_ok["Nova mensalidade (R$)"] = df_ok["Nova mensalidade (R$)"].apply(fmt_money)
                        st.dataframe(df_ok, use_container_width=True, hide_index=True)
    # ===== Fim bloco =====

    # Cadastrar
    with st.expander("➕ Cadastrar novo aluno", expanded=False):
        with st.form("form_new_student"):
            col1, col2, col3 = st.columns(3)
            with col1:
                n_name = st.text_input("Nome*", max_chars=200)
                n_birth = st.date_input("Data de nascimento",
                                        value=clamp_date(date(2000,1,1), BIRTH_MIN, BIRTH_MAX),
                                        min_value=BIRTH_MIN, max_value=BIRTH_MAX, format="DD/MM/YYYY")
            with col2:
                n_start = st.date_input("Início do treino",
                                        value=clamp_date(TODAY, START_MIN, START_MAX),
                                        min_value=START_MIN, max_value=START_MAX, format="DD/MM/YYYY")
                n_fee = st.number_input("Mensalidade (R$)", min_value=0.0, step=10.0, value=0.0)
            with col3:
                coaches = fetch_coaches()
                slots = fetch_slots()
                opt_coach = ["(nenhum)"] + [c["name"] for c in coaches]
                opt_slot  = ["(nenhum)"] + [s["name"] for s in slots]
                n_coach_name = st.selectbox("Professor responsável", opt_coach)
                n_slot_name  = st.selectbox("Horário de treino", opt_slot)
                n_override_pct = st.number_input("Repasse do aluno (%) (0 = usar padrão)", min_value=0, max_value=100, value=0, step=5)
                n_active = st.checkbox("Ativo?", value=True)
            sub = st.form_submit_button("💾 Salvar aluno")
        if sub:
            if not n_name.strip():
                st.error("Informe o nome.")
            else:
                try:
                    with engine.begin() as conn:
                        coach_id, slot_id = None, None
                        if n_coach_name != "(nenhum)":
                            r = conn.execute(select(coach.c.id).where(coach.c.name==n_coach_name)).first()
                            coach_id = r[0] if r else None
                        if n_slot_name != "(nenhum)":
                            r = conn.execute(select(train_slot.c.id).where(train_slot.c.name==n_slot_name)).first()
                            slot_id = r[0] if r else None
                        res = conn.execute(insert(student).values(
                            name=n_name.strip(), birth_date=n_birth, start_date=n_start,
                            monthly_fee=float(n_fee or 0), active=bool(n_active),
                            coach_id=coach_id, train_slot_id=slot_id,
                            master_percent_override=pct_ui_to_decimal(n_override_pct) if n_override_pct else None,
                        ))
                        stu_id = res.inserted_primary_key[0]
                        # graduação inicial: Branca com data de início do treino
                        conn.execute(insert(graduation).values(student_id=stu_id, grade="Branca", grade_date=(n_start or TODAY)))
                        conn.execute(update(student).where(student.c.id==stu_id).values(grade="Branca", grade_date=(n_start or TODAY)))
                    invalidate_all_cache()
                    st.success("✅ Aluno cadastrado com sucesso."); st.rerun()
                except SQLAlchemyError as e:
                    st.error(f"Erro ao salvar: {e}")

    # Editar
    with st.expander("✏️ Editar aluno", expanded=False):
        df = fetch_students_df()
        if df.empty:
            st.info("Sem alunos.")
        else:
            sid = st.selectbox("Selecionar aluno", options=df["id"].tolist(),
                               format_func=lambda i: df.loc[df["id"]==i,"name"].values[0])
            row = df.loc[df["id"]==sid].iloc[0].to_dict()
            with st.form("form_edit_student"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    e_name = st.text_input("Nome*", value=row["name"])
                    e_birth = st.date_input("Data de nascimento",
                                            value=clamp_date(row.get("birth_date"), BIRTH_MIN, BIRTH_MAX),
                                            min_value=BIRTH_MIN, max_value=BIRTH_MAX, format="DD/MM/YYYY")
                with col2:
                    e_start = st.date_input("Início do treino",
                                            value=clamp_date(row.get("start_date"), START_MIN, START_MAX),
                                            min_value=START_MIN, max_value=START_MAX, format="DD/MM/YYYY")
                    e_fee = st.number_input("Mensalidade (R$)", min_value=0.0, step=5.0,
                                            value=float(row.get("monthly_fee") or 0))
                with col3:
                    coaches = fetch_coaches(); slots = fetch_slots()
                    opt_coach = ["(nenhum)"] + [c["name"] for c in coaches]
                    opt_slot  = ["(nenhum)"] + [s["name"] for s in slots]
                    current_coach = next((c["name"] for c in coaches if c["id"]==row.get("coach_id")), "(nenhum)")
                    current_slot  = next((s["name"] for s in slots  if s["id"]==row.get("train_slot_id")), "(nenhum)")
                    e_coach_name = st.selectbox("Professor responsável", opt_coach, index=opt_coach.index(current_coach))
                    e_slot_name  = st.selectbox("Horário de treino",    opt_slot,  index=opt_slot.index(current_slot))
                    e_override_pct = st.number_input("Repasse do aluno (%) (0 = usar padrão)", min_value=0, max_value=100,
                                                     value=fmt_pct_decimal_to_ui(row.get("master_percent_override")) if row.get("master_percent_override") is not None else 0,
                                                     step=5)
                    e_active = st.checkbox("Ativo?", value=bool(row.get("active")))
                ok = st.form_submit_button("💾 Salvar alterações")
            if ok:
                try:
                    with engine.begin() as conn:
                        coach_id, slot_id = None, None
                        if e_coach_name != "(nenhum)":
                            r = conn.execute(select(coach.c.id).where(coach.c.name==e_coach_name)).first()
                            coach_id = r[0] if r else None
                        if e_slot_name != "(nenhum)":
                            r = conn.execute(select(train_slot.c.id).where(train_slot.c.name==e_slot_name)).first()
                            slot_id = r[0] if r else None
                        conn.execute(update(student).where(student.c.id==sid).values(
                            name=e_name.strip(), birth_date=e_birth, start_date=e_start,
                            monthly_fee=float(e_fee or 0), active=bool(e_active),
                            coach_id=coach_id, train_slot_id=slot_id,
                            master_percent_override=pct_ui_to_decimal(e_override_pct) if e_override_pct else None,
                        ))
                    invalidate_all_cache(); st.success("✅ Dados atualizados."); st.rerun()
                except SQLAlchemyError as e:
                    st.error(f"Erro ao atualizar: {e}")

    # Excluir
    with st.expander("🗑️ Excluir aluno", expanded=False):
        df = fetch_students_df()
        if df.empty:
            st.info("Sem alunos.")
        else:
            sid_del = st.selectbox("Escolha o aluno", options=df["id"].tolist(),
                                   format_func=lambda i: df.loc[df["id"]==i,"name"].values[0])
            st.warning("Isto remove aluno, pagamentos e graduações.")
            colx, coly = st.columns([1,3])
            with colx:
                confirm = st.checkbox("Confirmo a exclusão")
            with coly:
                if st.button("🗑️ Excluir aluno (definitivo)", disabled=not confirm):
                    try:
                        with engine.begin() as conn:
                            conn.execute(delete(payment).where(payment.c.student_id==sid_del))
                            conn.execute(delete(graduation).where(graduation.c.student_id==sid_del))
                            conn.execute(delete(student).where(student.c.id==sid_del))
                        invalidate_all_cache(); st.success("Aluno excluído."); st.rerun()
                    except SQLAlchemyError as e:
                        st.error(f"Erro ao excluir: {e}")

def page_graduacoes(role: str):
    st.header("Graduações")
    df = fetch_students_df()
    if df.empty:
        st.info("Cadastre alunos primeiro."); return

    sid = st.selectbox("Aluno", options=df["id"].tolist(),
                       format_func=lambda i: df.loc[df["id"]==i,"name"].values[0])
    stu = df.loc[df["id"]==sid].iloc[0].to_dict()

    # Histórico
    with engine.begin() as conn:
        rows = conn.execute(select(graduation)
                            .where(graduation.c.student_id==sid)
                            .order_by(graduation.c.grade_date.desc(), graduation.c.id.desc())).mappings().all()
    hist = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[c.name for c in graduation.c])

    if not hist.empty:
        hist["Data"] = pd.to_datetime(hist["grade_date"]).dt.strftime("%d/%m/%Y")
        st.subheader("Histórico")
        st.dataframe(hist[["grade","Data","notes"]].rename(columns={"grade":"Graduação","notes":"Observações"}),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Sem graduações registradas para este aluno.")

    # Adicionar
    with st.form("form_add_grade"):
        col1,col2 = st.columns([2,1])
        with col1:
            g_grade = st.selectbox("Graduação", BELTS, index=BELTS.index("Branca"))
        with col2:
            min_g = max(GRADE_MIN, stu.get("start_date") or GRADE_MIN)
            g_date = st.date_input("Data da graduação",
                                   value=clamp_date(TODAY, min_g, GRADE_MAX),
                                   min_value=min_g, max_value=GRADE_MAX, format="DD/MM/YYYY")
        g_notes = st.text_input("Observações (opcional)")
        ok_add = st.form_submit_button("💾 Salvar graduação")
    if ok_add:
        try:
            with engine.begin() as conn:
                conn.execute(insert(graduation).values(student_id=sid, grade=g_grade, grade_date=g_date, notes=g_notes or None))
                last = conn.execute(select(graduation.c.grade, graduation.c.grade_date)
                                    .where(graduation.c.student_id==sid)
                                    .order_by(graduation.c.grade_date.desc(), graduation.c.id.desc())
                                    .limit(1)).mappings().first()
                if last:
                    conn.execute(update(student).where(student.c.id==sid).values(grade=last["grade"], grade_date=last["grade_date"]))
            invalidate_all_cache(); st.success("✅ Graduação registrada."); st.rerun()
        except SQLAlchemyError as e:
            st.error(f"Erro ao salvar graduação: {e}")

    # Excluir
    st.subheader("Excluir graduação")
    if hist.empty:
        st.caption("Nada para excluir.")
    else:
        def _label_for(gid: int) -> str:
            row = hist.loc[hist["id"]==gid].iloc[0]
            data_str = pd.to_datetime(row["grade_date"]).strftime("%d/%m/%Y")
            obs = str(row["notes"]) if pd.notna(row.get("notes")) and row.get("notes") not in [None,""] else ""
            return f"{data_str} — {row['grade']}" + (f" — {obs}" if obs else "")
        gid_to_delete = st.selectbox("Selecione a graduação para excluir",
                                     options=hist["id"].tolist(), format_func=_label_for)
        colx, coly = st.columns([1,3])
        with colx:
            confirm_del = st.checkbox("Confirmo a exclusão")
        with coly:
            if st.button("🗑️ Excluir graduação", disabled=not confirm_del):
                try:
                    with engine.begin() as conn:
                        conn.execute(delete(graduation).where(graduation.c.id==gid_to_delete))
                        last = conn.execute(select(graduation.c.grade, graduation.c.grade_date)
                                            .where(graduation.c.student_id==sid)
                                            .order_by(graduation.c.grade_date.desc(), graduation.c.id.desc())
                                            .limit(1)).mappings().first()
                        if last:
                            conn.execute(update(student).where(student.c.id==sid)
                                         .values(grade=last["grade"], grade_date=last["grade_date"]))
                        else:
                            conn.execute(update(student).where(student.c.id==sid).values(grade=None, grade_date=None))
                    invalidate_all_cache(); st.success("✅ Graduação excluída."); st.rerun()
                except SQLAlchemyError as e:
                    st.error(f"Erro ao excluir graduação: {e}")

def page_receber(role: str):
    st.header("Receber Pagamento")
    df = fetch_students_df()
    if df.empty:
        st.info("Cadastre alunos primeiro."); return

    df_active = df[df["active"]==True].copy()
    coaches = fetch_coaches()

    with st.form("form_receive"):
        col1,col2,col3 = st.columns(3)
        with col1:
            filtro_coach = st.selectbox("Filtrar por professor", ["(todos)"] + [c["name"] for c in coaches])
        with col2:
            paid_date = st.date_input("Data do pagamento", value=clamp_date(TODAY, PAY_MIN, PAY_MAX),
                                      min_value=PAY_MIN, max_value=PAY_MAX, format="DD/MM/YYYY")
        with col3:
            month_ref = st.text_input("Mês de referência (AAAA-MM)", value=this_month_ref())

        if filtro_coach != "(todos)":
            cid = next((c["id"] for c in coaches if c["name"]==filtro_coach), None)
            df_active = df_active[df_active["coach_id"]==cid]

        status_filter = st.radio("Status", ["Pendentes","Pagos","Todos"], horizontal=True, index=0)
        paid_ids = get_paid_ids_for_month(month_ref)

        if status_filter == "Pendentes":
            df_list = df_active[~df_active["id"].isin(paid_ids)].copy()
        elif status_filter == "Pagos":
            df_list = df_active[df_active["id"].isin(paid_ids)].copy()
        else:
            df_list = df_active.copy()

        selectable_ids = df_list[~df_list["id"].isin(paid_ids)]["id"].tolist()
        sel = st.multiselect(
            "Selecione os alunos que pagaram (somente pendentes aparecem aqui)",
            options=selectable_ids,
            format_func=lambda i: df_active.loc[df_active["id"]==i,"name"].values[0] + \
                                  f" (R$ {float(df_active.loc[df_active['id']==i,'monthly_fee'].values[0]):.2f})"
        )
        method = st.selectbox("Forma", ["Dinheiro","PIX","Cartão","Transferência","Outro"])
        notes = st.text_input("Observações (opcional)")
        ok = st.form_submit_button("✅ Confirmar recebimento")

    if ok:
        if not sel:
            st.warning("Selecione pelo menos um aluno.")
        else:
            already, inserted = [], 0
            try:
                with engine.begin() as conn:
                    for sid in sel:
                        exists = conn.execute(select(payment.c.id).where(
                            (payment.c.student_id==sid) & (payment.c.month_ref==month_ref)
                        )).first()
                        if exists:
                            already.append(sid); continue
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
                if inserted: st.success(f"✅ {inserted} pagamento(s) registrado(s).")
                if already:
                    nomes = ", ".join(df.loc[df["id"].isin(already),"name"].tolist())
                    st.warning(f"⚠️ Já havia pagamento no mês para: {nomes}.")
                if inserted or already:
                    invalidate_all_cache(); st.rerun()
            except SQLAlchemyError as e:
                st.error(f"Erro: {e}")

    # Lista do mês
    st.subheader("Pagamentos do mês")
    month = st.text_input("Mês (AAAA-MM)", value=this_month_ref(), key="list_pay_month")
    with engine.begin() as conn:
        q = select(payment, student.c.name.label("aluno")).join(student, student.c.id==payment.c.student_id)\
             .where(payment.c.month_ref==month)\
             .order_by(payment.c.paid_date.desc(), payment.c.id.desc())
        if filtro_coach != "(todos)":
            cid = next((c["id"] for c in coaches if c["name"]==filtro_coach), None)
            if cid: q = q.where(student.c.coach_id==cid)
        rows = conn.execute(q).mappings().all()
    dfp = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[c.name for c in payment.c] + ["aluno"])
    if dfp.empty:
        st.info("Nenhum pagamento no período.")
    else:
        dfp["Data"] = pd.to_datetime(dfp["paid_date"]).dt.strftime("%d/%m/%Y")
        dfp["Valor (R$)"] = dfp["amount"].apply(fmt_money)
        dfp["Repasse (R$)"] = dfp["master_amount"].apply(fmt_money)
        show = dfp[["aluno","Data","month_ref","Valor (R$)","method","notes","Repasse (R$)"]]\
                .rename(columns={"aluno":"Aluno","month_ref":"Ref.","method":"Forma","notes":"Obs."})
        st.dataframe(show, use_container_width=True, hide_index=True)

        # Exclusão mostrando nomes
        def label_pag_id(pid: int) -> str:
            r = dfp.loc[dfp["id"]==pid].iloc[0]
            return f"{r['aluno']} — {r['Data']} — {fmt_money(r['amount'])}"
        ids_del = st.multiselect("Excluir pagamentos (seleção por aluno/data/valor)",
                                 options=dfp["id"].tolist(), format_func=label_pag_id)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ Excluir selecionados"):
                try:
                    with engine.begin() as conn:
                        if ids_del:
                            conn.execute(delete(payment).where(payment.c.id.in_(ids_del)))
                    invalidate_all_cache(); st.success("Excluído(s)."); st.rerun()
                except SQLAlchemyError as e:
                    st.error(f"Erro ao excluir: {e}")
        with c2:
            if st.button("🧹 Excluir TODOS deste mês"):
                try:
                    with engine.begin() as conn:
                        conn.execute(delete(payment).where(payment.c.month_ref==month))
                    invalidate_all_cache(); st.success("Pagamentos do mês excluídos."); st.rerun()
                except SQLAlchemyError as e:
                    st.error(f"Erro ao excluir: {e}")

def page_extras(role: str):
    st.header("Extras (Repasse)")
    df_stu = fetch_students_df()
    coaches = fetch_coaches()

    with st.form("form_extra"):
        col1,col2,col3 = st.columns([1,1,2])
        with col1:
            dt = st.date_input("Data", value=clamp_date(TODAY, START_MIN, GRADE_MAX),
                               min_value=START_MIN, max_value=GRADE_MAX, format="DD/MM/YYYY")
        with col2:
            month_ref = st.text_input("Mês de referência (AAAA-MM)", value=this_month_ref())
        with col3:
            desc = st.text_input("Descrição")
        col4,col5,col6 = st.columns([1,1,2])
        with col4:
            val = st.number_input("Valor do extra (R$) (negativo para desconto)", step=10.0, value=0.0)
        with col5:
            is_rec = st.checkbox("Recorrente mês a mês?", value=False)
        with col6:
            opt_stu = ["(sem aluno)"] + [f"{n}" for n in df_stu["name"]] if not df_stu.empty else ["(sem aluno)"]
            pick_stu_name = st.selectbox("Vincular a um aluno (opcional)", opt_stu)

        col7, = st.columns(1)
        with col7:
            opt_coach = ["(sem professor)"] + [c["name"] for c in coaches]
            pick_coach_name = st.selectbox("Vincular a um professor (opcional)", opt_coach)

        ok = st.form_submit_button("➕ Adicionar extra")

    if ok:
        try:
            with engine.begin() as conn:
                sid = None
                if pick_stu_name != "(sem aluno)" and not df_stu.empty:
                    m = df_stu[df_stu["name"]==pick_stu_name]
                    if not m.empty:
                        sid = int(m.iloc[0]["id"])
                # Se professor não selecionado, herda do aluno (se houver)
                coach_id = None
                if pick_coach_name != "(sem professor)":
                    r = conn.execute(select(coach.c.id).where(coach.c.name==pick_coach_name)).first()
                    coach_id = r[0] if r else None
                elif sid is not None:
                    r = conn.execute(select(student.c.coach_id).where(student.c.id==sid)).first()
                    coach_id = r[0] if r and r[0] is not None else None

                ensure_extra_has_coach()
                conn.execute(insert(extra_repasse).values(
                    date=dt, month_ref=month_ref, description=desc.strip(),
                    amount=float(val), is_recurring=bool(is_rec), student_id=sid, coach_id=coach_id
                ))
            st.success("✅ Extra adicionado."); st.rerun()
        except SQLAlchemyError as e:
            st.error(f"Erro ao salvar: {e}")

    st.subheader("Lista de extras por mês")
    m = st.text_input("Mês (AAAA-MM)", value=this_month_ref(), key="list_extra_month")
    filtro_coach = st.selectbox("Filtrar por professor", ["(todos)"] + [c["name"] for c in coaches], key="extra_filter_coach")

    with engine.begin() as conn:
        # Itens do mês OU recorrentes cujo mês de origem <= mês selecionado
        cond_mes = (extra_repasse.c.month_ref == m) | (
            (extra_repasse.c.is_recurring == True) & (extra_repasse.c.month_ref <= m)
        )

        q_ext = (
            select(
                extra_repasse.c.id,
                extra_repasse.c.date,
                extra_repasse.c.month_ref,
                extra_repasse.c.description,
                extra_repasse.c.amount,
                extra_repasse.c.is_recurring,
                extra_repasse.c.student_id,
                extra_repasse.c.coach_id,
                student.c.name.label("aluno"),
                coach.c.name.label("professor_name"),
            )
            .select_from(
                extra_repasse
                .outerjoin(student, extra_repasse.c.student_id == student.c.id)
                .outerjoin(coach,   extra_repasse.c.coach_id   == coach.c.id)
            )
            .where(cond_mes)
            .order_by(extra_repasse.c.date.desc(), extra_repasse.c.id.desc())
        )
        if filtro_coach != "(todos)":
            cid = next((c["id"] for c in coaches if c["name"]==filtro_coach), None)
            if cid:
                q_ext = q_ext.where(extra_repasse.c.coach_id == cid)

        rows = conn.execute(q_ext).mappings().all()

    dfe = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "id","date","month_ref","description","amount","is_recurring","student_id","coach_id","aluno","professor_name"
    ])
    if dfe.empty:
        st.info("Nenhum extra para o mês.")
    else:
        dfe["Data"] = pd.to_datetime(dfe["date"]).dt.strftime("%d/%m/%Y")
        dfe["Valor (R$)"] = dfe["amount"].apply(fmt_money)
        tabela = dfe[["Data","month_ref","description","Valor (R$)","is_recurring","aluno","professor_name"]]\
            .rename(columns={"month_ref":"Ref.","description":"Descrição","is_recurring":"Recorrente?","aluno":"Aluno","professor_name":"Professor"})
        st.dataframe(tabela, use_container_width=True, hide_index=True)
        st.caption("Itens com 'Recorrente?' marcado aparecem automaticamente nos meses seguintes.")

        ids_del = st.multiselect(
            "Excluir extras",
            options=dfe["id"].tolist(),
            format_func=lambda i: f"{dfe.loc[dfe['id']==i,'Data'].values[0]} — {dfe.loc[dfe['id']==i,'description'].values[0]}"
        )
        if st.button("🗑️ Excluir selecionados"):
            try:
                with engine.begin() as conn:
                    if ids_del:
                        conn.execute(delete(extra_repasse).where(extra_repasse.c.id.in_(ids_del)))
                st.success("Excluído(s)."); st.rerun()
            except SQLAlchemyError as e:
                st.error(f"Erro ao excluir: {e}")

def page_relatorios(role: str):
    st.header("Relatórios")
    coaches = fetch_coaches()
    cfilter = st.selectbox("Professor (opcional)", ["(todos)"] + [c["name"] for c in coaches])
    month = st.text_input("Mês (AAAA-MM)", value=this_month_ref())

    with engine.begin() as conn:
        # Pagamentos
        q = (select(payment, student.c.name.label("aluno"),
                    student.c.birth_date, student.c.start_date, student.c.grade)
             .join(student, student.c.id==payment.c.student_id)
             .where(payment.c.month_ref==month))
        if cfilter != "(todos)":
            cid = next((c["id"] for c in coaches if c["name"]==cfilter), None)
            if cid: q = q.where(student.c.coach_id==cid)
        p_rows = conn.execute(q.order_by(student.c.name)).mappings().all()

        # Extras (inclui recorrentes cujo month_ref <= month)
        cond_mes = (extra_repasse.c.month_ref == month) | (
            (extra_repasse.c.is_recurring == True) & (extra_repasse.c.month_ref <= month)
        )
        q_extra = (
            select(
                extra_repasse.c.id,
                extra_repasse.c.date,
                extra_repasse.c.month_ref,
                extra_repasse.c.description,
                extra_repasse.c.amount,
                extra_repasse.c.is_recurring,
                extra_repasse.c.student_id,
                extra_repasse.c.coach_id,
                student.c.name.label("aluno"),
                coach.c.name.label("professor_name"),
            )
            .select_from(
                extra_repasse
                .outerjoin(student, extra_repasse.c.student_id == student.c.id)
                .outerjoin(coach,   extra_repasse.c.coach_id   == coach.c.id)
            )
            .where(cond_mes)
            .order_by(extra_repasse.c.date)
        )
        if cfilter != "(todos)":
            cid = next((c["id"] for c in coaches if c["name"] == cfilter), None)
            if cid:
                q_extra = q_extra.where(extra_repasse.c.coach_id == cid)

        e_rows = conn.execute(q_extra).mappings().all()

    pag = pd.DataFrame(p_rows) if p_rows else pd.DataFrame(columns=[c.name for c in payment.c] + ["aluno","birth_date","start_date","grade"])
    ext = pd.DataFrame(e_rows) if e_rows else pd.DataFrame(columns=[
        "id","date","month_ref","description","amount","is_recurring","student_id","coach_id","aluno","professor_name"
    ])

    # Mensalidades
    st.subheader("Mensalidades (alunos)")
    if pag.empty:
        st.info("Sem pagamentos no mês."); total_rep = 0.0
    else:
        def idade(dt: Optional[date]) -> str:
            if not dt or pd.isna(dt): return "-"
            if isinstance(dt, str):
                try: dt = pd.to_datetime(dt).date()
                except Exception: return "-"
            today = TODAY
            y = today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day))
            return f"{y} anos"
        def tempo_treino(sd: Optional[date]) -> str:
            if not sd or pd.isna(sd): return "-"
            if isinstance(sd, str):
                try: sd = pd.to_datetime(sd).date()
                except Exception: return "-"
            total_months = (TODAY.year - sd.year)*12 + (TODAY.month - sd.month)
            if total_months < 12: return f"{total_months} meses"
            anos, meses = divmod(total_months, 12)
            return f"{anos} anos e {meses} meses" if meses else f"{anos} anos"

        pag["Idade"] = pag["birth_date"].apply(idade)
        pag["Tempo de treino"] = pag["start_date"].apply(tempo_treino)
        pag["Valor (R$)"]   = pag["amount"].apply(fmt_money)
        pag["Repasse (R$)"] = pag["master_amount"].apply(fmt_money)
        show = pag[["aluno","Idade","Tempo de treino","grade","paid_date","month_ref","Valor (R$)","method","notes","Repasse (R$)"]]\
                .rename(columns={"aluno":"Aluno","grade":"Graduação","paid_date":"Data","month_ref":"Ref.","method":"Forma","notes":"Obs."})
        show["Data"] = pd.to_datetime(show["Data"]).dt.strftime("%d/%m/%Y")
        st.dataframe(show, use_container_width=True, hide_index=True)
        total_rep = float(pag["master_amount"].astype(float).sum())
        st.metric("Total de repasse (mensalidades)", fmt_money(total_rep))

    # Extras
    st.subheader("Extras (detalhado)")
    if ext.empty:
        st.info("Sem extras no mês."); total_ext = 0.0
    else:
        ext["Data"] = pd.to_datetime(ext["date"]).dt.strftime("%d/%m/%Y")
        ext["Valor (R$)"] = ext["amount"].apply(fmt_money)
        st.dataframe(
            ext[["Data","month_ref","description","Valor (R$)","is_recurring","aluno","professor_name"]] \
                .rename(columns={"month_ref":"Ref.","description":"Descrição","is_recurring":"Recorrente?","aluno":"Aluno","professor_name":"Professor"}),
            use_container_width=True, hide_index=True
        )
        total_ext = float(ext["amount"].astype(float).sum())
        st.metric("Total (extras)", fmt_money(total_ext))

    # Total geral
    st.subheader("Total geral (repasse + extras)")
    total_geral = (locals().get("total_rep",0.0) or 0.0) + (locals().get("total_ext",0.0) or 0.0)
    st.metric("Total geral (a repassar)", fmt_money(total_geral))

def page_config(role: str):
    st.header("Configurações")
    if role != "admin":
        st.warning("Apenas administradores podem acessar."); return

    cfg = fetch_settings()
    with st.form("form_cfg"):
        pct_ui = st.number_input("Percentual padrão de repasse (%)", min_value=0, max_value=100,
                                 value=fmt_pct_decimal_to_ui(cfg.get("master_percent")), step=5)
        ok = st.form_submit_button("💾 Salvar configuração")
    if ok:
        try:
            with engine.begin() as conn:
                conn.execute(update(settings).where(settings.c.id==1).values(master_percent=pct_ui_to_decimal(pct_ui)))
            invalidate_all_cache(); st.success("✅ Configuração atualizada.")
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
            invalidate_all_cache(); st.success("✅ Professor adicionado."); st.rerun()
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
            invalidate_all_cache(); st.success("✅ Horário adicionado."); st.rerun()
        except SQLAlchemyError as e:
            st.error(f"Erro: {e}")

    dfs = pd.DataFrame(fetch_slots())
    if not dfs.empty:
        st.dataframe(dfs[["name"]].rename(columns={"name":"Horário"}),
                     use_container_width=True, hide_index=True)

# -------------------- Main -------------------------
def main():
    role = require_login()
    with st.sidebar:
        st.markdown("### Navegação")
        page = st.radio("Ir para:",
                        ["Alunos","Graduações","Receber Pagamento","Extras (Repasse)","Relatórios","Configurações"],
                        index=0, label_visibility="collapsed")
        if st.button("🚪 Sair"):
            st.session_state.clear(); st.rerun()

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
