# streamlit_app.py
# JAT - Gestão de alunos (Muay Thai)
# Tecnologias: Streamlit + SQLAlchemy Core + Postgres (Neon)
# Esquema esperado no banco:
#   settings(id, master_percent)
#   coach(id, name, full_pass)
#   train_slot(id, name)
#   student(id, name, birth_date, start_date, monthly_fee, active, master_percent_override, coach_id, train_slot_id)
#   graduation_history(id, student_id, grade, grade_date, notes)
#   payment(id, student_id, paid_date, month_ref, amount, method, notes, master_percent_used, master_adjustment, master_amount)
#   extra_repasse(id, student_id, description, amount, month_ref, is_recurring, created_at)

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Date, DateTime, Boolean,
    Numeric, select, insert, update, delete, and_, or_, func, text
)
from sqlalchemy.engine import Engine

# -----------------------------------------------------------------------------
# Configuração básica
# -----------------------------------------------------------------------------

st.set_page_config(page_title="JAT - Gestão de alunos", page_icon="logo.png", layout="wide")

# Tenta pegar URL do banco
DB_URL = st.secrets.get("DATABASE_URL") or os.getenv("DATABASE_URL")

# Caso não encontre, cai para SQLite local (apenas para desenvolvimento)
if not DB_URL:
    DB_URL = "sqlite:///muaythai.db"

# -----------------------------------------------------------------------------
# Helpers de formatação/calculo
# -----------------------------------------------------------------------------

GRADES = [
    "Branca",
    "Amarelo", "Amarelo e Branca",
    "Verde", "Verde e Branca",
    "Azul", "Azul e Branca",
    "Marrom", "Marrom e Branca",
    "Vermelha", "Vermelha e Branca",
    "Preta",
]

def fmt_currency(x: Optional[float]) -> str:
    try:
        return "R$ {:,.2f}".format(float(x)).replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0,00"

def fmt_date(d: Optional[date]) -> str:
    if not d:
        return "—"
    return d.strftime("%d/%m/%Y")

def parse_date(obj) -> Optional[date]:
    if obj is None or obj == "":
        return None
    if isinstance(obj, date):
        return obj
    if isinstance(obj, datetime):
        return obj.date()
    try:
        return datetime.strptime(str(obj), "%Y-%m-%d").date()
    except Exception:
        try:
            return datetime.strptime(str(obj), "%d/%m/%Y").date()
        except Exception:
            return None

def month_of(d: date) -> str:
    return d.strftime("%Y-%m")

def idade_anos(dt_nasc: Optional[date]) -> str:
    if not dt_nasc:
        return "—"
    today = date.today()
    years = today.year - dt_nasc.year - ((today.month, today.day) < (dt_nasc.month, dt_nasc.day))
    return f"{years} anos"

def tempo_treino(dt_ini: Optional[date]) -> str:
    if not dt_ini:
        return "—"
    today = date.today()
    months = (today.year - dt_ini.year) * 12 + (today.month - dt_ini.month)
    if today.day < dt_ini.day:
        months -= 1
    if months < 12:
        return f"{months} meses"
    anos = months // 12
    rest = months % 12
    return f"{anos} anos e {rest} meses" if rest else f"{anos} anos"

# -----------------------------------------------------------------------------
# Conexão e tabelas (fixas, alinhadas ao schema)
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    # psycopg3 precisa de ?sslmode=require no Neon
    engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
    return engine

engine = get_engine()
metadata = MetaData()

T_SETTINGS  = "settings"
T_COACH     = "coach"
T_SLOT      = "train_slot"
T_STUDENT   = "student"
T_GH        = "graduation_history"
T_PAY       = "payment"
T_EXTRA     = "extra_repasse"

settings = Table(
    T_SETTINGS, metadata,
    Column("id", Integer, primary_key=True),
    Column("master_percent", Numeric(7,4), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now())
)

coach = Table(
    T_COACH, metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("full_pass", Boolean, nullable=False, server_default=text("false")),
    Column("created_at", DateTime(timezone=True), server_default=func.now())
)

slot = Table(
    T_SLOT, metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now())
)

student = Table(
    T_STUDENT, metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("birth_date", Date),
    Column("start_date", Date),
    Column("monthly_fee", Numeric(12,2), nullable=False, server_default="0"),
    Column("active", Boolean, nullable=False, server_default=text("true")),
    Column("master_percent_override", Numeric(7,4)),
    Column("coach_id", Integer),
    Column("train_slot_id", Integer),
    Column("created_at", DateTime(timezone=True), server_default=func.now())
)

gh = Table(
    T_GH, metadata,
    Column("id", Integer, primary_key=True),
    Column("student_id", Integer, nullable=False),
    Column("grade", String, nullable=False),
    Column("grade_date", Date, nullable=False),
    Column("notes", String),
    Column("created_at", DateTime(timezone=True), server_default=func.now())
)

pay = Table(
    T_PAY, metadata,
    Column("id", Integer, primary_key=True),
    Column("student_id", Integer, nullable=False),
    Column("paid_date", Date, nullable=False),
    Column("month_ref", String(7), nullable=False),
    Column("amount", Numeric(12,2), nullable=False),
    Column("method", String(30), nullable=False),
    Column("notes", String),
    Column("master_percent_used", Numeric(7,4), nullable=False, server_default="0"),
    Column("master_adjustment", Numeric(12,2), nullable=False, server_default="0"),
    Column("master_amount", Numeric(12,2), nullable=False, server_default="0"),
    Column("created_at", DateTime(timezone=True), server_default=func.now())
)

extra = Table(
    T_EXTRA, metadata,
    Column("id", Integer, primary_key=True),
    Column("student_id", Integer),
    Column("description", String, nullable=False),
    Column("amount", Numeric(12,2), nullable=False),
    Column("month_ref", String(7), nullable=False),
    Column("is_recurring", Boolean, nullable=False, server_default=text("false")),
    Column("created_at", Date, server_default=func.current_date())
)

# -----------------------------------------------------------------------------
# CRUD helpers
# -----------------------------------------------------------------------------

def ensure_settings() -> Dict[str, Any]:
    with engine.begin() as conn:
        row = conn.execute(select(settings)).mappings().first()
        if not row:
            conn.execute(insert(settings).values(master_percent=0.60))
            row = conn.execute(select(settings)).mappings().first()
    return dict(row)

def default_percent_for_student(stu: Dict[str, Any], coach_row: Optional[Dict[str, Any]], cfg: Dict[str, Any]) -> float:
    """Regra: se coach.full_pass True -> 1.0; senão aluno.override se houver; senão settings.master_percent."""
    if coach_row and coach_row.get("full_pass"):
        return 1.0
    if stu.get("master_percent_override") is not None:
        return float(stu["master_percent_override"])
    return float(cfg.get("master_percent", 0.60))

def add_default_white_grade(student_id: int, start_dt: Optional[date]):
    with engine.begin() as conn:
        # só insere se não existir qualquer graduação ainda
        exists = conn.execute(
            select(func.count()).select_from(gh).where(gh.c.student_id==student_id)
        ).scalar_one()
        if exists == 0:
            conn.execute(insert(gh).values(
                student_id=student_id,
                grade="Branca",
                grade_date=start_dt or date.today(),
                notes=None
            ))

def list_coaches() -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(select(coach).order_by(coach.c.name)).mappings().all()
    return [dict(r) for r in rows]

def list_slots() -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        rows = conn.execute(select(slot).order_by(slot.c.name)).mappings().all()
    return [dict(r) for r in rows]

@st.cache_data(ttl=20)
def fetch_students_df() -> pd.DataFrame:
    with engine.begin() as conn:
        stu_rows = conn.execute(select(student)).mappings().all()
        c_map = {r["id"]: dict(r) for r in conn.execute(select(coach)).mappings().all()}
        s_map = {r["id"]: dict(r) for r in conn.execute(select(slot)).mappings().all()}
        # graduacao mais recente por aluno
        gh_rows = conn.execute(select(gh).order_by(gh.c.student_id, gh.c.grade_date.desc())).mappings().all()

    df = pd.DataFrame([dict(r) for r in stu_rows])
    if df.empty:
        return pd.DataFrame(columns=[
            "id","name","birth_date","start_date","monthly_fee","active","coach","train_slot",
            "grade","grade_date","idade","tempo"
        ])

    # map coach/slot
    df["coach"] = df["coach_id"].map(lambda i: c_map.get(i, {}).get("name") if i else None)
    df["train_slot"] = df["train_slot_id"].map(lambda i: s_map.get(i, {}).get("name") if i else None)

    # última graduação
    latest_grade: Dict[int, Tuple[str, date]] = {}
    for r in gh_rows:
        sid = int(r["student_id"])
        if sid not in latest_grade:
            latest_grade[sid] = (r["grade"], parse_date(r["grade_date"]))
    df["grade"] = df["id"].map(lambda i: latest_grade.get(int(i), ("Branca", None))[0])
    df["grade_date"] = df["id"].map(lambda i: latest_grade.get(int(i), ("Branca", None))[1])

    # idade/tempo
    df["idade"] = df["birth_date"].map(lambda d: idade_anos(parse_date(d)))
    df["tempo"] = df["start_date"].map(lambda d: tempo_treino(parse_date(d)))

    return df

def add_student_row(values: Dict[str, Any]) -> int:
    with engine.begin() as conn:
        rid = conn.execute(insert(student).values(**values).returning(student.c.id)).scalar_one()
    return int(rid)

def update_student_row(student_id: int, values: Dict[str, Any]) -> None:
    with engine.begin() as conn:
        conn.execute(update(student).where(student.c.id==student_id).values(**values))

def record_payment(sid: int, paid_dt: date, amount_value: float, method_value: str, notes_value: Optional[str]) -> int:
    cfg = ensure_settings()
    # carrega aluno e coach
    with engine.begin() as conn:
        stu = conn.execute(select(student).where(student.c.id==sid)).mappings().first()
        if not stu:
            raise RuntimeError("Aluno não encontrado")
        c_row = None
        if stu.get("coach_id"):
            c_row = conn.execute(select(coach).where(coach.c.id==stu["coach_id"])).mappings().first()
    percent = default_percent_for_student(dict(stu), dict(c_row) if c_row else None, cfg)
    master_amount = float(amount_value) * float(percent)
    month_ref = month_of(paid_dt)

    with engine.begin() as conn:
        rid = conn.execute(
            insert(pay).values(
                student_id=sid,
                paid_date=paid_dt,
                month_ref=month_ref,
                amount=float(amount_value),
                method=method_value,
                notes=notes_value,
                master_percent_used=percent,
                master_adjustment=0.0,
                master_amount=master_amount
            ).returning(pay.c.id)
        ).scalar_one()
    return int(rid)

def delete_payments(ids: List[int]) -> int:
    if not ids:
        return 0
    with engine.begin() as conn:
        res = conn.execute(delete(pay).where(pay.c.id.in_(ids)))
        return res.rowcount or 0

# -----------------------------------------------------------------------------
# Login simples (opcional via Secrets)
# -----------------------------------------------------------------------------

def check_login() -> str:
    """Retorna 'admin' ou 'operador'. Se não houver secrets, assume admin."""
    admin_pwd = st.secrets.get("admin") or os.getenv("admin")
    op_pwd    = st.secrets.get("operador") or os.getenv("operador")

    if not admin_pwd and not op_pwd:
        return "admin"

    if "role" in st.session_state:
        return st.session_state["role"]

    st.title("🔐 JAT - Gestão de alunos")
    st.caption("Faça login para continuar")
    with st.form("login"):
        role = st.selectbox("Perfil", ["admin", "operador"])
        pwd  = st.text_input("Senha", type="password")
        ok   = st.form_submit_button("Entrar", type="primary")
        if ok:
            if (role == "admin" and pwd == admin_pwd) or (role == "operador" and pwd == op_pwd):
                st.session_state["role"] = role
                st.success("Login efetuado!")
                st.rerun()
            else:
                st.error("Credenciais inválidas.")
    st.stop()

# -----------------------------------------------------------------------------
# UI — Cabeçalho
# -----------------------------------------------------------------------------

def header():
    c1, c2 = st.columns([1,6])
    with c1:
        try:
            st.image("logo.png", width=80)
        except Exception:
            st.write("🥊")
    with c2:
        st.markdown("## JAT - Gestão de alunos")

# -----------------------------------------------------------------------------
# Páginas
# -----------------------------------------------------------------------------

def page_alunos(role: str):
    st.markdown("### 👥 Alunos")
    df = fetch_students_df()

    # Tabela principal
    st.dataframe(
        df[["id","name","active","monthly_fee","coach","train_slot","grade","idade","tempo"]]
          .rename(columns={
              "id":"ID","name":"Aluno","active":"Ativo?","monthly_fee":"Mensalidade (R$)",
              "coach":"Professor","train_slot":"Horário","grade":"Graduação","idade":"Idade","tempo":"Tempo de treino"
          }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.markdown("#### ➕ Cadastrar novo aluno")
    coaches = list_coaches()
    slots   = list_slots()
    coach_opts = ["(sem professor)"] + [c["name"] for c in coaches]
    slot_opts  = ["(sem horário)"]   + [s["name"] for s in slots]

    with st.form("form_add_student"):
        col1, col2, col3 = st.columns([3,2,2])
        with col1:
            n_name = st.text_input("Nome *")
        with col2:
            n_birth = st.date_input("Data de nascimento", value=None)
        with col3:
            n_start = st.date_input("Início no treino", value=None)

        col4, col5, col6 = st.columns([2,2,2])
        with col4:
            n_fee = st.number_input("Mensalidade (R$)", min_value=0.0, value=0.0, step=10.0, format="%.2f")
        with col5:
            c_sel = st.selectbox("Professor responsável", options=coach_opts)
        with col6:
            s_sel = st.selectbox("Horário do treino", options=slot_opts)

        col7, col8 = st.columns([2,2])
        with col7:
            override_pct = st.number_input("Repasse do aluno (%) (0 = usar padrão)", min_value=0, max_value=100, step=5, value=0)
        with col8:
            ativo = st.checkbox("Ativo?", value=True)

        submit = st.form_submit_button("Cadastrar", type="primary")

    if submit:
        if not n_name:
            st.error("Informe o nome do aluno.")
        else:
            coach_id = None
            if c_sel != coach_opts[0]:
                coach_id = next((c["id"] for c in coaches if c["name"]==c_sel), None)
            slot_id = None
            if s_sel != slot_opts[0]:
                slot_id = next((s["id"] for s in slots if s["name"]==s_sel), None)

            values = {
                "name": n_name.strip(),
                "birth_date": n_birth,
                "start_date": n_start,
                "monthly_fee": float(n_fee or 0.0),
                "active": bool(ativo),
                "coach_id": coach_id,
                "train_slot_id": slot_id,
                "master_percent_override": (float(override_pct)/100.0) if override_pct>0 else None
            }
            rid = add_student_row(values)
            add_default_white_grade(rid, n_start)
            st.success(f"Aluno cadastrado (ID {rid}).")
            st.cache_data.clear()
            st.rerun()

    st.divider()
    st.markdown("#### ✏️ Editar aluno")
    if df.empty:
        st.info("Nenhum aluno cadastrado.")
        return

    sid = st.selectbox(
        "Selecione o aluno (por ID)",
        options=df["id"].tolist(),
        format_func=lambda i: f"ID {i} — {df.loc[df['id']==i,'name'].values[0]}"
    )
    row = df[df["id"]==sid].iloc[0].to_dict()

    with st.form("form_edit_student"):
        col1, col2, col3 = st.columns([3,2,2])
        with col1:
            e_name = st.text_input("Nome *", value=row.get("name",""))
        with col2:
            e_birth = st.date_input("Data de nascimento", value=parse_date(row.get("birth_date")))
        with col3:
            e_start = st.date_input("Início no treino", value=parse_date(row.get("start_date")))

        col4, col5, col6 = st.columns([2,2,2])
        with col4:
            e_fee = st.number_input("Mensalidade (R$)", min_value=0.0, value=float(row.get("monthly_fee") or 0.0), step=10.0, format="%.2f")
        with col5:
            e_coach = st.selectbox("Professor responsável", options=coach_opts,
                                   index=0 if not row.get("coach_id") else (1 + [c["id"] for c in coaches].index(int(row["coach_id"])) ))
        with col6:
            e_slot = st.selectbox("Horário do treino", options=slot_opts,
                                  index=0 if not row.get("train_slot_id") else (1 + [s["id"] for s in slots].index(int(row["train_slot_id"])) ))

        col7, col8 = st.columns([2,2])
        with col7:
            cur_override = row.get("master_percent_override")
            e_override = st.number_input(
                "Repasse do aluno (%) (0 = usar padrão)",
                min_value=0, max_value=100,
                value=int(round(float(cur_override)*100)) if cur_override not in (None,"") else 0,
                step=5
            )
        with col8:
            e_active = st.checkbox("Ativo?", value=bool(row.get("active", True)))

        save = st.form_submit_button("Salvar alterações", type="primary")

    if save:
        vals = {
            "name": e_name.strip(),
            "birth_date": e_birth,
            "start_date": e_start,
            "monthly_fee": float(e_fee or 0.0),
            "active": bool(e_active),
            "master_percent_override": (float(e_override)/100.0) if e_override>0 else None
        }
        if e_coach != coach_opts[0]:
            vals["coach_id"] = next((c["id"] for c in coaches if c["name"]==e_coach), None)
        else:
            vals["coach_id"] = None
        if e_slot != slot_opts[0]:
            vals["train_slot_id"] = next((s["id"] for s in slots if s["name"]==e_slot), None)
        else:
            vals["train_slot_id"] = None

        update_student_row(int(sid), vals)
        st.success("Aluno atualizado.")
        st.cache_data.clear()
        st.rerun()

def page_graduacoes(role: str):
    st.markdown("### 🎖️ Graduações")
    df = fetch_students_df()
    if df.empty:
        st.info("Cadastre alunos primeiro.")
        return

    sid = st.selectbox(
        "Aluno", options=df["id"].tolist(),
        format_func=lambda i: f"ID {i} — {df.loc[df['id']==i,'name'].values[0]}"
    )

    # Histórico
    with engine.begin() as conn:
        rows = conn.execute(
            select(gh).where(gh.c.student_id==sid).order_by(gh.c.grade_date.desc())
        ).mappings().all()
    hist = pd.DataFrame([dict(r) for r in rows])
    st.markdown("#### Histórico")
    if hist.empty:
        st.info("Sem graduações registradas.")
    else:
        st.dataframe(
            hist[["id","grade","grade_date","notes"]].rename(columns={
                "id":"ID","grade":"Graduação","grade_date":"Data","notes":"Observações"
            }).assign(Data=lambda d: d["Data"].map(fmt_date)).drop(columns=["grade_date"]),
            use_container_width=True, hide_index=True
        )

    st.markdown("#### Adicionar graduação")
    with st.form("form_add_grad"):
        col1, col2 = st.columns([2,2])
        with col1:
            gg = st.selectbox("Graduação", options=GRADES, index=0)
        with col2:
            gd = st.date_input("Data da graduação", value=date.today())
        notes = st.text_input("Observações (opcional)")
        ok = st.form_submit_button("Salvar", type="primary")

    if ok:
        with engine.begin() as conn:
            conn.execute(insert(gh).values(student_id=int(sid), grade=gg, grade_date=gd, notes=notes or None))
        st.success("Graduação registrada.")
        st.cache_data.clear()
        st.rerun()

def page_receber(role: str):
    st.markdown("### 💵 Receber Pagamento")
    df = fetch_students_df()
    if df.empty:
        st.info("Cadastre alunos primeiro.")
        return

    # Filtro por professor
    profs = ["(todos)"] + sorted([p for p in df["coach"].dropna().unique()])
    coach_sel = st.selectbox("Filtrar por professor", profs)

    df_active = df[df["active"]==True].copy()
    if coach_sel != "(todos)":
        df_active = df_active[df_active["coach"]==coach_sel]

    st.markdown("#### Confirmar pagamento de vários alunos")
    with st.form("form_receive_many"):
        c1, c2, c3 = st.columns([3,2,2])
        with c1:
            ids = st.multiselect("Selecione os alunos", options=df_active["id"].tolist(),
                                 format_func=lambda i: f"ID {i} — {df_active.loc[df_active['id']==i,'name'].values[0]}")
        with c2:
            paid_dt = st.date_input("Data do pagamento", value=date.today())
        with c3:
            method = st.selectbox("Forma", ["Dinheiro","PIX","Cartão","Transferência"])
        notes = st.text_input("Observações (opcional)")
        ok = st.form_submit_button("Registrar", type="primary")

    if ok:
        if not ids:
            st.warning("Selecione ao menos um aluno.")
        else:
            cnt = 0
            for sid in ids:
                fee = float(df_active.loc[df_active["id"]==sid,"monthly_fee"].values[0] or 0.0)
                record_payment(int(sid), paid_dt, fee, method, notes or None)
                cnt += 1
            st.success(f"Pagamentos registrados para {cnt} aluno(s).")
            st.cache_data.clear()
            st.rerun()

    st.divider()
    st.markdown("#### Pagamentos do mês e exclusão")
    month = st.text_input("Mês (AAAA-MM)", value=month_of(date.today()))
    # Mostrar tabela de pagos no mês
    with engine.begin() as conn:
        rows = conn.execute(select(pay).where(pay.c.month_ref==month).order_by(pay.c.paid_date.desc())).mappings().all()
    dfp = pd.DataFrame([dict(r) for r in rows])
    if dfp.empty:
        st.info("Sem pagamentos nesse mês.")
        return

    # Juntar com nomes
    df_name = fetch_students_df()[["id","name"]].rename(columns={"id":"student_id"})
    dft = dfp.merge(df_name, on="student_id", how="left")
    dft["paid_date_fmt"] = dft["paid_date"].map(fmt_date)
    dft["amount_fmt"]    = dft["amount"].map(fmt_currency)
    dft["repasse_fmt"]   = dft["master_amount"].map(fmt_currency)
    dft["percent_fmt"]   = (dft["master_percent_used"].astype(float)*100.0).round(0).astype(int).astype(str) + "%"

    st.dataframe(
        dft[["id","name","paid_date_fmt","amount_fmt","percent_fmt","repasse_fmt","method","notes"]]
         .rename(columns={"id":"ID","name":"Aluno","paid_date_fmt":"Data","amount_fmt":"Valor","percent_fmt":"% repasse","repasse_fmt":"Repasse (R$)","method":"Forma","notes":"Obs"}),
        use_container_width=True, hide_index=True
    )

    with st.form("form_delete_pay"):
        ids_to_delete = st.multiselect("Selecionar pagamentos para excluir",
                                       options=dft["id"].tolist(),
                                       format_func=lambda i: f"ID {i} — {dft.loc[dft['id']==i,'name'].values[0]} ({dft.loc[dft['id']==i,'paid_date_fmt'].values[0]})")
        c1, c2 = st.columns([1,1])
        with c1:
            btn = st.form_submit_button("Excluir selecionados", type="secondary")
        with c2:
            allm = st.form_submit_button(f"Excluir todos de {month}", type="secondary")

    if btn and ids_to_delete:
        n = delete_payments([int(i) for i in ids_to_delete])
        st.success(f"{n} pagamento(s) excluído(s).")
        st.rerun()

    if allm:
        n = delete_payments(dft["id"].tolist())
        st.success(f"Todos os pagamentos de {month} foram excluídos ({n} registro(s)).")
        st.rerun()

def page_extras(role: str):
    st.markdown("### ➕ Extras (Repasse)")
    df = fetch_students_df()
    students_opts = ["(Sem aluno vinculado)"] + [f"ID {i} — {n}" for i, n in df[["id","name"]].values.tolist()]

    with st.form("form_extra"):
        col1, col2, col3 = st.columns([2,2,2])
        with col1:
            dt = st.date_input("Data do lançamento", value=date.today())
        with col2:
            month = st.text_input("Mês de referência (AAAA-MM)", value=month_of(date.today()))
        with col3:
            val = st.number_input("Valor do extra (R$) (pode ser negativo)", value=0.0, step=10.0, format="%.2f")
        desc = st.text_input("Descrição")
        rec  = st.radio("Recorrente?", ["Não","Sim"], horizontal=True)
        vinc = st.selectbox("Vincular a um aluno (opcional)", options=students_opts)
        ok   = st.form_submit_button("Salvar", type="primary")

    if ok:
        sid = None
        if vinc != students_opts[0]:
            sid = int(vinc.split(" ")[1])
        with engine.begin() as conn:
            conn.execute(insert(extra).values(
                student_id=sid,
                description=desc.strip() if desc else "Outros",
                amount=float(val),
                month_ref=month,
                is_recurring=(rec=="Sim"),
                created_at=dt
            ))
        st.success("Extra registrado.")
        st.rerun()

    st.divider()
    st.markdown("#### Lista de extras por mês")
    filtro = st.text_input("Mês (AAAA-MM)", value=month_of(date.today()))
    with engine.begin() as conn:
        rows = conn.execute(select(extra).where(
            or_(
                extra.c.month_ref==filtro,
                and_(extra.c.is_recurring==True, extra.c.created_at <= text(f"DATE '{filtro}-01'") + text(" + INTERVAL '1 month' - INTERVAL '1 day'"))
            )
        ).order_by(extra.c.id)).mappings().all()
    dfe = pd.DataFrame([dict(r) for r in rows])
    if dfe.empty:
        st.info("Sem extras para o mês.")
    else:
        name_map = dict(df[["id","name"]].values.tolist())
        dfe["Aluno"] = dfe["student_id"].map(lambda x: name_map.get(int(x)) if pd.notna(x) else "Outros")
        dfe["Data"]  = dfe["created_at"].map(fmt_date)
        dfe["Valor (R$)"] = dfe["amount"].map(fmt_currency)
        dfe["Recorrente?"] = dfe["is_recurring"].map(lambda v: "Sim" if v else "Não")
        st.dataframe(dfe[["id","Data","Aluno","description","Valor (R$)","Recorrente?"]]
                     .rename(columns={"id":"ID","description":"Descrição"}),
                     use_container_width=True, hide_index=True)

def page_relatorios(role: str):
    st.markdown("### 📊 Relatórios")
    df = fetch_students_df()
    if df.empty:
        st.info("Cadastre alunos primeiro.")
        return

    c1, c2 = st.columns([2,2])
    with c1:
        month = st.text_input("Mês (AAAA-MM)", value=month_of(date.today()))
    with c2:
        profs = ["(todos)"] + sorted([p for p in df["coach"].dropna().unique()])
        coach_sel = st.selectbox("Filtrar por professor", profs)

    # Pagamentos
    with engine.begin() as conn:
        pagos = conn.execute(select(pay).where(pay.c.month_ref==month)).mappings().all()
        rows_e = conn.execute(select(extra)).mappings().all()

    dpp = pd.DataFrame([dict(r) for r in pagos])
    dpp = dpp.merge(df[["id","name","birth_date","start_date","grade","coach"]].rename(columns={"id":"student_id"}),
                    on="student_id", how="left")

    if coach_sel != "(todos)":
        dpp = dpp[dpp["coach"]==coach_sel]

    if not dpp.empty:
        dpp["Idade"] = dpp["birth_date"].map(lambda d: idade_anos(parse_date(d)))
        dpp["Tempo de treino"] = dpp["start_date"].map(lambda d: tempo_treino(parse_date(d)))
        dpp["Graduação"] = dpp["grade"]
        dpp["Data"] = dpp["paid_date"].map(fmt_date)
        dpp["Valor (R$)"] = dpp["amount"].map(fmt_currency)
        dpp["Repasse (%)"] = (dpp["master_percent_used"].astype(float)*100.0).round(0).astype(int).astype(str) + "%"
        dpp["Repasse (R$)"] = dpp["master_amount"].map(fmt_currency)

        st.markdown("#### Relatório de repasse de mensalidades")
        st.dataframe(dpp[["student_id","name","Idade","Tempo de treino","Graduação","Data","method","Valor (R$)","Repasse (%)","Repasse (R$)"]]
                     .rename(columns={"student_id":"ID","name":"Aluno","method":"Forma"}),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Sem pagamentos nesse mês com o filtro selecionado.")

    total_repasse_pag = float(dpp["master_amount"].astype(float).sum()) if not dpp.empty else 0.0

    # Extras (detalhado)
    dfe = pd.DataFrame([dict(r) for r in rows_e])
    if not dfe.empty:
        df_names = df[["id","name","coach"]].rename(columns={"id":"student_id"})
        dfe = dfe.merge(df_names, on="student_id", how="left")
        if coach_sel != "(todos)":
            # incluir extras sem aluno (Outros) + extras de alunos do professor
            dfe = dfe[(dfe["name"].isna()) | (dfe["coach"]==coach_sel)]

        # Recorrentes: entram em qualquer mês a partir da created_at; pontuais: exatamente no month_ref
        month_first = datetime.strptime(month+"-01", "%Y-%m-%d").date()
        dfe = dfe[( (dfe["is_recurring"]==True) & (dfe["created_at"]<=month_first) ) | (dfe["month_ref"]==month)]
        dfe["Aluno"] = dfe.apply(lambda r: ("Outros" if pd.isna(r.get("name")) else r.get("name")), axis=1)
        dfe["Data"]  = dfe["created_at"].map(fmt_date)
        dfe["Valor (R$)"] = dfe["amount"].map(fmt_currency)
        dfe["Recorrente?"] = dfe["is_recurring"].map(lambda v: "Sim" if v else "Não")

        st.markdown("#### Relatório de extras (detalhado)")
        st.dataframe(dfe[["id","Data","Aluno","description","Valor (R$)","Recorrente?"]]
                     .rename(columns={"id":"ID","description":"Descrição"}),
                     use_container_width=True, hide_index=True)
        total_extras = float(dfe["amount"].astype(float).sum())
    else:
        st.info("Sem extras cadastrados.")
        total_extras = 0.0

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Repasse sobre mensalidades", fmt_currency(total_repasse_pag))
    with c2:
        st.metric("Extras (repasse)", fmt_currency(total_extras))
    with c3:
        st.metric("Total geral", fmt_currency(total_repasse_pag + total_extras))

def page_config(role: str):
    if role != "admin":
        st.info("Acesso restrito ao administrador.")
        return

    st.markdown("### ⚙️ Configurações")
    cfg = ensure_settings()
    curr_pct = int(round(float(cfg.get("master_percent", 0.60))*100))

    with st.form("form_cfg"):
        pct = st.number_input("Percentual padrão de repasse (%)", min_value=0, max_value=100, step=5, value=curr_pct)
        ok  = st.form_submit_button("Salvar", type="primary")

    if ok:
        with engine.begin() as conn:
            conn.execute(update(settings).where(settings.c.id==cfg["id"]).values(master_percent=float(pct)/100.0))
        st.success("Configurações salvas.")
        st.rerun()

    st.divider()
    st.markdown("#### Cadastros auxiliares")
    st.caption("Professores e horários utilizados no cadastro de alunos")
    tab1, tab2 = st.tabs(["👨‍🏫 Professores", "🕒 Horários"])

    with tab1:
        with st.form("form_coach"):
            name = st.text_input("Nome do professor")
            full = st.checkbox("Repasse completo (100%)", value=False)
            ok = st.form_submit_button("Adicionar", type="primary")
        if ok:
            if not name.strip():
                st.error("Informe o nome.")
            else:
                with engine.begin() as conn:
                    conn.execute(insert(coach).values(name=name.strip(), full_pass=bool(full)))
                st.success("Professor cadastrado.")
                st.rerun()

        lst = list_coaches()
        if lst:
            dfc = pd.DataFrame(lst)
            dfc["full_pass"] = dfc["full_pass"].map(lambda v: "Sim" if v else "Não")
            st.dataframe(dfc.rename(columns={"id":"ID","name":"Nome","full_pass":"Repasse completo?"}),
                         use_container_width=True, hide_index=True)

    with tab2:
        with st.form("form_slot"):
            sname = st.text_input("Nome/descrição do horário (ex.: Seg/Qua 19h)")
            ok2 = st.form_submit_button("Adicionar", type="primary")
        if ok2:
            if not sname.strip():
                st.error("Informe o nome/descrição.")
            else:
                with engine.begin() as conn:
                    conn.execute(insert(slot).values(name=sname.strip()))
                st.success("Horário cadastrado.")
                st.rerun()

        lsts = list_slots()
        if lsts:
            dfs = pd.DataFrame(lsts)
            st.dataframe(dfs.rename(columns={"id":"ID","name":"Horário"}), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

def main():
    role = check_login()
    header()

    pages_admin = {
        "Alunos": page_alunos,
        "Graduações": page_graduacoes,
        "Receber Pagamento": page_receber,
        "Extras (Repasse)": page_extras,
        "Relatórios": page_relatorios,
        "Configurações": page_config,
    }
    pages_oper = {
        "Alunos": page_alunos,
        "Receber Pagamento": page_receber,
        "Relatórios": page_relatorios,
    }

    menu = pages_admin if role=="admin" else pages_oper

    with st.sidebar:
        st.markdown("### Navegação")
        page = st.radio("Ir para:", list(menu.keys()))
        st.divider()
        if st.button("Sair"):
            st.session_state.pop("role", None)
            st.rerun()

    # Render
    menu[page](role)

if __name__ == "__main__":
    main()
