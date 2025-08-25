# streamlit_app.py — v2.6.3
from __future__ import annotations

import os
from datetime import date, datetime
from typing import Optional, List, Dict

import pandas as pd
import streamlit as st
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy.exc import OperationalError

# -----------------------------------------------
# Página / Branding
# -----------------------------------------------
st.set_page_config(
    page_title="Muay Thai — Gestão de Alunos",
    layout="wide",
    page_icon="logo.png",
)

# Cabeçalho com logo
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=80)
with col_title:
    st.title("Gestão da Turma de Muay Thai")

# -----------------------------------------------
# Constantes & Conexão
# -----------------------------------------------
ALLOWED_GRADES = [
    "Branca", "Amarelo", "Amarelo e Branca", "Verde", "Verde e Branca",
    "Azul", "Azul e Branca", "Marrom", "Marrom e Branca",
    "Vermelha", "Vermelha e Branca", "Preta"
]

DB_PATH = "muaythai.db"
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SQLModel.metadata.clear()  # evita conflito de definição em reloads

# -----------------------------------------------
# Models
# (sem index=True nos Fields para não recriar índices antigos automaticamente)
# -----------------------------------------------
class Settings(SQLModel, table=True):
    __tablename__ = "settings"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    default_master_percent: float = Field(default=0.20)
    currency_symbol: str = Field(default="R$")

class Coach(SQLModel, table=True):
    __tablename__ = "coach"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    full_pass: bool = Field(default=False, description="Repasse completo (100%)")

class TrainSlot(SQLModel, table=True):
    __tablename__ = "train_slot"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    label: str

class Student(SQLModel, table=True):
    __tablename__ = "student"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    birth_date: Optional[date] = Field(default=None)
    start_date: Optional[date] = Field(default=None)
    grade: Optional[str] = Field(default=None)          # derivada do histórico
    grade_date: Optional[date] = Field(default=None)    # derivada do histórico
    active: bool = Field(default=True)
    monthly_fee: float = Field(default=0.0)
    master_percent_override: Optional[float] = Field(default=None)
    coach_id: Optional[int] = Field(default=None)
    train_slot_id: Optional[int] = Field(default=None)
    # legados
    coach: Optional[str] = Field(default=None)
    train_time: Optional[str] = Field(default=None)

class Graduation(SQLModel, table=True):
    __tablename__ = "graduation"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: int = Field()
    grade: str
    grade_date: date
    notes: Optional[str] = Field(default=None)

class ExtraRepasse(SQLModel, table=True):
    __tablename__ = "extra_repasse"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    extra_date: date = Field(default_factory=date.today)
    month_ref: str = Field()
    description: str
    amount: float                                   # pode ser negativo
    is_recurring: bool = Field(default=False)       # aplica todo mês a partir do mês do lançamento
    student_id: Optional[int] = Field(default=None)

class Payment(SQLModel, table=True):
    __tablename__ = "payment"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: int = Field()
    paid_date: date = Field(default_factory=date.today)
    month_ref: str = Field()
    amount: float
    method: Optional[str] = Field(default="PIX")
    notes: Optional[str] = Field(default=None)
    master_percent_used: float = Field(default=0.0)
    master_adjustment: float = Field(default=0.0)
    master_amount: float = Field(default=0.0)

# -----------------------------------------------
# DB Init & Migrações
# -----------------------------------------------
def init_db():
    # drop de índices antigos/duplicados antes de recriar o schema
    with engine.begin() as conn:
        try:
            rows = conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
            for (idx_name,) in rows:
                # remove todos ix_* e quaisquer índices antigos
                if idx_name.startswith("ix_") or idx_name.startswith("idx_") or idx_name.startswith("sqlite_autoindex"):
                    try:
                        conn.exec_driver_sql(f'DROP INDEX IF EXISTS "{idx_name}"')
                    except Exception:
                        pass
        except Exception:
            pass
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        if not session.exec(select(Settings)).first():
            session.add(Settings())
            session.commit()

def migrate_db():
    import sqlite3
    conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
    # student: colunas novas
    cur.execute("PRAGMA table_info(student);")
    cols = [r[1] for r in cur.fetchall()]
    if "master_percent_override" not in cols:
        cur.execute("ALTER TABLE student ADD COLUMN master_percent_override REAL;"); conn.commit()
    if "coach_id" not in cols:
        cur.execute("ALTER TABLE student ADD COLUMN coach_id INTEGER;"); conn.commit()
    if "train_slot_id" not in cols:
        cur.execute("ALTER TABLE student ADD COLUMN train_slot_id INTEGER;"); conn.commit()
    # payment: renomear date -> paid_date se existir
    cur.execute("PRAGMA table_info(payment);")
    pcols = [r[1] for r in cur.fetchall()]
    if "date" in pcols and "paid_date" not in pcols:
        try:
            cur.execute("ALTER TABLE payment RENAME COLUMN date TO paid_date;"); conn.commit()
        except Exception:
            pass
    # extra_repasse: is_recurring
    cur.execute("PRAGMA table_info(extra_repasse);")
    ecols = [r[1] for r in cur.fetchall()]
    if "is_recurring" not in ecols:
        try:
            cur.execute("ALTER TABLE extra_repasse ADD COLUMN is_recurring INTEGER DEFAULT 0;"); conn.commit()
        except Exception:
            pass
    conn.close()

def ensure_coach_full_pass_column():
    # cria a coluna full_pass em coach se não existir
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("ALTER TABLE coach ADD COLUMN full_pass BOOLEAN DEFAULT 0")
    except Exception:
        # já existe ou outra condição segura de ignorar
        pass


    # Migrar textos legados coach/train_time -> FKs
    with Session(engine) as session:
        students = list(session.exec(select(Student)))
        for s in students:
            if s.coach and not s.coach_id:
                nm = s.coach.strip()
                if nm:
                    obj = session.exec(select(Coach).where(Coach.name == nm)).first()
                    if not obj:
                        obj = Coach(name=nm); session.add(obj); session.commit(); session.refresh(obj)
                    s.coach_id = obj.id
            if s.train_time and not s.train_slot_id:
                lb = s.train_time.strip()
                if lb:
                    obj = session.exec(select(TrainSlot).where(TrainSlot.label == lb)).first()
                    if not obj:
                        obj = TrainSlot(label=lb); session.add(obj); session.commit(); session.refresh(obj)
                    s.train_slot_id = obj.id
            session.add(s)
        session.commit()

# -----------------------------------------------
# Helpers
# -----------------------------------------------
def get_settings() -> Settings:
    with Session(engine) as session:
        return session.exec(select(Settings)).first()

def save_settings(p: float, sym: str):
    with Session(engine) as session:
        cfg = session.exec(select(Settings)).first() or Settings()
        cfg.default_master_percent = p; cfg.currency_symbol = sym
        session.add(cfg); session.commit()

def coach_map() -> Dict[int, str]:
    with Session(engine) as session:
        return {c.id: c.name for c in session.exec(select(Coach).order_by(Coach.name))}

def slot_map() -> Dict[int, str]:
    with Session(engine) as session:
        return {t.id: t.label for t in session.exec(select(TrainSlot).order_by(TrainSlot.label))}

def list_coaches() -> List[Coach]:
    with Session(engine) as session:
        return list(session.exec(select(Coach).order_by(Coach.name)))

def list_train_slots() -> List[TrainSlot]:
    with Session(engine) as session:
        return list(session.exec(select(TrainSlot).order_by(TrainSlot.label)))

def add_coach(name: str) -> int:
    name = (name or "").strip()
    if not name: return 0
    with Session(engine) as session:
        ex = session.exec(select(Coach).where(Coach.name == name)).first()
        if ex: return ex.id
        c = Coach(name=name); session.add(c); session.commit(); session.refresh(c); return c.id

def delete_coach(cid: int):
    with Session(engine) as session:
        for s in session.exec(select(Student).where(Student.coach_id == cid)):
            s.coach_id = None; session.add(s)
        c = session.get(Coach, cid)
        if c: session.delete(c)
        session.commit()

def set_coach_full_pass(coach_id: int, full_pass: bool):
    with Session(engine) as session:
        c = session.get(Coach, coach_id)
        if c:
            c.full_pass = bool(full_pass)
            session.add(c); session.commit()


def add_train_slot(label: str) -> int:
    label = (label or "").strip()
    if not label: return 0
    with Session(engine) as session:
        ex = session.exec(select(TrainSlot).where(TrainSlot.label == label)).first()
        if ex: return ex.id
        t = TrainSlot(label=label); session.add(t); session.commit(); session.refresh(t); return t.id

def delete_train_slot(tid: int):
    with Session(engine) as session:
        for s in session.exec(select(Student).where(Student.train_slot_id == tid)):
            s.train_slot_id = None; session.add(s)
        t = session.get(TrainSlot, tid)
        if t: session.delete(t)
        session.commit()


def add_student(
    *,
    name: str,
    birth_date: date | None = None,
    start_date: date | None = None,
    monthly_fee: float = 0.0,
    active: bool = True,
    coach_id: int | None = None,
    train_slot_id: int | None = None,
    master_percent: float | None = None,
    master_percent_override: float | None = None,
) -> int:
    """Cria aluno e retorna o ID. Usa override de repasse se informado."""
    with Session(engine) as session:
        stu = Student(
            name=name,
            birth_date=birth_date,
            start_date=start_date,
            monthly_fee=monthly_fee,
            active=active,
            coach_id=coach_id,
            train_slot_id=train_slot_id,
            master_percent=(master_percent_override if master_percent_override is not None else master_percent),
        )
        session.add(stu)
        session.commit()
        session.refresh(stu)
        try:
            # Graduação inicial padrão (Branca) com data = início do treino
            if start_date is not None:
                add_graduation(int(stu.id), "Branca", start_date, "Graduação padrão no cadastro")
        except Exception:
            pass
        return int(stu.id)

def get_students_by_coach(coach_id: Optional[int], active_only=True) -> List[Student]:
    with Session(engine) as session:
        q = select(Student)
        if active_only: q = q.where(Student.active == True)
        if coach_id: q = q.where(Student.coach_id == coach_id)
        return list(session.exec(q))

def get_student_by_id(sid: int) -> Optional[Student]:
    with Session(engine) as session:
        return session.get(Student, sid)

# ---- Graduações
def add_graduation(student_id: int, grade: str, grade_date: date, notes: Optional[str] = None):
    with Session(engine) as session:
        g = Graduation(student_id=student_id, grade=grade, grade_date=grade_date, notes=notes)
        session.add(g); session.commit()
    refresh_student_grade(student_id)

def list_graduations(student_id: int) -> pd.DataFrame:
    with Session(engine) as session:
        rows = list(session.exec(select(Graduation).where(Graduation.student_id == student_id).order_by(Graduation.grade_date)))
    if not rows:
        return pd.DataFrame(columns=["id","student_id","grade","grade_date","notes"])
    return pd.DataFrame([r.model_dump() for r in rows])

def refresh_student_grade(student_id: int):
    gdf = list_graduations(student_id)
    if not gdf.empty:
        latest = gdf.sort_values("grade_date", ascending=False).iloc[0]
        update_student(student_id, grade=latest["grade"], grade_date=latest["grade_date"])

# ---- Pagamentos/Extras
def record_payment(student_id: int, paid_date_val: date, amount: float, method: str, notes: str,
                   month_ref: str, master_percent_used: float, master_adjustment: float):
    master_amount = round(amount * master_percent_used + master_adjustment, 2)
    with Session(engine) as session:
        p = Payment(student_id=student_id, paid_date=paid_date_val, amount=amount, method=method, notes=notes,
                    month_ref=month_ref, master_percent_used=master_percent_used, master_adjustment=master_adjustment,
                    master_amount=master_amount)
        session.add(p); session.commit()

def record_payment_batch(student_ids: list, paid_date_val: date, amount_common: Optional[float],
                         use_each_fee: bool, method: str, notes: str, month_ref: str,
                         base_master_percent: float, master_adjustment: float):
    n = 0
    for sid in student_ids:
        s = get_student_by_id(sid)
        if not s: continue
        amt = float(s.monthly_fee or 0.0) if use_each_fee else float(amount_common or 0.0)
        # se professor tiver repasse completo (100%), força 1.0
        coach_full = False
        try:
            if s.coach_id:
                with Session(engine) as session:
                    cobj = session.get(Coach, s.coach_id)
                    coach_full = bool(getattr(cobj, "full_pass", False))
        except Exception:
            coach_full = False
        pct = 1.0 if coach_full else (float(s.master_percent_override) if s.master_percent_override is not None else float(base_master_percent))
        record_payment(sid, paid_date_val, amt, method, notes, month_ref, pct, master_adjustment); n += 1
    return n

def get_payments(month_ref: Optional[str] = None) -> pd.DataFrame:
    with Session(engine) as session:
        q = select(Payment)
        if month_ref: q = q.where(Payment.month_ref == month_ref)
        rows = list(session.exec(q))
    if not rows:
        return pd.DataFrame(columns=["id","student_id","paid_date","month_ref","amount","method","notes","master_percent_used","master_adjustment","master_amount"])
    return pd.DataFrame([r.model_dump() for r in rows])

def paid_ids_for_month(month_ref: str) -> set[int]:
    df = get_payments(month_ref=month_ref)
    if df is None or df.empty:
        return set()
    try:
        return set(int(x) for x in df["student_id"].tolist())
    except Exception:
        return set()


def add_extra(month_ref: str, description: str, amount: float, date_val: date,
              student_id: Optional[int] = None, is_recurring: bool = False):
    with Session(engine) as session:
        e = ExtraRepasse(month_ref=month_ref, description=description, amount=amount,
                         extra_date=date_val, student_id=student_id, is_recurring=is_recurring)
        session.add(e); session.commit()

def get_extras(month_ref: Optional[str] = None) -> pd.DataFrame:
    with Session(engine) as session:
        rows = list(session.exec(select(ExtraRepasse)))
    if not rows:
        return pd.DataFrame(columns=["id","extra_date","month_ref","description","amount","is_recurring","student_id"])
    df = pd.DataFrame([r.model_dump() for r in rows])
    if month_ref and not df.empty:
        sel_y, sel_m = map(int, month_ref.split("-"))
        def applies(row):
            if row.get("month_ref") == month_ref:
                return True
            if row.get("is_recurring"):
                d = pd.to_datetime(row.get("extra_date"))
                return (d.year < sel_y) or (d.year == sel_y and d.month <= sel_m)
            return False
        df = df[df.apply(applies, axis=1)]
    return df

# -----------------------------------------------
# Utils
# -----------------------------------------------
def fmt_date(d):
    if d is None or (isinstance(d, float) and pd.isna(d)): return None
    if isinstance(d, (date, datetime)): return d.strftime("%d/%m/%Y")
    try: return pd.to_datetime(d, dayfirst=True).strftime("%d/%m/%Y")
    except Exception: return str(d)

def idade_atual(birth_date):
    """Idade completa em anos baseada na data de nascimento."""
    if not birth_date:
        return 0
    today = date.today()
    years = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return max(0, years)

def tempo_meses(student):
    """Total de meses desde a data de início do treino até hoje."""
    if not student or not getattr(student, "start_date", None):
        return 0
    start = student.start_date
    today = date.today()
    months = (today.year - start.year) * 12 + (today.month - start.month)
    if today.day < start.day:
        months -= 1
    return max(0, months)

# --- Helpers de consulta ---
def get_coaches():
    with Session(engine) as s:
        try:
            stmt = select(Coach).order_by(Coach.name)
            return list(s.exec(stmt))
        except OperationalError:
            try:
                ensure_coach_full_pass_column()
            except Exception:
                pass
            stmt = select(Coach).order_by(Coach.name)
            return list(s.exec(stmt))

def get_train_slots():
    with Session(engine) as s:
        stmt = select(TrainSlot).order_by(TrainSlot.label)
        return list(s.exec(stmt))

def get_students(only_active: bool = True):
    with Session(engine) as s:
        q = select(Student)
        if only_active:
            q = q.where(Student.active == True)
        q = q.order_by(Student.name)
        return list(s.exec(q))

def update_student(
    student_id: int,
    name: str | None = None,
    birth_date: date | None = None,
    start_date: date | None = None,
    monthly_fee: float | None = None,
    active: bool | None = None,
    coach_id: int | None = None,
    train_slot_id: int | None = None,
    master_percent: float | None = None,
    master_percent_override: float | None = None,
    grade: str | None = None,
    grade_date: date | None = None,
) -> None:
    """Atualiza campos do aluno conforme kwargs não nulos."""
    with Session(engine) as session:
        obj = session.get(Student, student_id)
        if not obj:
            return
        if name is not None: obj.name = name
        if birth_date is not None: obj.birth_date = birth_date
        if start_date is not None: obj.start_date = start_date
        if monthly_fee is not None: obj.monthly_fee = monthly_fee
        if active is not None: obj.active = active
        if coach_id is not None: obj.coach_id = coach_id
        if train_slot_id is not None: obj.train_slot_id = train_slot_id
        if master_percent is not None: obj.master_percent = master_percent
        if master_percent_override is not None: obj.master_percent_override = master_percent_override
        if grade is not None: obj.grade = grade
        if grade_date is not None: obj.grade_date = grade_date
        session.add(obj); session.commit()






def month_key(d: date) -> str:
    return d.strftime("%Y-%m")

def money(x: float, sym: str) -> str:
    try:
        return f"{sym} {float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return f"{sym} 0,00"

def fmt_percent(dec):
    if dec is None: 
        return ""
    try:
        return f"{float(dec)*100:.0f}%"
    except Exception:
        return ""

def fmt_duration_months(m):
    if m is None:
        return None
    try:
        m = int(m)
    except Exception:
        return None
    anos = m // 12
    meses = m % 12
    if anos <= 0:
        return f"{meses} mês" if meses == 1 else f"{meses} meses"
    a_txt = "ano" if anos == 1 else "anos"
    m_txt = "mês" if meses == 1 else "meses"
    return f"{anos} {a_txt} e {meses} {m_txt}"





def delete_student(student_id: int) -> int:
    """Exclui o aluno e seus pagamentos/graduações associados (via ON DELETE RESTRICT/NULL se houver)."""
    with Session(engine) as s:
        obj = s.get(Student, student_id)
        if not obj:
            return 0
        s.delete(obj)
        s.commit()
        return 1
def delete_payments(payment_ids: list[int]) -> int:
    """Delete payments by primary key IDs."""
    if not payment_ids:
        return 0
    deleted = 0
    with Session(engine) as session:
        for pid in payment_ids:
            try:
                p = session.get(Payment, int(pid))
                if p:
                    session.delete(p)
                    deleted += 1
            except Exception:
                pass
        session.commit()
    return deleted

def delete_all_payments_month(month_ref: str) -> int:
    """Delete ALL payments for a given month_ref (AAAA-MM)."""
    n = 0
    with Session(engine) as session:
        rows = list(session.exec(select(Payment).where(Payment.month_ref == month_ref)))
        for p in rows:
            session.delete(p); n += 1
        session.commit()
    return n


def delete_graduations(grad_ids: list[int]) -> int:
    """Exclui graduações pelos IDs e atualiza a graduação atual do(s) aluno(s)."""
    if not grad_ids:
        return 0
    affected_students = set()
    deleted = 0
    with Session(engine) as session:
        for gid in grad_ids:
            try:
                g = session.get(Graduation, int(gid))
                if g:
                    affected_students.add(g.student_id)
                    session.delete(g)
                    deleted += 1
            except Exception:
                pass
        session.commit()
    # Recalcula graduação atual para cada aluno afetado
    for sid in affected_students:
        try:
            refresh_student_grade(int(sid))
        except Exception:
            pass
    return deleted

# -----------------------------------------------
# UI base
# -----------------------------------------------
ensure_coach_full_pass_column(); cfg = get_settings()

with st.sidebar:
    st.caption("Versão: v2.12.20")
    st.caption(f"Script: {os.path.basename(__file__)}")
    st.caption("Logo: OK" if os.path.exists("logo.png") else "Logo: arquivo 'logo.png' não encontrado")
    st.header("Navegação")
    page = st.radio("Ir para:", [
        "Alunos", "Graduações", "Receber Pagamento",
        "Extras (Repasse)", "Relatórios", "Importar / Exportar", "Configurações"
    ])

# -----------------------------------------------
# Página: Alunos
# -----------------------------------------------
if page == "Alunos":
    st.subheader("Cadastro e gestão de alunos")

    # -------- Novo aluno --------
    st.markdown("### 🧾 Novo aluno")
    col = st.columns(3)
    with col[0]:
        n_name = st.text_input("Nome", key="new_name")
    with col[1]:
        n_birth = st.date_input("Data de nascimento (DD/MM/AAAA)", key="new_birth")
    with col[2]:
        n_start = st.date_input("Data início treino (DD/MM/AAAA)", key="new_start")
    col2 = st.columns(3)
    with col2[0]:
        n_fee = st.number_input("Mensalidade (R$)", min_value=0.0, step=10.0, key="new_fee")
    with col2[1]:
        n_coach = st.selectbox("Professor responsável", options=["(selecione)"] + [c.name for c in get_coaches()], key="new_coach")
    with col2[2]:
        n_slot = st.selectbox("Horário de treino", options=["(selecione)"] + [t.label for t in get_train_slots()], key="new_slot")
    use_override = st.checkbox("Usar repasse específico?", key="new_use_override")
    override = st.number_input("Repasse específico (%) — opcional", min_value=0.0, max_value=100.0, step=1.0, value=0.0, key="new_override")
    if st.button("➕ Adicionar aluno", use_container_width=True, key="btn_add_aluno"):
        coach_id = next((c.id for c in get_coaches() if c.name == n_coach), None) if n_coach != "(selecione)" else None
        slot_id = next((t.id for t in get_train_slots() if t.label == n_slot), None) if n_slot != "(selecione)" else None
        add_student(name=n_name, birth_date=n_birth, start_date=n_start, monthly_fee=float(n_fee or 0.0), active=True, coach_id=coach_id, train_slot_id=slot_id, master_percent_override=(override/100.0 if use_override else None))
        st.success("Aluno cadastrado!")
        st.rerun()

    st.markdown("---")

    # -------- Lista de alunos --------
    month_badge = st.text_input("Mês referência para status de pagamento (AAAA-MM)", value=month_key(date.today()), key="alunos_month_status")
    pids = paid_ids_for_month(month_badge)
    status_filter = st.multiselect("Status", ["Ativos", "Inativos"], default=["Ativos"])

    students = get_students(False)
    if not students:
        st.info("Sem alunos cadastrados.")
    else:
        rows = []
        coaches = get_coaches()
        slots = get_train_slots()
        id2coach = {c.id: c.name for c in coaches}
        id2slot = {t.id: t.label for t in slots}
        for s in students:
            if ("Ativos" in status_filter and s.active) or ("Inativos" in status_filter and not s.active):
                idade = idade_atual(s.birth_date)
                meses = tempo_meses(s)
                rows.append({
                    "ID": s.id,
                    "Selecionar": False,
                    "Nome": (s.name + (" — ✅ pago" if s.id in pids else " — ❌ não pago")),
                    "Idade": idade,
                    "Tempo de treino": fmt_duration_months(meses),
                    "Mensalidade": money(float(s.monthly_fee or 0.0), cfg.currency_symbol),
                    "Repasse específico": fmt_percent(s.master_percent_override),
                    "Professor": id2coach.get(s.coach_id, "—"),
                    "Horário": id2slot.get(s.train_slot_id, "—"),
                    "Graduação atual": s.grade or "Branca",
                    "Data da graduação": fmt_date(s.grade_date) if s.grade_date else "—",
                    "Ativo?": "Sim" if s.active else "Não",
                })
        import pandas as pd
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ID","Selecionar","Nome","Idade","Tempo de treino","Mensalidade","Repasse específico","Professor","Horário","Graduação atual","Data da graduação","Ativo?"])
        df = df.sort_values("Nome") if not df.empty else df

        st.markdown("### 👥 Alunos")
        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Selecionar": st.column_config.CheckboxColumn("Selecionar"),
                "ID": st.column_config.NumberColumn("ID", help="Identificador interno", disabled=True),
                "Nome": st.column_config.TextColumn("Nome", disabled=True),
                "Idade": st.column_config.NumberColumn("Idade", disabled=True),
                "Tempo de treino": st.column_config.TextColumn("Tempo de treino", disabled=True),
                "Mensalidade": st.column_config.TextColumn("Mensalidade", disabled=True),
                "Repasse específico": st.column_config.TextColumn("Repasse específico", disabled=True),
                "Professor": st.column_config.TextColumn("Professor", disabled=True),
                "Horário": st.column_config.TextColumn("Horário", disabled=True),
                "Graduação atual": st.column_config.TextColumn("Graduação atual", disabled=True),
                "Data da graduação": st.column_config.TextColumn("Data da graduação", disabled=True),
                "Ativo?": st.column_config.TextColumn("Ativo?", disabled=True),
            },
        )

        # -------- Painel de edição (aparece quando selecionar um aluno na tabela) --------
        sel_ids = edited.loc[edited["Selecionar"] == True, "ID"].tolist() if not edited.empty else []
        if len(sel_ids) == 0:
            st.info("Selecione um aluno na tabela para editar.")
        else:
            if len(sel_ids) > 1:
                st.warning("Você selecionou mais de um aluno. Editando o primeiro da lista.")
            sel_id = int(sel_ids[0])
            stu = get_student_by_id(sel_id)
            if not stu:
                st.error("Aluno não encontrado.")
            else:
                st.markdown("### ✏️ Editar aluno selecionado")
                latest_grade = stu.grade or "Branca"
                latest_grade_date = fmt_date(stu.grade_date) if stu.grade_date else "—"
                col1, col2 = st.columns(2)
                with col1:
                    e_name = st.text_input("Nome", value=stu.name, key=f"edit_name_{stu.id}")
                    e_fee = st.number_input("Mensalidade (R$)", step=10.0, value=float(stu.monthly_fee or 0.0), key=f"edit_fee_{stu.id}")
                    e_active = st.checkbox("Ativo?", value=bool(stu.active), key=f"edit_active_{stu.id}")
                    e_birth = st.date_input("Data de nascimento (DD/MM/AAAA)", value=stu.birth_date, key=f"edit_birth_{stu.id}")
                    e_start = st.date_input("Data início treino (DD/MM/AAAA)", value=stu.start_date, key=f"edit_start_{stu.id}")
                with col2:
                    st.text_input("Graduação atual (derivada)", value=latest_grade, disabled=True)
                    st.text_input("Data da graduação (derivada)", value=latest_grade_date, disabled=True)
                    coach_names = ["(selecione)"] + [c.name for c in coaches]
                    cmap = {c.id: c.name for c in coaches}
                    smap = {t.id: t.label for t in slots}
                    idx = coach_names.index(cmap.get(stu.coach_id, "")) if (getattr(stu, "coach_id", None) in cmap) else 0
                    e_coach_sel = st.selectbox("Professor responsável", options=coach_names, index=idx, key=f"edit_coach_sel_{stu.id}")
                    slot_labels = ["(selecione)"] + [t.label for t in slots]
                    idxs = slot_labels.index(smap.get(stu.train_slot_id, "")) if (getattr(stu, "train_slot_id", None) in smap) else 0
                    e_slot_sel = st.selectbox("Horário de treino", options=slot_labels, index=idxs, key=f"edit_slot_sel_{stu.id}")
                    e_use_override = st.checkbox("Usar repasse específico?", value=(stu.master_percent_override is not None), key=f"edit_use_override_{stu.id}")
                    e_override = st.number_input("Repasse específico (%) — opcional", min_value=0.0, max_value=100.0, step=1.0, value=float((stu.master_percent_override or 0.0)*100.0), key=f"edit_override_num_{stu.id}")
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("💾 Atualizar", use_container_width=True, key=f"btn_update_{stu.id}"):
                        coach_id = next((c.id for c in coaches if c.name == e_coach_sel), None) if e_coach_sel != "(selecione)" else None
                        slot_id = next((t.id for t in slots if t.label == e_slot_sel), None) if e_slot_sel != "(selecione)" else None
                        update_student(stu.id,
                                       name=e_name,
                                       monthly_fee=float(e_fee or 0.0),
                                       active=bool(e_active),
                                       birth_date=e_birth,
                                       start_date=e_start,
                                       coach_id=coach_id,
                                       train_slot_id=slot_id,
                                       master_percent_override=(e_override/100.0 if e_use_override else None))
                        st.success("Aluno atualizado.")
                        st.rerun()
                with b2:
                    if st.button(("🟢 Ativar" if not stu.active else "⚪ Desativar"), use_container_width=True, key=f"btn_toggle_{stu.id}"):
                        update_student(stu.id, active=(not stu.active))
                        st.rerun()
                with b3:
                    if st.button("🗑️ Excluir aluno", use_container_width=True, key=f"btn_del_{stu.id}"):
                        delete_student(stu.id)
                        st.success("Aluno excluído.")
                        st.rerun()
# Página: Graduações

# -----------------------------------------------
elif page == "Graduações":
    st.subheader("Histórico de Graduações por Aluno")
    students = get_students(False)
    if not students:
        st.info("Cadastre alunos primeiro.")
    else:
        sid_map = {f"{s.name} (ID {s.id})": s.id for s in students}
        sid = sid_map[st.selectbox("Aluno", list(sid_map.keys()))]
        gdf = list_graduations(sid)
        st.markdown("#### Histórico")
        if gdf.empty:
            st.info("Sem graduações registradas para este aluno.")
        else:
            gdf_disp = gdf.copy()
            gdf_disp["grade_date"] = gdf_disp["grade_date"].apply(fmt_date)
            gdf_disp = gdf_disp.rename(columns={"grade":"Graduação","grade_date":"Data","notes":"Observações"})
            cols = [c for c in ["Data","Graduação","Observações"] if c in gdf_disp.columns]
            st.dataframe(gdf_disp[cols], use_container_width=True)

        st.markdown("#### Excluir graduações")
        if not gdf.empty:
            try:
                opts2 = [f'ID {int(r["id"])} — {fmt_date(r["grade_date"])} — {r["grade"]}' for _, r in gdf.iterrows()]
                map_ids2 = {opts2[i]: int(gdf.iloc[i]["id"]) for i in range(len(opts2))}
                pick2 = st.multiselect("Selecionar graduações para excluir", opts2, key="del_grads_page")
                if st.button("🗑️ Excluir selecionadas", use_container_width=True, key="btn_del_grads_page"):
                    cnt = delete_graduations([map_ids2[o] for o in pick2])
                    st.success(f"{cnt} graduação(ões) excluída(s).")
                    st.rerun()
            except Exception:
                pass

        st.markdown("#### Adicionar nova graduação")
        colg1, colg2 = st.columns(2)
        with colg1:
            gg = st.selectbox("Graduação", options=ALLOWED_GRADES)
        with colg2:
            gd = st.date_input("Data da graduação (DD/MM/AAAA)", value=date.today())
        gnotes = st.text_input("Observações (opcional)")
        if st.button("➕ Adicionar graduação", use_container_width=True):
            add_graduation(sid, gg, gd, gnotes if gnotes else None)
            refresh_student_grade(sid)
            st.success("Graduação adicionada!")
            st.rerun()

# -----------------------------------------------
# Página: Receber Pagamento
# -----------------------------------------------
elif page == "Receber Pagamento":
    st.subheader("Dar baixa de pagamento (vários alunos de uma vez)")

    # Filtros
    month_ref = st.text_input("Mês de referência (AAAA-MM)", value=month_key(date.today()))
    coaches = get_coaches()
    coach_filter = st.selectbox("Filtrar por professor (opcional)", options=["(todos)"] + [c.name for c in coaches])
    pf = st.radio("Filtro de pagamento do mês", ["Todos", "Pagaram", "Não pagaram"], horizontal=True)

    # Carregar alunos e status
    students = get_students(False)
    if coach_filter != "(todos)":
        coach_id = next((c.id for c in coaches if c.name == coach_filter), None)
        students = [s for s in students if s.coach_id == coach_id]
    paid_df = get_payments(month_ref=month_ref)
    paid_ids = set(paid_df["student_id"].tolist()) if paid_df is not None and not paid_df.empty else set()

    if pf == "Pagaram":
        students = [s for s in students if s.id in paid_ids]
    elif pf == "Não pagaram":
        students = [s for s in students if s.id not in paid_ids]

    if not students:
        st.info("Nenhum aluno encontrado para o filtro atual.")
    else:
        # Seleção de alunos para registrar pagamento
        labels = [f"{s.name} (ID {s.id}) — Mensalidade: {money(float(s.monthly_fee or 0.0), cfg.currency_symbol)}" for s in students]
        id_map = {labels[i]: students[i].id for i in range(len(students))}
        chosen_labels = st.multiselect("Alunos", labels, key="pay_labels")
        chosen_ids = [int(id_map[l]) for l in chosen_labels]

        col1, col2, col3 = st.columns(3)
        with col1:
            paid_date_val = st.date_input("Data do pagamento (DD/MM/AAAA)", value=date.today())
            use_each_fee = st.checkbox("Usar mensalidade de cada aluno", value=True)
            amount_common = None
            if not use_each_fee:
                amount_common = st.number_input("Valor pago (aplica a todos) — R$", step=10.0, value=0.0)
        with col2:
            base_percent = st.number_input("Percentual de repasse base (%)", min_value=0.0, max_value=100.0, step=1.0, value=float(cfg.default_master_percent*100))
            method = st.selectbox("Forma de pagamento", ["Dinheiro","PIX","Cartão","Outros"])
        with col3:
            notes = st.text_input("Observações", "")
            st.caption("Dica: se usar 'Mensalidade de cada aluno', o sistema buscará o valor cadastrado de cada um.")
            if st.button("✅ Confirmar pagamento", use_container_width=True, key="btn_conf_pay"):
                if not chosen_ids:
                    st.warning("Selecione ao menos um aluno.")
                else:
                    n = record_payment_batch(chosen_ids, paid_date_val, amount_common, use_each_fee, method, notes, month_ref, base_percent/100.0, 0.0)
                    st.success(f"{n} pagamento(s) registrado(s).")
                    st.rerun()

        st.markdown("---")
        st.markdown("### 🧾 Pagamentos do mês")
        dfp = get_payments(month_ref=month_ref)
        if dfp is None or dfp.empty:
            st.info("Sem pagamentos neste mês.")
        else:
            dfp["Aluno"] = dfp["student_id"].apply(lambda i: get_student_by_id(int(i)).name if get_student_by_id(int(i)) else f"ID {i}")
            dfp["Valor pago"] = dfp["amount"].apply(lambda x: money(float(x or 0.0), cfg.currency_symbol))
            dfp["Repasse"] = dfp["master_amount"].apply(lambda x: money(float(x or 0.0), cfg.currency_symbol))
            dfp["Data"] = dfp["paid_date"].apply(fmt_date)
            show_cols = ["id","Aluno","Data","Valor pago","Repasse","method","notes"]
            st.dataframe(dfp[show_cols].rename(columns={"id":"ID","method":"Forma","notes":"Obs"}), use_container_width=True)

            ids_to_delete = st.multiselect("Selecionar pagamentos para excluir", dfp["id"].tolist(), key="del_pays_ids")
            colx, coly = st.columns([1,1])
            with colx:
                if st.button("🗑️ Excluir selecionados", use_container_width=True, key="btn_del_sel"):
                    if ids_to_delete:
                        n = delete_payments(ids_to_delete)
                        st.success(f"{n} pagamento(s) excluído(s).")
                        st.rerun()
                    else:
                        st.warning("Nenhum pagamento selecionado.")
            with coly:
                if st.button("🧹 Excluir todos deste mês", use_container_width=True, key="btn_del_all_month"):
                    all_ids = dfp["id"].tolist()
                    if all_ids:
                        n = delete_payments(all_ids)
                        st.success(f"Todos os {n} pagamentos deste mês foram excluídos.")
                        st.rerun()
                    else:
                        st.info("Não há pagamentos neste mês.")
                st.caption("Dica: use o filtro de mês acima. Excluir todos remove todos deste mês.")
# -----------------------------------------------

# -----------------------------------------------
# Página: Extras (Repasse)
# -----------------------------------------------
elif page == "Extras (Repasse)":
    st.subheader("Lançamentos extras para repasse")
    month = st.text_input("Mês de referência (AAAA-MM)", value=month_key(date.today()), key="extras_month")
    st.caption("Valores podem ser positivos (acréscimos) ou negativos (descontos).")
    st.markdown("### ➕ Novo lançamento")
    col1, col2 = st.columns([2,1])
    with col1:
        e_desc = st.text_input("Descrição")
        e_date = st.date_input("Data do lançamento (DD/MM/AAAA)", value=date.today())
    with col2:
        e_amount = st.number_input("Valor (R$)", step=10.0, value=0.0)
        e_rec = st.checkbox("Fixo mês a mês (recorrente)?", value=False)
    # aluno opcional
    students = get_students(False)
    sid = None
    if students:
        label = st.selectbox("Aluno (opcional — se for geral, deixe em branco)", options=["(geral)"] + [f"{s.name} (ID {s.id})" for s in students])
        if label != "(geral)":
            sid = int(label.split("ID")[-1].strip(") ").strip())
    if st.button("💾 Salvar extra", use_container_width=True):
        if not e_desc.strip():
            st.warning("Informe a descrição.")
        else:
            add_extra_repasse(description=e_desc, amount=e_amount, date_val=e_date, month_ref=month, is_recurring=e_rec, student_id=sid)
            st.success("Extra lançado.")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Lançamentos do mês (e recorrentes)")
    dfe = get_extras(month_ref=month)
    if dfe is None or dfe.empty:
        st.info("Sem lançamentos para este mês.")
    else:
        dfe = dfe.copy()
        dfe["Data"] = pd.to_datetime(dfe["extra_date"]).dt.strftime("%d/%m/%Y")
        dfe["Aluno"] = dfe["student_id"].apply(lambda i: get_student_by_id(int(i)).name if pd.notna(i) and get_student_by_id(int(i)) else "Outros")
        dfe["Valor (R$)"] = dfe["amount"].apply(lambda x: money(float(x or 0.0), cfg.currency_symbol))
        dfe["Recorrente?"] = dfe["is_recurring"].apply(lambda b: "Sim" if b else "Não")
        st.dataframe(dfe[["id","Data","Aluno","description","Valor (R$)","Recorrente?"]].rename(columns={"id":"ID","description":"Descrição"}), use_container_width=True)

        dels = st.multiselect("Selecionar lançamentos para excluir", dfe["id"].tolist(), key="del_extras_ids")
        cdx1, cdx2 = st.columns([1,1])
        with cdx1:
            if st.button("🗑️ Excluir selecionados", use_container_width=True):
                if dels:
                    n = delete_extra_repasse(dels)
                    st.success(f"{n} lançamento(s) excluído(s).")
                    st.rerun()
                else:
                    st.warning("Nenhum lançamento selecionado.")
        with cdx2:
            if st.button("🧹 Excluir todos listados", use_container_width=True):
                n = delete_extra_repasse(dfe["id"].tolist())
                st.success(f"Todos os {n} lançamentos listados foram excluídos.")
                st.rerun()
# Página: Relatórios
# -----------------------------------------------

elif page == "Relatórios":
    st.subheader("Relatórios de repasse")
    # --- visual cards css ---
    st.markdown("""
    <style>
    :root{
      --card-bg:#ffffff;
      --card-text:#111827;
      --card-muted:#6b7280;
      --card-border:#e5e7eb;
    }
    @media (prefers-color-scheme: dark){
      :root{
        --card-bg:#111827;
        --card-text:#f9fafb;
        --card-muted:#9ca3af;
        --card-border:#374151;
      }
    }
    .card{
      border-radius:12px;
      padding:16px 18px;
      background:var(--card-bg);
      color:var(--card-text);
      border:1px solid var(--card-border);
      box-shadow:0 2px 8px rgba(0,0,0,.04);
      text-align:center;
    }
    .card .label{font-size:12px;color:var(--card-muted);letter-spacing:.2px}
    .card .value{font-size:24px;font-weight:700;margin-top:6px}
    .card--mensal{border-top:4px solid #3b82f6}
    .card--extras{border-top:4px solid #f59e0b}
    .card--total{border-top:4px solid #10b981}
    </style>
    """, unsafe_allow_html=True)
    mes_ref = st.text_input("Mês de referência (AAAA-MM)", value=month_key(date.today()), key="rel_mes_ref")

    # Filtro por professor
    coaches = get_coaches()
    coach_filter = st.selectbox("Filtrar por professor", options=["(todos)"] + [c.name for c in coaches], index=0)
    coach_id_filter = next((c.id for c in coaches if c.name == coach_filter), None) if coach_filter != "(todos)" else None

    # Mapas auxiliares
    students_all = get_students(False)
    id2stu = {s.id: s for s in students_all}
    id2name = {s.id: s.name for s in students_all}

    # -----------------------------------------------------
    # 1) RELATÓRIO DE REPASSE DE MENSALIDADES (DETALHADO)
    # -----------------------------------------------------
    st.markdown("## 💵 Relatório de repasse de mensalidades (detalhado)")
    dfp = get_payments(month_ref=mes_ref)
    if dfp is not None and not dfp.empty:
        pag = dfp.copy()
        # aplicar filtro por professor
        if coach_id_filter is not None:
            def _coach_of_payment(stu_id):
                stn = id2stu.get(int(stu_id)) if pd.notna(stu_id) else None
                return stn.coach_id if stn else None
            pag = pag[pag["student_id"].apply(_coach_of_payment) == coach_id_filter]

        if not pag.empty:
            # enriquecer
            pag["Aluno"] = pag["student_id"].map(id2name)
            pag["Idade"] = pag["student_id"].apply(lambda i: idade_atual(id2stu[int(i)].birth_date) if int(i) in id2stu else "")
            pag["Tempo de treino"] = pag["student_id"].apply(lambda i: fmt_duration_months(tempo_meses(id2stu[int(i)])) if int(i) in id2stu else "")
            pag["Graduação"] = pag["student_id"].apply(lambda i: (id2stu[int(i)].grade or "Branca") if int(i) in id2stu else "")
            pag["Data"] = pag["paid_date"].apply(fmt_date)
            pag["Valor pago"] = pag["amount"].apply(lambda x: money(float(x or 0.0), cfg.currency_symbol))
            pag["Repasse"] = pag["master_amount"].apply(lambda x: money(float(x or 0.0), cfg.currency_symbol))

            cols_show = ["Aluno","Idade","Tempo de treino","Graduação","Data","Valor pago","Repasse"]
            st.dataframe(pag[cols_show], use_container_width=True)

            total_pagamentos_repasse = float(pag["master_amount"].astype(float).sum())
            c1, c2, c3 = st.columns([1,2,1])
            with c2:
                st.markdown(f'<div class="card card--mensal"><div class="label">💵 Total (repasse sobre mensalidades)</div><div class="value">{money(total_pagamentos_repasse, cfg.currency_symbol)}</div></div>', unsafe_allow_html=True)

            # Export CSV detalhado de mensalidades
            out_csv_pag = pag[["student_id","Aluno","Idade","Tempo de treino","Graduação","paid_date","amount","master_amount","method","notes"]].rename(columns={
                "paid_date":"Data","amount":"Valor pago (num)","master_amount":"Repasse (num)"
            }).to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Exportar mensalidades (CSV)", data=out_csv_pag, file_name=f"mensalidades_detalhe_{mes_ref}.csv", mime="text/csv", use_container_width=True)
        else:
            st.info("Sem pagamentos para o filtro atual.")
            total_pagamentos_repasse = 0.0
    else:
        st.info("Sem pagamentos neste mês.")
        total_pagamentos_repasse = 0.0

    st.markdown("---")

    # ------------------------------------
    # 2) RELATÓRIO DE EXTRAS (DETALHADO)
    # ------------------------------------
    st.markdown("## ➕ Relatório de extras (detalhado)")
    dfe = get_extras(month_ref=mes_ref)
    if dfe is not None and not dfe.empty:
        extras = dfe.copy()

        # Aplicar filtro por professor: mantém extras do(s) alunos do professor.
        # "Outros" (sem student_id) só aparece quando filtro = (todos)
        if coach_id_filter is not None:
            def _coach_match(v):
                try:
                    vv = int(v)
                    stn = id2stu.get(vv)
                    return (stn.coach_id == coach_id_filter) if stn else False
                except Exception:
                    return False
            extras = extras[extras["student_id"].apply(_coach_match)]
        # Enriquecer exibição
        extras["Data"] = pd.to_datetime(extras["extra_date"]).dt.strftime("%d/%m/%Y")
        def _aluno_nome_row(v):
            try:
                if pd.isna(v):  # na: "Outros"
                    return "Outros"
            except Exception:
                pass
            try:
                vv = int(v)
                return id2name.get(vv, f"ID {vv}")
            except Exception:
                return "Outros"
        extras["Aluno"] = extras["student_id"].apply(_aluno_nome_row)
        extras["Descrição"] = extras["description"]
        extras["Valor (R$)"] = extras["amount"].apply(lambda x: money(float(x or 0.0), cfg.currency_symbol))
        extras["Recorrente?"] = extras["is_recurring"].apply(lambda b: "Sim" if b else "Não")

        cols_ex = ["Data","Aluno","Descrição","Valor (R$)","Recorrente?"]
        # Se filtro por professor estiver ativo, "Outros" são ocultos
        if coach_id_filter is not None:
            extras = extras[extras["Aluno"] != "Outros"]

        st.dataframe(extras[cols_ex], use_container_width=True)

        total_extras = float(extras["amount"].astype(float).sum()) if not extras.empty else 0.0
        e1, e2, e3 = st.columns([1,2,1])
        with e2:
            st.markdown(f'<div class="card card--extras"><div class="label">➕ Total (extras detalhados)</div><div class="value">{money(total_extras, cfg.currency_symbol)}</div></div>', unsafe_allow_html=True)

        # Export CSV de extras
        out_csv_ext = extras[["Data","Aluno","Descrição","Valor (R$)","Recorrente?"]].to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Exportar extras (CSV)", data=out_csv_ext, file_name=f"extras_detalhe_{mes_ref}.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("Sem extras neste mês.")
        total_extras = 0.0

    st.markdown("---")

    # ----------------------------
    # SOMATÓRIO FINAL (2 tabelas)
    # ----------------------------
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="card card--mensal"><div class="label">💵 Mensalidades</div><div class="value">{money(total_pagamentos_repasse, cfg.currency_symbol)}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="card card--extras"><div class="label">➕ Extras</div><div class="value">{money(total_extras, cfg.currency_symbol)}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="card card--total"><div class="label">🧮 Total geral</div><div class="value">{money(total_pagamentos_repasse + total_extras, cfg.currency_symbol)}</div></div>', unsafe_allow_html=True)


elif page == "Importar / Exportar":
    st.subheader("Importar alunos de CSV")
    st.caption("Colunas: name, birth_date(DD/MM/AAAA/AAAA-MM-DD), start_date(DD/MM/AAAA/AAAA-MM-DD), active(True/False/X), monthly_fee, master_percent_override(0-1), coach_name, train_slot_label")
    file = st.file_uploader("CSV de alunos", type=["csv"])
    if file is not None:
        df = pd.read_csv(file); ok, fail = 0, 0
        for _, row in df.iterrows():
            try:
                def parse_dt(x):
                    if pd.isna(x): return None
                    try: return pd.to_datetime(x, dayfirst=True).date()
                    except Exception: return None
                bd = parse_dt(row.get("birth_date")); sd = parse_dt(row.get("start_date"))
                active = str(row.get("active")).strip().upper() in ["TRUE","1","X","SIM"]
                monthly_fee = float(row.get("monthly_fee")) if not pd.isna(row.get("monthly_fee")) else 0.0
                override_raw = row.get("master_percent_override"); override = None
                if override_raw is not None and str(override_raw).strip() != "" and not pd.isna(override_raw): override = float(override_raw)
                coach_name = None if pd.isna(row.get("coach_name")) else str(row.get("coach_name")).strip()
                train_label = None if pd.isna(row.get("train_slot_label")) else str(row.get("train_slot_label")).strip()
                coach_id = add_coach(coach_name) if coach_name else None
                slot_id = add_train_slot(train_label) if train_label else None
                add_student(Student(name=str(row.get("name")), birth_date=bd, start_date=sd, active=active, monthly_fee=monthly_fee,
                                    master_percent_override=override, coach_id=coach_id, train_slot_id=slot_id))
                ok += 1
            except Exception:
                fail += 1
        st.success(f"Importação finalizada — {ok} inseridos, {fail} erros.")
    st.markdown("---"); st.subheader("Exportar base completa")
    students = get_students(False); df_students = pd.DataFrame([s.model_dump() for s in students]) if students else pd.DataFrame()
    payments = get_payments(); extras = get_extras()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("⬇️ Baixar alunos.csv", data=df_students.to_csv(index=False).encode("utf-8-sig"), file_name="alunos.csv", mime="text/csv")
    with col2:
        st.download_button("⬇️ Baixar pagamentos.csv", data=payments.to_csv(index=False).encode("utf-8-sig"), file_name="pagamentos.csv", mime="text/csv")
    with col3:
        st.download_button("⬇️ Baixar extras.csv", data=extras.to_csv(index=False).encode("utf-8-sig"), file_name="extras.csv", mime="text/csv")

# -----------------------------------------------
# Página: Configurações
# -----------------------------------------------

elif page == "Configurações":
    st.subheader("Preferências")
    cfg = get_settings()
    perc = st.number_input("Percentual padrão de repasse à equipe/mestre (%)", min_value=0.0, max_value=100.0, step=1.0, value=float(cfg.default_master_percent*100))
    sym = st.text_input("Símbolo de moeda", value=cfg.currency_symbol)
    if st.button("💾 Salvar configurações", use_container_width=True):
        save_settings(perc/100.0, sym); st.success("Configurações salvas!")
    st.markdown("---")
    st.subheader("📋 Cadastros auxiliares")
    st.markdown("#### 👨‍🏫 Professores")
    new_coach = st.text_input("Novo professor (nome)")
    new_full = st.checkbox("Repasse completo (100%)", key="new_coach_full")
    if st.button("➕ Adicionar professor"):
        if not new_coach.strip(): st.warning("Informe um nome.")
        else:
            add_coach(new_coach.strip(), full_pass=bool(new_full));
            st.success("Professor adicionado!")
    clist = list_coaches()
    if clist:
        for c in clist:
            c1, c2, c3 = st.columns([4,2,1])
            c1.write(f"- {c.name}")
            toggled = c2.checkbox("Repasse 100%", value=bool(getattr(c, 'full_pass', False)), key=f'coach_full_{c.id}')
            if c2.button("Salvar", key=f'savec_{c.id}'):
                set_coach_full_pass(c.id, bool(toggled)); st.success("Repasse atualizado!")
            if c3.button("Excluir", key=f"delc_{c.id}"):
                delete_coach(c.id); st.warning(f"Professor '{c.name}' excluído.")
    else: st.info("Nenhum professor cadastrado.")
    st.markdown("---"); st.markdown("#### 🕒 Horários de Treino")
    new_slot = st.text_input("Novo horário (ex.: Ter/Qui 19h-20h)")
    if st.button("➕ Adicionar horário"):
        if not new_slot.strip(): st.warning("Informe um horário/descrição.")
        else: add_train_slot(new_slot.strip()); st.success("Horário adicionado!")
    slist = list_train_slots()
    if slist:
        for t in slist:
            t1, t2 = st.columns([4,1]); t1.write(f"- {t.label}")
            if t2.button("Excluir", key=f"delt_{t.id}"):
                delete_train_slot(t.id); st.warning(f"Horário '{t.label}' excluído.")
    else: st.info("Nenhum horário cadastrado.")

def add_extra_repasse(description: str, amount: float, date_val: date, month_ref: str, is_recurring: bool = False, student_id: Optional[int] = None) -> int:
    """Cria um lançamento extra. 'amount' pode ser negativo (desconto). Se is_recurring=True, aplica mês a mês a partir do mês de lançamento."""
    with Session(engine) as session:
        e = ExtraRepasse(description=description.strip(),
                         amount=float(amount or 0.0),
                         extra_date=date_val,
                         month_ref=month_ref,
                         is_recurring=bool(is_recurring),
                         student_id=student_id)
        session.add(e); session.commit(); session.refresh(e)
        return int(e.id)


def update_extra_repasse(extra_id: int, **fields) -> int:
    with Session(engine) as session:
        obj = session.get(ExtraRepasse, extra_id)
        if not obj:
            return 0
        for k,v in fields.items():
            if hasattr(obj, k) and v is not None:
                setattr(obj, k, v)
        session.add(obj); session.commit()
        return 1


def delete_extra_repasse(ids: list[int]) -> int:
    if not ids: return 0
    with Session(engine) as session:
        count = 0
        for i in ids:
            obj = session.get(ExtraRepasse, int(i))
            if obj:
                session.delete(obj); count += 1
        session.commit()
        return count