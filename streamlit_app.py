# -*- coding: utf-8 -*-
import os, math
from datetime import date, datetime
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import (
    create_engine, MetaData, Table, select, insert, update, delete,
    text, or_
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoSuchTableError

# =========================================================
# CONFIG / BRAND
# =========================================================
APP_TITLE = "JAT - Gestão de alunos"

HERE = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
LOGO_PATH = os.path.join(HERE, "logo.png")
PAGE_ICON = LOGO_PATH if os.path.isfile(LOGO_PATH) else "🥋"

st.set_page_config(page_title=APP_TITLE, page_icon=PAGE_ICON, layout="wide")

# Se existir, mostra o logo no topo
if os.path.isfile(LOGO_PATH):
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        st.image(LOGO_PATH, use_container_width=True)
    with col_title:
        st.title(APP_TITLE)
else:
    st.title(APP_TITLE)

# =========================================================
# DB / TABLE NAMES
# =========================================================
DB_URL = (
    st.secrets.get("DATABASE_URL")
    or os.getenv("DATABASE_URL")
    or f"sqlite:///{os.path.join(HERE, 'muaythai.db')}"
)

T_STUDENT     = "student"
T_GRADUATION  = "graduation_history"
T_PAYMENT     = "payment"
T_EXTRA       = "extra_repasse"
T_SETTINGS    = "settings"
T_COACH       = "coach"
T_SLOT        = "train_slot"

BIRTH_MIN, BIRTH_MAX = date(1900,1,1), date.today()
TRAIN_MIN, TRAIN_MAX = date(1990,1,1), date.today()

GRADE_CHOICES = [
    "Branca","Amarelo","Amarelo e Branca","Verde","Verde e Branca",
    "Azul","Azul e Branca","Marrom","Marrom e Branca","Vermelha",
    "Vermelha e Branca","Preta"
]

# =========================================================
# ENGINE / REFLECTION
# =========================================================
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    return create_engine(DB_URL, pool_pre_ping=True, future=True)

engine: Engine = get_engine()
metadata = MetaData()

def reflect_table(name: str) -> Optional[Table]:
    try:
        return Table(name, metadata, autoload_with=engine)
    except NoSuchTableError:
        return None

def has_col(tbl: Optional[Table], col: str) -> bool:
    return (tbl is not None) and (col in tbl.c)

def _dialect() -> str:
    try:
        return engine.dialect.name or ""
    except Exception:
        return ""

# =========================================================
# FLASH MESSAGES (persistem após rerun)
# =========================================================
def flash(kind: str, msg: str):
    st.session_state.setdefault("_flash", []).append((kind, msg))

def show_flashes():
    if st.session_state.get("_flash"):
        for kind, msg in st.session_state["_flash"]:
            if   kind == "success": st.success(msg)
            elif kind == "warning": st.warning(msg)
            elif kind == "error":   st.error(msg)
            else:                   st.info(msg)
        st.session_state["_flash"] = []

show_flashes()

# =========================================================
# HELPERS
# =========================================================
def fmt_date(d: Optional[date]) -> str:
    if d in (None, "", "None"): return "—"
    if isinstance(d, str):
        try: d = datetime.fromisoformat(d).date()
        except Exception: return d
    return d.strftime("%d/%m/%Y")

def parse_date(v: Any) -> Optional[date]:
    if v in (None, "", "—"): return None
    if isinstance(v, date): return v
    try:
        if isinstance(v, str) and "/" in v:
            return datetime.strptime(v, "%d/%m/%Y").date()
        return datetime.fromisoformat(str(v)).date()
    except Exception:
        return None

def idade_atual(dn: Any) -> str:
    dn = parse_date(dn)
    if not dn: return "—"
    today = date.today()
    years = today.year - dn.year - ((today.month, today.day) < (dn.month, dn.day))
    return f"{years} anos"

def tempo_treino_fmt(dt_inicio: Any) -> str:
    di = parse_date(dt_inicio)
    if not di: return "—"
    today = date.today()
    months = (today.year - di.year) * 12 + (today.month - di.month)
    if today.day < di.day: months -= 1
    if months < 0: months = 0
    anos, meses = months // 12, months % 12
    return f"{anos} anos e {meses} meses" if anos > 0 else f"{meses} meses"

def money(v: Any) -> str:
    try: f = float(v or 0)
    except: f = 0.0
    s = f"{f:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

# =========================================================
# LOGIN
# =========================================================
def _do_login():
    st.sidebar.subheader("🔐 Login")
    u = st.sidebar.text_input("Usuário")
    p = st.sidebar.text_input("Senha", type="password")
    if st.sidebar.button("Entrar", type="primary", use_container_width=True):
        users_section = st.secrets.get("users", {}) or {
            "admin": st.secrets.get("admin", ""),
            "operador": st.secrets.get("operador", ""),
        }
        users = {
            "admin": {"pw": users_section.get("admin",""), "role": "admin"},
            "operador": {"pw": users_section.get("operador",""), "role": "operador"},
        }
        u_in, p_in = (u or "").strip().lower(), (p or "").strip()
        if u_in in users and p_in == users[u_in]["pw"] and p_in != "":
            st.session_state["user"], st.session_state["role"] = u_in, users[u_in]["role"]
            flash("success", f"Bem-vindo, {u_in}!")
            st.rerun()
        else:
            st.sidebar.error("Usuário ou senha inválidos.")
    st.stop()

if "role" not in st.session_state: _do_login()
with st.sidebar:
    st.caption(f"Usuário: **{st.session_state['user']}** · Perfil: **{st.session_state['role']}**")
    if st.button("Sair", use_container_width=True):
        for k in ("user","role"): st.session_state.pop(k, None)
        flash("info", "Sessão encerrada.")
        st.rerun()

def require_admin():
    if st.session_state["role"] != "admin":
        st.warning("Somente o administrador pode acessar esta seção.")
        st.stop()

# =========================================================
# SETTINGS / DATA LOADERS
# =========================================================
@st.cache_data(ttl=30)
def get_settings() -> Dict[str, Any]:
    tbl = reflect_table(T_SETTINGS)
    if tbl is None: return {}
    with engine.begin() as conn:
        row = conn.execute(select(tbl)).mappings().first()
    return dict(row) if row else {}

@st.cache_data(ttl=30)
def get_coaches_df() -> pd.DataFrame:
    tbl = reflect_table(T_COACH)
    if tbl is None: return pd.DataFrame()
    with engine.begin() as conn:
        rows = conn.execute(select(tbl)).mappings().all()
    return pd.DataFrame(rows)

@st.cache_data(ttl=30)
def get_slots_df() -> pd.DataFrame:
    tbl = reflect_table(T_SLOT)
    if tbl is None: return pd.DataFrame()
    with engine.begin() as conn:
        rows = conn.execute(select(tbl)).mappings().all()
    return pd.DataFrame(rows)

def compute_share_and_percent(student_row: Dict[str, Any], amount: float) -> Tuple[float, float]:
    """Retorna (repasse_em_reais, percentual_usado 0..1)."""
    try:
        # 100% para professor com flag
        if "coach_id" in student_row and student_row.get("coach_id") is not None:
            dfc = get_coaches_df()
            if not dfc.empty and "id" in dfc.columns:
                row = dfc[dfc["id"] == student_row["coach_id"]]
                if not row.empty and "full_pass" in row.columns and bool(row.iloc[0].get("full_pass", False)):
                    return float(amount), 1.0
        # Override por aluno (decimal 0..1)
        ov = student_row.get("master_percent_override")
        if ov is not None and not (isinstance(ov, float) and math.isnan(ov)):
            p = float(ov)
            return float(amount) * p, p
        # Padrão (settings)
        cfg = get_settings()
        if cfg.get("master_percent") is not None:
            p = float(cfg["master_percent"])
            return float(amount) * p, p
    except Exception:
        pass
    return 0.0, 0.0

def clear_data_cache():
    for fn in (
        fetch_students_df, fetch_grads_df, fetch_payments_df, fetch_extras_df,
        get_settings, get_coaches_df, get_slots_df
    ):
        try: fn.clear()
        except Exception: pass

# =========================================================
# CRUD GENÉRICO
# =========================================================
def insert_row(tbl_name: str, values: Dict[str, Any]) -> Optional[int]:
    tbl = reflect_table(tbl_name)
    if tbl is None: return None
    payload = {k: v for k, v in values.items() if has_col(tbl, k)}
    if not payload: return None
    stmt = insert(tbl).values(**payload)
    if has_col(tbl, "id"): stmt = stmt.returning(tbl.c.id)
    with engine.begin() as conn:
        res = conn.execute(stmt)
        new_id = int(res.scalar_one()) if has_col(tbl, "id") else 1
    clear_data_cache()
    return new_id

def update_row(tbl_name: str, row_id: int, values: Dict[str, Any]) -> int:
    tbl = reflect_table(tbl_name)
    if tbl is None or not has_col(tbl, "id"): return 0
    payload = {k: v for k, v in values.items() if has_col(tbl, k)}
    stmt = update(tbl).where(tbl.c.id == row_id).values(**payload)
    with engine.begin() as conn:
        res = conn.execute(stmt)
        n = res.rowcount or 0
    clear_data_cache()
    return n

def delete_rows(tbl_name: str, ids: List[int]) -> int:
    tbl = reflect_table(tbl_name)
    if tbl is None or not has_col(tbl, "id") or not ids: return 0
    stmt = delete(tbl).where(tbl.c.id.in_(ids))
    with engine.begin() as conn:
        res = conn.execute(stmt)
        n = res.rowcount or 0
    clear_data_cache()
    return n

# =========================================================
# FETCH DATAFRAMES
# =========================================================
@st.cache_data(ttl=30)
def fetch_students_df() -> pd.DataFrame:
    tbl = reflect_table(T_STUDENT)
    if tbl is None: return pd.DataFrame()
    with engine.begin() as conn:
        rows = conn.execute(select(tbl)).mappings().all()
    df = pd.DataFrame(rows)
    if df.empty: return df

    # derivados
    if "birth_date" in df.columns: df["Idade"] = df["birth_date"].apply(idade_atual)
    else: df["Idade"] = "—"
    if "start_date" in df.columns: df["Tempo de treino"] = df["start_date"].apply(tempo_treino_fmt)
    else: df["Tempo de treino"] = "—"

    # graduação atual
    gh = reflect_table(T_GRADUATION)
    if gh is not None and "id" in df.columns:
        # últimas por data
        date_cols = [c for c in ["date","grade_date","created_at"] if has_col(gh, c)]
        dcol = date_cols[0] if date_cols else None
        grade_col = "grade" if has_col(gh,"grade") else ("graduation" if has_col(gh,"graduation") else None)
        if dcol and grade_col:
            with engine.begin() as conn:
                grads = conn.execute(text(f"""
                    SELECT DISTINCT ON (student_id)
                           student_id, {grade_col} AS g, {dcol} AS d
                    FROM {T_GRADUATION}
                    ORDER BY student_id, {dcol} DESC, id DESC
                """)).mappings().all()
            gmap = {r["student_id"]: (r["g"], r["d"]) for r in grads}
            df["Graduação"] = df["id"].map(lambda i: gmap.get(i, ("Branca", None))[0])
            df["Data Graduação"] = df["id"].map(lambda i: fmt_date(gmap.get(i, (None, None))[1]))
        else:
            df["Graduação"] = "Branca"; df["Data Graduação"] = "—"
    else:
        df["Graduação"] = "Branca"; df["Data Graduação"] = "—"
    return df

@st.cache_data(ttl=30)
def fetch_grads_df(student_id: Optional[int]=None) -> pd.DataFrame:
    tbl = reflect_table(T_GRADUATION)
    if tbl is None: return pd.DataFrame()
    stmt = select(tbl)
    if student_id is not None and has_col(tbl, "student_id"):
        stmt = stmt.where(tbl.c.student_id == student_id)
    order_col = None
    for c in ["date","grade_date","created_at"]:
        if has_col(tbl, c): order_col = tbl.c[c]; break
    with engine.begin() as conn:
        rows = conn.execute(stmt.order_by(order_col if order_col is not None else list(tbl.c.values())[0])).mappings().all()
    return pd.DataFrame(rows)

@st.cache_data(ttl=30)
def fetch_payments_df(month: Optional[str]=None) -> pd.DataFrame:
    tbl = reflect_table(T_PAYMENT)
    if tbl is None: return pd.DataFrame()
    stmt = select(tbl)
    if month and has_col(tbl, "month_ref"):
        stmt = stmt.where(tbl.c.month_ref == month)
    with engine.begin() as conn:
        order_col = tbl.c.get("paid_date", list(tbl.c.values())[0])
        rows = conn.execute(stmt.order_by(order_col.desc())).mappings().all()
    return pd.DataFrame(rows)

@st.cache_data(ttl=30)
def fetch_extras_df(month: Optional[str]=None) -> pd.DataFrame:
    tbl = reflect_table(T_EXTRA)
    if tbl is None: return pd.DataFrame()
    stmt = select(tbl)
    if month and has_col(tbl, "month_ref"):
        stmt = stmt.where(or_(tbl.c.month_ref == month, tbl.c.get("is_recurring", text("0")) == True))
    with engine.begin() as conn:
        order_col = tbl.c.get("created_at", list(tbl.c.values())[0])
        rows = conn.execute(stmt.order_by(order_col.desc())).mappings().all()
    return pd.DataFrame(rows)

# =========================================================
# GRAD PAYLOAD FLEXÍVEL
# =========================================================
def build_grad_payload(student_id: int, grade: str, when: date, notes: Optional[str]) -> Dict[str, Any]:
    tbl = reflect_table(T_GRADUATION)
    if tbl is None: return {}
    payload: Dict[str, Any] = {}
    if has_col(tbl, "student_id"): payload["student_id"] = student_id
    # grade
    if has_col(tbl, "grade"): payload["grade"] = grade
    elif has_col(tbl, "graduation"): payload["graduation"] = grade
    # date
    if has_col(tbl, "date"): payload["date"] = when
    elif has_col(tbl, "grade_date"): payload["grade_date"] = when
    elif has_col(tbl, "created_at"): payload["created_at"] = when
    # notes
    if notes:
        for c in ["notes","obs","observations"]:
            if has_col(tbl, c): payload[c] = notes; break
    return payload

# =========================================================
# NAV
# =========================================================
ALL_PAGES = ["Alunos","Graduações","Receber Pagamento","Extras (Repasse)","Relatórios","Importar / Exportar","Configurações"]
PAGES = ["Alunos","Relatórios"] if st.session_state["role"] == "operador" else ALL_PAGES
st.sidebar.markdown("### Navegação")
page = st.sidebar.radio("Ir para:", PAGES, index=0, label_visibility="collapsed")

# =========================================================
# ALUNOS
# =========================================================
if page == "Alunos":
    df_students = fetch_students_df()
    dfc, dfs = get_coaches_df(), get_slots_df()

    st.subheader("Alunos cadastrados")
    if df_students.empty:
        st.info("Nenhum aluno.")
    else:
        dfx = df_students.copy()
        for c in ("birth_date","start_date"):
            if c in dfx.columns: dfx[c] = dfx[c].apply(fmt_date)
        show = [c for c in ["id","name","birth_date","start_date","monthly_fee","active","Graduação","Data Graduação","Idade","Tempo de treino"] if c in dfx.columns]
        st.dataframe(
            dfx[show].rename(columns={
                "id":"ID","name":"Nome","birth_date":"Nascimento","start_date":"Início","monthly_fee":"Mensalidade (R$)","active":"Ativo?"
            }),
            use_container_width=True, hide_index=True
        )

    st.divider()
    col1, col2 = st.columns([1,1])

    # opções dropdown
    coach_opts = [(None,"(Sem professor)")]
    if not dfc.empty and "id" in dfc.columns and "name" in dfc.columns:
        coach_opts += [(int(i), n) for i,n in zip(dfc["id"], dfc["name"])]
    slot_opts = [(None,"(Sem horário)")]
    if not dfs.empty and "id" in dfs.columns:
        label_col = next((c for c in ["name","label","title","slot","descricao","hora","time"] if c in dfs.columns), None)
        if label_col:
            slot_opts += [(int(i), str(l)) for i,l in zip(dfs["id"], dfs[label_col])]
        else:
            slot_opts += [(int(i), f"ID {int(i)}") for i in dfs["id"]]

    # CADASTRAR
    with col1:
        st.markdown("### ➕ Cadastrar novo aluno")
        with st.form("form_new_student", clear_on_submit=False):
            n_name  = st.text_input("Nome *")
            n_birth = st.date_input("Data de nascimento", value=date(2000,1,1), min_value=BIRTH_MIN, max_value=BIRTH_MAX, format="DD/MM/YYYY")
            n_start = st.date_input("Início do treino", value=date.today(), min_value=TRAIN_MIN, max_value=TRAIN_MAX, format="DD/MM/YYYY")
            n_fee   = st.number_input("Mensalidade (R$)", min_value=0.0, step=10.0, format="%.2f")
            n_active= st.checkbox("Ativo?", value=True)
            n_override_pct = st.number_input("Repasse do aluno (%) (deixe 0 para usar padrão)", min_value=0, max_value=100, value=0, step=5)
            n_coach = st.selectbox("Professor responsável", [o[0] for o in coach_opts], format_func=lambda v: dict(coach_opts)[v])
            n_slot  = st.selectbox("Horário de treino", [o[0] for o in slot_opts], format_func=lambda v: dict(slot_opts)[v])
            submit_new = st.form_submit_button("Salvar", type="primary", use_container_width=True)
        if submit_new:
            try:
                payload = {
                    "name": (n_name or "").strip(),
                    "birth_date": n_birth,
                    "start_date": n_start,
                    "monthly_fee": float(n_fee or 0.0),
                    "active": bool(n_active),
                }
                if n_coach is not None: payload["coach_id"] = int(n_coach)
                if n_slot  is not None: payload["train_slot_id"] = int(n_slot)
                if n_override_pct > 0: payload["master_percent_override"] = float(n_override_pct)/100.0

                new_id = insert_row(T_STUDENT, payload) or 0
                gp = build_grad_payload(new_id, "Branca", n_start, None)
                if gp: insert_row(T_GRADUATION, gp)

                flash("success", f"Aluno cadastrado (ID {new_id}).")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao cadastrar: {e}")

    # EDITAR
    with col2:
        st.markdown("### ✏️ Editar aluno")
        if df_students.empty:
            st.info("Cadastre primeiro.")
        else:
            ids = df_students["id"].tolist()
            sid = st.selectbox("Selecionar aluno (ID)", ids, format_func=lambda i: f"ID {i} — {df_students.loc[df_students['id']==i,'name'].values[0]}")
            if sid:
                row = df_students[df_students["id"]==sid].iloc[0]
                with st.form(f"form_edit_{sid}"):
                    c1, c2 = st.columns([2,1])
                    with c1:
                        e_name  = st.text_input("Nome *", value=str(row.get("name","")))
                        e_birth = st.date_input("Data de nascimento", value=parse_date(row.get("birth_date")) or date(2000,1,1), min_value=BIRTH_MIN, max_value=BIRTH_MAX, format="DD/MM/YYYY")
                        e_start = st.date_input("Início do treino", value=parse_date(row.get("start_date")) or date.today(), min_value=TRAIN_MIN, max_value=TRAIN_MAX, format="DD/MM/YYYY")
                        e_active= st.checkbox("Ativo?", value=bool(row.get("active",True)))
                        st.text_input("Graduação atual", value=str(row.get("Graduação","Branca")), disabled=True)
                    with c2:
                        e_fee = st.number_input("Mensalidade (R$)", value=float(row.get("monthly_fee",0.0) or 0.0), min_value=0.0, step=10.0, format="%.2f")

                        val_override = row.get("master_percent_override")
                        if val_override is None or (isinstance(val_override, float) and math.isnan(val_override)):
                            cur_override_pct = 0
                        else:
                            try: cur_override_pct = int(round(float(val_override) * 100.0))
                            except Exception: cur_override_pct = 0
                        e_override = st.number_input("Repasse do aluno (%) (0 = usar padrão)", min_value=0, max_value=100, value=cur_override_pct, step=5)

                        coach_ids = [o[0] for o in coach_opts]
                        slot_ids  = [o[0] for o in slot_opts]
                        def idx(lst, val): 
                            try: return lst.index(val)
                            except ValueError: return 0
                        e_coach = st.selectbox("Professor responsável", coach_ids, index=idx(coach_ids, row.get("coach_id")), format_func=lambda v: dict(coach_opts)[v])
                        e_slot  = st.selectbox("Horário de treino",   slot_ids,  index=idx(slot_ids,  row.get("train_slot_id")), format_func=lambda v: dict(slot_opts)[v])

                    b_save = st.form_submit_button("Atualizar", type="primary")
                    b_del  = st.form_submit_button("Excluir", type="secondary")

                if b_save:
                    try:
                        pay = {
                            "name": e_name.strip(),
                            "birth_date": e_birth,
                            "start_date": e_start,
                            "monthly_fee": float(e_fee or 0.0),
                            "active": bool(e_active),
                        }
                        if e_coach is not None: pay["coach_id"] = int(e_coach)
                        if e_slot  is not None: pay["train_slot_id"] = int(e_slot)
                        pay["master_percent_override"] = (float(e_override)/100.0) if e_override > 0 else None
                        n = update_row(T_STUDENT, int(sid), pay)
                        flash("success" if n else "info", "Aluno atualizado." if n else "Nada para atualizar.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao atualizar: {e}")

                if b_del:
                    try:
                        n = delete_rows(T_STUDENT, [int(sid)])
                        flash("success" if n else "warning", "Aluno excluído." if n else "Aluno não encontrado.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao excluir: {e}")

# =========================================================
# GRADUAÇÕES
# =========================================================
elif page == "Graduações":
    require_admin()
    st.subheader("Histórico de Graduações")
    df_students = fetch_students_df()
    if df_students.empty:
        st.info("Cadastre alunos primeiro.")
    else:
        sid = st.selectbox("Aluno", df_students["id"].tolist(), format_func=lambda i: df_students.loc[df_students["id"]==i,"name"].values[0])
        if sid:
            st.markdown("#### Histórico")
            gdf = fetch_grads_df(sid)
            if not gdf.empty:
                gdf2 = gdf.copy()
                dcol = "date" if "date" in gdf2.columns else ("grade_date" if "grade_date" in gdf2.columns else ("created_at" if "created_at" in gdf2.columns else None))
                gcol = "grade" if "grade" in gdf2.columns else ("graduation" if "graduation" in gdf2.columns else None)
                if dcol: gdf2[dcol] = gdf2[dcol].apply(fmt_date)
                show = [c for c in ["id", gcol, dcol, "notes","obs","observations"] if c and c in gdf2.columns]
                ren = {}
                if gcol: ren[gcol] = "Graduação"
                if dcol: ren[dcol] = "Data"
                if "notes" in show: ren["notes"]="Observações"
                if "obs" in show: ren["obs"]="Observações"
                if "observations" in show: ren["observations"]="Observações"
                if "id" in show: ren["id"]="ID"
                st.dataframe(gdf2[show].rename(columns=ren), use_container_width=True, hide_index=True)
            else:
                st.info("Sem lançamentos.")

            st.divider()
            st.markdown("#### Adicionar nova graduação")
            with st.form("form_add_grad"):
                gg = st.selectbox("Graduação", GRADE_CHOICES, index=0)
                gd = st.date_input("Data da graduação", value=date.today(), min_value=TRAIN_MIN, max_value=TRAIN_MAX, format="DD/MM/YYYY")
                gn = st.text_input("Observações (opcional)")
                ok = st.form_submit_button("Adicionar", type="primary")
            if ok:
                try:
                    payload = build_grad_payload(int(sid), gg, gd, (gn or None))
                    if not payload:
                        st.error("Tabela de graduações não possui colunas compatíveis.")
                    else:
                        nid = insert_row(T_GRADUATION, payload)
                        if nid:
                            flash("success", "Graduação registrada.")
                        else:
                            flash("warning", "Nada foi inserido (verifique as colunas da tabela).")
                        st.rerun()
                except Exception as e:
                    st.error(f"Erro ao inserir graduação: {e}")

            if not gdf.empty and "id" in gdf.columns:
                st.markdown("#### Excluir graduação")
                gid = st.selectbox("Selecione uma entrada", gdf["id"].tolist())
                if st.button("🗑️ Excluir graduação selecionada", type="secondary"):
                    try:
                        n = delete_rows(T_GRADUATION, [int(gid)])
                        flash("success" if n else "warning", "Graduação excluída." if n else "Não encontrada.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao excluir: {e}")

# =========================================================
# RECEBER PAGAMENTO
# =========================================================
elif page == "Receber Pagamento":
    require_admin()
    st.subheader("Receber Pagamentos")

    df_students = fetch_students_df()
    if df_students.empty:
        st.info("Cadastre alunos.")
    else:
        dfc = get_coaches_df()
        coach_filter = None
        if not dfc.empty and "id" in dfc.columns and "name" in dfc.columns and "coach_id" in df_students.columns:
            opts = [{"id":None,"name":"(Todos)"}] + dfc[["id","name"]].to_dict("records")
            lbl = {o["id"]: o["name"] for o in opts}
            coach_filter = st.selectbox("Professor", [o["id"] for o in opts], format_func=lambda v: lbl[v])
            dfl = df_students.copy()
            if coach_filter is not None:
                dfl = dfl[dfl["coach_id"]==coach_filter]
        else:
            dfl = df_students.copy()

        st.markdown("#### Seleção de alunos para receber")
        id_choices = dfl["id"].tolist() if "id" in dfl.columns else []
        pick = st.multiselect("Alunos", id_choices, format_func=lambda i: f"ID {i} — {dfl.loc[dfl['id']==i,'name'].values[0]}")

        with st.form("form_receive_pay"):
            mref  = st.text_input("Mês de referência (AAAA-MM) (opcional)", value=datetime.today().strftime("%Y-%m"))
            pdate = st.date_input("Data do pagamento", value=date.today(), min_value=TRAIN_MIN, max_value=date.today(), format="DD/MM/YYYY")
            method= st.selectbox("Forma", ["Dinheiro","PIX","Cartão","Transferência"])
            notes = st.text_input("Observações (opcional)")
            ok    = st.form_submit_button("Confirmar recebimento", type="primary", use_container_width=True)

        if ok:
            try:
                tbl_p = reflect_table(T_PAYMENT)
                if tbl_p is None:
                    st.error("Tabela de pagamentos não encontrada.")
                else:
                    okc = 0
                    for sid in pick:
                        row = dfl[dfl["id"]==sid].iloc[0].to_dict()
                        amount = float(row.get("monthly_fee",0.0) or 0.0)
                        master_amount, pct_used = compute_share_and_percent(row, amount)
                        payload = {
                            "student_id": int(sid),
                            "amount": amount,
                            "master_amount": master_amount,
                            "paid_date": pdate,
                            "method": method,
                            "notes": (notes or None)
                        }
                        if has_col(tbl_p, "month_ref") and mref: payload["month_ref"] = mref
                        if has_col(tbl_p, "master_percent_used"): payload["master_percent_used"] = pct_used
                        if has_col(tbl_p, "master_adjustment"):    payload["master_adjustment"] = 0.0  # evita NOT NULL
                        insert_row(T_PAYMENT, payload)
                        okc += 1
                    flash("success", f"{okc} pagamento(s) registrado(s).")
                    st.rerun()
            except Exception as e:
                st.error(f"Erro ao receber pagamento: {e}")

        st.divider()
        st.markdown("#### Pagamentos do mês")
        month_list = st.text_input("Filtrar mês (AAAA-MM)", value=datetime.today().strftime("%Y-%m"))
        dpp = fetch_payments_df(month_list)
        if not dpp.empty:
            m = dpp.merge(df_students[["id","name"]], left_on="student_id", right_on="id", how="left", suffixes=("","_stu"))
            m["name"] = m["name"].fillna("(Aluno removido)")
            if "paid_date" in m.columns: m["paid_date"] = m["paid_date"].apply(fmt_date)
            display_cols = [c for c in ["id","paid_date","name","amount","master_amount","method","notes"] if c in m.columns]
            out = m[display_cols].rename(columns={"id":"ID","name":"Aluno","paid_date":"Data","amount":"Valor (R$)","master_amount":"Repasse (R$)"})
            st.dataframe(out, use_container_width=True, hide_index=True)

            if "ID" in out.columns:
                del_ids = st.multiselect("Selecionar para excluir", out["ID"].tolist())
                c1, c2 = st.columns([1,1])
                with c1:
                    if st.button("🗑️ Excluir selecionados", type="secondary", use_container_width=True):
                        try:
                            n = delete_rows(T_PAYMENT, [int(i) for i in del_ids])
                            flash("success" if n else "info", f"{n} registro(s) removido(s)." if n else "Nada selecionado.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erro ao excluir: {e}")
                with c2:
                    if st.button("🧹 Excluir TODOS deste mês", type="secondary", use_container_width=True):
                        try:
                            tbl_p = reflect_table(T_PAYMENT)
                            if tbl_p is None:
                                st.error("Tabela não encontrada.")
                            else:
                                if has_col(tbl_p, "month_ref") and month_list:
                                    stmt = delete(tbl_p).where(tbl_p.c.month_ref == month_list)
                                else:
                                    if _dialect() == "postgresql":
                                        stmt = delete(tbl_p).where(text("to_char(paid_date, 'YYYY-MM') = :m")).params(m=month_list)
                                    else:  # sqlite
                                        stmt = delete(tbl_p).where(text("strftime('%Y-%m', paid_date) = :m")).params(m=month_list)
                                with engine.begin() as conn:
                                    res = conn.execute(stmt)
                                clear_data_cache()
                                flash("success", f"{res.rowcount or 0} registro(s) removido(s).")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Erro ao excluir em massa: {e}")
        else:
            st.info("Sem pagamentos para o mês.")

# =========================================================
# EXTRAS
# =========================================================
elif page == "Extras (Repasse)":
    require_admin()
    st.subheader("Lançamentos Extras (positivos/negativos)")

    df_students = fetch_students_df()
    student_opts = [(None,"(Sem aluno vinculado)")]
    if not df_students.empty:
        student_opts += [(int(i), f"ID {int(i)} — {n}") for i,n in zip(df_students["id"], df_students["name"])]

    with st.form("form_extra"):
        edate = st.date_input("Data (DD/MM/AAAA)", value=date.today(), min_value=TRAIN_MIN, max_value=date.today(), format="DD/MM/YYYY")
        mref  = st.text_input("Mês de referência (AAAA-MM)", value=datetime.today().strftime("%Y-%m"))
        desc  = st.text_input("Descrição")
        val   = st.number_input("Valor do extra (R$) — pode ser negativo", step=10.0, format="%.2f")
        sid   = st.selectbox("Vincular a um aluno (opcional)", [o[0] for o in student_opts], format_func=lambda v: dict(student_opts)[v])
        rec   = st.checkbox("Fixo mês a mês? (recorrente)", value=False)
        ok    = st.form_submit_button("Adicionar extra", type="primary", use_container_width=True)

    if ok:
        try:
            payload = {"description": desc, "amount": float(val or 0.0), "month_ref": mref, "created_at": edate}
            if sid is not None: payload["student_id"] = int(sid)
            tbl = reflect_table(T_EXTRA)
            if has_col(tbl,"is_recurring"): payload["is_recurring"] = bool(rec)
            nid = insert_row(T_EXTRA, payload)
            if nid: flash("success", "Extra adicionado!")
            else:   flash("warning", "Nada foi inserido (verifique as colunas).")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao adicionar extra: {e}")

    st.divider()
    st.markdown("#### Lista de extras por mês")
    month_list = st.text_input("Mês (AAAA-MM)", value=datetime.today().strftime("%Y-%m"))
    dfe = fetch_extras_df(month_list)
    if dfe.empty:
        st.info("Sem extras.")
    else:
        if "student_id" in dfe.columns and not df_students.empty:
            dfe = dfe.merge(df_students[["id","name"]], left_on="student_id", right_on="id", how="left")
            dfe["Aluno"] = dfe["name"].fillna("Outros")
        else:
            dfe["Aluno"] = "Outros"
        view = []
        if "id" in dfe.columns: view.append("id")
        if "created_at" in dfe.columns: dfe["created_at"] = dfe["created_at"].apply(fmt_date); view.append("created_at")
        view += [c for c in ["Aluno","description","amount","is_recurring"] if c in dfe.columns]
        out = dfe[view].rename(columns={"id":"ID","created_at":"Data","description":"Descrição","amount":"Valor (R$)","is_recurring":"Recorrente?"})
        st.dataframe(out, use_container_width=True, hide_index=True)

        if "ID" in out.columns:
            del_ids = st.multiselect("Selecionar extras para excluir", out["ID"].tolist())
            if st.button("🗑️ Excluir extras selecionados", type="secondary"):
                try:
                    n = delete_rows(T_EXTRA, [int(i) for i in del_ids])
                    flash("success" if n else "info", f"{n} removido(s)." if n else "Nada selecionado.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao excluir: {e}")

# =========================================================
# RELATÓRIOS
# =========================================================
elif page == "Relatórios":
    st.subheader("Relatório de repasse (mensalidades + extras)")
    month = st.text_input("Mês de referência (AAAA-MM)", value=datetime.today().strftime("%Y-%m"))

    df_students = fetch_students_df()
    dpp = fetch_payments_df(month)
    dfe = fetch_extras_df(month)

    dfc = get_coaches_df()
    coach_filter = None
    if not dfc.empty and "id" in dfc.columns and "name" in dfc.columns and "coach_id" in df_students.columns:
        opts = [{"id":None,"name":"(Todos)"}] + dfc[["id","name"]].to_dict("records")
        lbl = {o["id"]: o["name"] for o in opts}
        coach_filter = st.selectbox("Professor", [o["id"] for o in opts], format_func=lambda v: lbl[v])

    st.markdown("### 📒 Mensalidades (alunos)")
    if dpp.empty:
        st.info("Sem pagamentos no mês."); pag = pd.DataFrame()
    else:
        pag = dpp.merge(df_students, left_on="student_id", right_on="id", how="left", suffixes=("","_s"))
        if coach_filter is not None and "coach_id" in pag.columns: pag = pag[pag["coach_id"] == coach_filter]
        if not pag.empty and "paid_date" in pag.columns: pag["paid_date"] = pag["paid_date"].apply(fmt_date)
        cols = [c for c in ["id","paid_date","name","Idade","Tempo de treino","Graduação","amount","master_amount","method","notes"] if c in pag.columns]
        pag = pag[cols].rename(columns={"id":"ID","name":"Aluno","paid_date":"Data","amount":"Valor (R$)","master_amount":"Repasse (R$)"})
        st.dataframe(pag, use_container_width=True, hide_index=True)

    total_pag = float(pag["Valor (R$)"].sum() if "Valor (R$)" in pag.columns else 0.0)

    st.markdown("### ➕ Relatório de extras (detalhado)")
    if dfe.empty:
        st.info("Sem extras no mês (recorrentes também aparecem)."); ext = pd.DataFrame()
    else:
        if "student_id" in dfe.columns:
            ext = dfe.merge(df_students[["id","name"]], left_on="student_id", right_on="id", how="left")
            ext["Aluno"] = ext["name"].fillna("Outros")
        else:
            ext = dfe.copy(); ext["Aluno"] = "Outros"
        if "created_at" in ext.columns: ext["created_at"] = ext["created_at"].apply(fmt_date)
        cols = [c for c in ["id","created_at","Aluno","description","amount","is_recurring"] if c in ext.columns]
        ext = ext[cols].rename(columns={"id":"ID","created_at":"Data","description":"Descrição","amount":"Valor (R$)","is_recorrente?":"Recorrente?"})
        # correção do nome da coluna
        if "Recorrente?" not in ext.columns and "is_recurring" in cols:
            ext.rename(columns={"is_recurring":"Recorrente?"}, inplace=True)
        st.dataframe(ext, use_container_width=True, hide_index=True)

    total_ext = float(ext["Valor (R$)"].sum() if not ext.empty and "Valor (R$)" in ext.columns else 0.0)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Mensalidades (bruto)", money(total_pag))
    c2.metric("Extras", money(total_ext))
    c3.metric("Total geral", money(total_pag + total_ext))

    st.markdown("#### Exportar CSV")
    if not pag.empty:
        out_csv_pag = pag.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Exportar mensalidades", out_csv_pag, file_name=f"mensalidades_{month}.csv", mime="text/csv")
    if not ext.empty:
        out_csv_ext = ext.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Exportar extras", out_csv_ext, file_name=f"extras_{month}.csv", mime="text/csv")

# =========================================================
# IMPORT / EXPORT
# =========================================================
elif page == "Importar / Exportar":
    require_admin()
    st.subheader("Importar / Exportar")

    st.markdown("### Exportar")
    df_students = fetch_students_df()
    if not df_students.empty:
        out = df_students.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Alunos (CSV)", out, file_name="alunos.csv", mime="text/csv")

    dpp = fetch_payments_df()
    if not dpp.empty:
        out = dpp.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Pagamentos (CSV)", out, file_name="pagamentos.csv", mime="text/csv")

    dfe = fetch_extras_df()
    if not dfe.empty:
        out = dfe.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Extras (CSV)", out, file_name="extras.csv", mime="text/csv")

    st.divider()
    st.markdown("### Importar (alunos) — CSV com colunas compatíveis")
    up = st.file_uploader("Arquivo CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            if df.empty: st.warning("CSV vazio.")
            else:
                count = 0
                for _, r in df.iterrows():
                    payload = {k: r[k] for k in r.index}
                    if "birth_date" in payload: payload["birth_date"] = parse_date(payload["birth_date"])
                    if "start_date" in payload: payload["start_date"] = parse_date(payload["start_date"])
                    if "monthly_fee" in payload:
                        try: payload["monthly_fee"] = float(payload["monthly_fee"])
                        except: payload["monthly_fee"] = 0.0
                    nid = insert_row(T_STUDENT, payload); 
                    if nid: count += 1
                flash("success", f"{count} aluno(s) importado(s).")
                st.rerun()
        except Exception as e:
            st.error(f"Erro ao importar: {e}")

# =========================================================
# CONFIGURAÇÕES
# =========================================================
elif page == "Configurações":
    require_admin()
    st.subheader("Configurações")

    # Settings
    tbl = reflect_table(T_SETTINGS)
    if tbl is None:
        st.info("Tabela de configurações não encontrada.")
    else:
        with engine.begin() as conn: row = conn.execute(select(tbl)).mappings().first()
        st.markdown("#### Parâmetros gerais")
        with st.form("form_settings"):
            sid = row.get("id") if row else None
            master_current = float(row.get("master_percent", 0.0) or 0.0) if row else 0.0
            master_percent = st.number_input("Percentual padrão de repasse (0.00 = 0%, 0.50 = 50%, 1.00 = 100%)", value=master_current, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
            save = st.form_submit_button("Salvar", type="primary")
        if save:
            try:
                if sid: update_row(T_SETTINGS, int(sid), {"master_percent": float(master_percent)})
                else:   insert_row(T_SETTINGS, {"master_percent": float(master_percent)})
                flash("success", "Configurações salvas.")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao salvar configurações: {e}")

    st.divider()
    # Professores
    st.markdown("#### Professores")
    dfc = get_coaches_df()
    if dfc.empty:
        st.info("Tabela de professores não encontrada ou vazia.")
    else:
        show = [c for c in ["id","name","full_pass"] if c in dfc.columns]
        st.dataframe(dfc[show], use_container_width=True, hide_index=True)
        with st.form("form_coach_new"):
            nc = st.text_input("Nome do professor")
            fp = st.checkbox("Repasse completo (100%)?", value=False)
            ok = st.form_submit_button("Adicionar professor", type="primary")
        if ok:
            try:
                nid = insert_row(T_COACH, {"name": nc, "full_pass": bool(fp)})
                if nid: flash("success", "Professor adicionado.")
                else:   flash("warning", "Nada inserido (verifique colunas).")
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao adicionar professor: {e}")
