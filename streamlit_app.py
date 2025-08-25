# -*- coding: utf-8 -*-
import os
from datetime import date, datetime
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import (
    create_engine, MetaData, Table, select, insert, update, delete,
    text, and_, or_
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoSuchTableError, SQLAlchemyError

# ==============================================================
# CONFIG PÁGINA
# ==============================================================
st.set_page_config(page_title="JAT - Gestão de Alunos", page_icon="🥊", layout="wide")

# ==============================================================
# CONEXÃO / REFLEXÃO
# ==============================================================
DB_URL = (
    st.secrets.get("DATABASE_URL")
    or os.getenv("DATABASE_URL")
    or f"sqlite:///{os.path.join(os.path.dirname(__file__), 'muaythai.db')}"
)

@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    return create_engine(DB_URL, pool_pre_ping=True, future=True)

engine: Engine = get_engine()
metadata = MetaData()

# nomes padrão (ajuste aqui se seus nomes forem diferentes)
T_STUDENT     = "student"
T_GRADUATION  = "graduation_history"
T_PAYMENT     = "payment"
T_EXTRA       = "extra_repasse"
T_SETTINGS    = "settings"
T_COACH       = "coach"
T_SLOT        = "train_slot"

def reflect_table(name: str) -> Optional[Table]:
    try:
        return Table(name, metadata, autoload_with=engine)
    except NoSuchTableError:
        return None

def has_col(tbl: Optional[Table], col: str) -> bool:
    return (tbl is not None) and (col in tbl.c)

# ==============================================================
# HELPERS (datas, formatos, etc.)
# ==============================================================
BIRTH_MIN = date(1900, 1, 1)
BIRTH_MAX = date.today()
TRAIN_MIN = date(1990, 1, 1)
TRAIN_MAX = date.today()

GRADE_CHOICES = [
    "Branca","Amarelo","Amarelo e Branca","Verde","Verde e Branca",
    "Azul","Azul e Branca","Marrom","Marrom e Branca","Vermelha",
    "Vermelha e Branca","Preta"
]

def fmt_date(d: Optional[date]) -> str:
    if d in (None, "", "None"):
        return "—"
    if isinstance(d, str):
        try:
            d = datetime.fromisoformat(d).date()
        except Exception:
            return d
    return d.strftime("%d/%m/%Y")

def parse_date(v: Any) -> Optional[date]:
    if v in (None, "", "—"):
        return None
    if isinstance(v, date):
        return v
    try:
        if isinstance(v, str) and "/" in v:
            return datetime.strptime(v, "%d/%m/%Y").date()
        return datetime.fromisoformat(str(v)).date()
    except Exception:
        return None

def idade_atual(dn: Any) -> str:
    dn = parse_date(dn)
    if not dn:
        return "—"
    today = date.today()
    years = today.year - dn.year - ((today.month, today.day) < (dn.month, dn.day))
    return f"{years} anos"

def tempo_treino_fmt(dt_inicio: Any) -> str:
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
        f = 0.0
    s = f"{f:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

# ==============================================================
# LOGIN (admin/operador) — aceita [users] OU chaves no topo
# ==============================================================
def _do_login():
    st.sidebar.subheader("🔐 Login")
    u = st.sidebar.text_input("Usuário", placeholder="admin ou operador")
    p = st.sidebar.text_input("Senha", type="password")
    if st.sidebar.button("Entrar", type="primary", use_container_width=True):
        users_section = st.secrets.get("users", {}) or {
            "admin": st.secrets.get("admin", ""),
            "operador": st.secrets.get("operador", ""),
        }
        users = {
            "admin":    {"pw": users_section.get("admin", ""), "role": "admin"},
            "operador": {"pw": users_section.get("operador", ""), "role": "operador"},
        }
        u_in = (u or "").strip().lower()
        p_in = (p or "").strip()
        if u_in in users and p_in == users[u_in]["pw"] and p_in != "":
            st.session_state["user"] = u_in
            st.session_state["role"] = users[u_in]["role"]
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
# SETTINGS / COACH HELPERS (tudo opcional, só usa se existir)
# ==============================================================
def get_settings() -> Dict[str, Any]:
    tbl = reflect_table(T_SETTINGS)
    if tbl is None:
        return {}
    with engine.begin() as conn:
        row = conn.execute(select(tbl)).mappings().first()
    return dict(row) if row else {}

def get_coaches_df() -> pd.DataFrame:
    tbl = reflect_table(T_COACH)
    if tbl is None:
        return pd.DataFrame()
    with engine.begin() as conn:
        rows = conn.execute(select(tbl).order_by(tbl.c[list(tbl.c.keys())[1]])).mappings().all()
    return pd.DataFrame(rows)

def get_slots_df() -> pd.DataFrame:
    tbl = reflect_table(T_SLOT)
    if tbl is None:
        return pd.DataFrame()
    with engine.begin() as conn:
        rows = conn.execute(select(tbl).order_by(list(tbl.c.values())[1])).mappings().all()
    return pd.DataFrame(rows)

def compute_master_share(student_row: Dict[str, Any], amount: float) -> float:
    """
    Calcula repasse: se coach.full_pass == True -> 100%.
    Senão usa override do aluno (master_percent_override) se existir;
    senão configurações (settings.master_percent) se existir; default 0.0.
    """
    try:
        # 1) full_pass?
        if "coach_id" in student_row and student_row.get("coach_id") is not None:
            dfc = get_coaches_df()
            if not dfc.empty and "id" in dfc.columns:
                row = dfc[dfc["id"] == student_row["coach_id"]]
                if not row.empty:
                    if "full_pass" in row.columns and bool(row.iloc[0].get("full_pass", False)):
                        return float(amount)
        # 2) override aluno?
        if "master_percent_override" in student_row and student_row["master_percent_override"] is not None:
            return float(amount) * float(student_row["master_percent_override"])
        # 3) settings?
        cfg = get_settings()
        if "master_percent" in cfg and cfg["master_percent"] is not None:
            return float(amount) * float(cfg["master_percent"])
    except Exception:
        pass
    return 0.0

# ==============================================================
# CRUD genéricos
# ==============================================================
def insert_row(tbl_name: str, values: Dict[str, Any]) -> Optional[int]:
    tbl = reflect_table(tbl_name)
    if tbl is None:
        return None
    payload = {k: v for k, v in values.items() if has_col(tbl, k)}
    if not payload:
        return None
    stmt = insert(tbl).values(**payload)
    if has_col(tbl, "id"):
        stmt = stmt.returning(tbl.c.id)
    with engine.begin() as conn:
        try:
            res = conn.execute(stmt)
            if has_col(tbl, "id"):
                return int(res.scalar_one())
            return 1
        except SQLAlchemyError as e:
            raise e

def update_row(tbl_name: str, row_id: int, values: Dict[str, Any]) -> int:
    tbl = reflect_table(tbl_name)
    if tbl is None or not has_col(tbl, "id"):
        return 0
    payload = {k: v for k, v in values.items() if has_col(tbl, k)}
    stmt = update(tbl).where(tbl.c.id == row_id).values(**payload)
    with engine.begin() as conn:
        res = conn.execute(stmt)
        return res.rowcount or 0

def delete_rows(tbl_name: str, ids: List[int]) -> int:
    tbl = reflect_table(tbl_name)
    if tbl is None or not has_col(tbl, "id"):
        return 0
    if not ids:
        return 0
    stmt = delete(tbl).where(tbl.c.id.in_(ids))
    with engine.begin() as conn:
        res = conn.execute(stmt)
        return res.rowcount or 0

# ==============================================================
# DATAFRAMES de referência (alunos, graduações, pagamentos, extras)
# ==============================================================
def fetch_students_df() -> pd.DataFrame:
    tbl = reflect_table(T_STUDENT)
    if tbl is None:
        return pd.DataFrame()
    with engine.begin() as conn:
        rows = conn.execute(select(tbl)).mappings().all()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # derivados
    if "birth_date" in df.columns:
        df["Idade"] = df["birth_date"].apply(idade_atual)
    else:
        df["Idade"] = "—"
    if "start_date" in df.columns:
        df["Tempo de treino"] = df["start_date"].apply(tempo_treino_fmt)
    else:
        df["Tempo de treino"] = "—"

    # graduação atual
    gh = reflect_table(T_GRADUATION)
    if gh is not None and "id" in df.columns:
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

def fetch_grads_df(student_id: Optional[int]=None) -> pd.DataFrame:
    tbl = reflect_table(T_GRADUATION)
    if tbl is None:
        return pd.DataFrame()
    stmt = select(tbl)
    if student_id is not None and has_col(tbl, "student_id"):
        stmt = stmt.where(tbl.c.student_id == student_id)
    with engine.begin() as conn:
        rows = conn.execute(stmt.order_by(tbl.c.get("date", list(tbl.c.values())[0]))).mappings().all()
    return pd.DataFrame(rows)

def fetch_payments_df(month: Optional[str]=None) -> pd.DataFrame:
    tbl = reflect_table(T_PAYMENT)
    if tbl is None:
        return pd.DataFrame()
    stmt = select(tbl)
    if month and has_col(tbl, "month_ref"):
        stmt = stmt.where(tbl.c.month_ref == month)
    with engine.begin() as conn:
        rows = conn.execute(stmt.order_by(tbl.c.get("paid_date", list(tbl.c.values())[0]).desc())).mappings().all()
    return pd.DataFrame(rows)

def fetch_extras_df(month: Optional[str]=None) -> pd.DataFrame:
    tbl = reflect_table(T_EXTRA)
    if tbl is None:
        return pd.DataFrame()
    stmt = select(tbl)
    if month and has_col(tbl, "month_ref"):
        # inclui recorrentes + do mês
        stmt = stmt.where(or_(tbl.c.month_ref == month, tbl.c.get("is_recurring", text("0")) == True))
    with engine.begin() as conn:
        rows = conn.execute(stmt.order_by(tbl.c.get("created_at", list(tbl.c.values())[0]).desc())).mappings().all()
    return pd.DataFrame(rows)

# ==============================================================
# NAVEGAÇÃO
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
PAGES = ["Alunos","Relatórios"] if st.session_state["role"] == "operador" else ALL_PAGES
st.sidebar.markdown("### Navegação")
page = st.sidebar.radio("Ir para:", PAGES, index=0, label_visibility="collapsed")

st.title("🥊 Gestão da Turma de Muay Thai")

# ==============================================================
# ALUNOS
# ==============================================================
if page == "Alunos":
    df_students = fetch_students_df()

    st.subheader("Lista de alunos")
    if df_students.empty:
        st.info("Nenhum aluno.")
    else:
        dfx = df_students.copy()
        for c in ("birth_date","start_date"):
            if c in dfx.columns:
                dfx[c] = dfx[c].apply(fmt_date)
        show = [c for c in ["id","name","birth_date","start_date","monthly_fee","active","Idade","Tempo de treino","Graduação","Data Graduação"] if c in dfx.columns]
        st.dataframe(dfx[show].rename(columns={
            "id":"ID","name":"Nome","birth_date":"Nascimento","start_date":"Início","monthly_fee":"Mensalidade (R$)","active":"Ativo?"
        }), use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns([1,1])

    # CADASTRAR
    with col1:
        st.markdown("### ➕ Cadastrar novo aluno")
        with st.form("form_new_student", clear_on_submit=False):
            n_name  = st.text_input("Nome *")
            n_birth = st.date_input("Data de nascimento", value=date(2000,1,1), min_value=BIRTH_MIN, max_value=BIRTH_MAX, format="DD/MM/YYYY")
            n_start = st.date_input("Início do treino", value=date.today(), min_value=TRAIN_MIN, max_value=TRAIN_MAX, format="DD/MM/YYYY")
            n_fee   = st.number_input("Mensalidade (R$)", min_value=0.0, step=10.0, format="%.2f")
            n_active = st.checkbox("Ativo?", value=True)
            # opcionais
            n_coach = st.text_input("ID do Professor (opcional)", value="")
            n_slot  = st.text_input("ID do Horário (opcional)", value="")
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
                if n_coach.strip():
                    payload["coach_id"] = int(n_coach)
                if n_slot.strip():
                    payload["train_slot_id"] = int(n_slot)
                new_id = insert_row(T_STUDENT, payload) or 0
                # graduação branca automática com data do início
                insert_row(T_GRADUATION, {
                    "student_id": new_id,
                    "grade": "Branca",
                    "date": n_start
                })
                st.success(f"Aluno cadastrado (ID {new_id}).")
                st.rerun()
            except Exception as e:
                st.error(f"Erro: {e}")

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
                    with c2:
                        e_fee   = st.number_input("Mensalidade (R$)", value=float(row.get("monthly_fee",0.0) or 0.0), min_value=0.0, step=10.0, format="%.2f")
                        e_coach = st.text_input("ID do Professor (opcional)", value=str(row.get("coach_id","") or ""))
                        e_slot  = st.text_input("ID do Horário (opcional)", value=str(row.get("train_slot_id","") or ""))

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
                        if e_coach.strip(): pay["coach_id"] = int(e_coach)
                        if e_slot.strip():  pay["train_slot_id"] = int(e_slot)
                        n = update_row(T_STUDENT, int(sid), pay)
                        st.success("Atualizado." if n else "Nada para fazer.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro: {e}")

                if b_del:
                    require_admin()
                    try:
                        n = delete_rows(T_STUDENT, [int(sid)])
                        st.success("Excluído." if n else "Não encontrado.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro: {e}")

# ==============================================================
# GRADUAÇÕES
# ==============================================================
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
                if "date" in gdf2.columns:
                    gdf2["date"] = gdf2["date"].apply(fmt_date)
                show = [c for c in ["id","grade","date","notes"] if c in gdf2.columns]
                st.dataframe(gdf2[show].rename(columns={"grade":"Graduação","date":"Data","notes":"Observações","id":"ID"}), use_container_width=True, hide_index=True)
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
                    insert_row(T_GRADUATION, {"student_id": int(sid), "grade": gg, "date": gd, "notes": (gn or None)})
                    st.success("Graduação registrada.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")

            # EXCLUSÃO
            if not gdf.empty and "id" in gdf.columns:
                st.markdown("#### Excluir graduação")
                gid = st.selectbox("Selecione uma entrada", gdf["id"].tolist())
                if st.button("Excluir graduação selecionada", type="secondary"):
                    try:
                        delete_rows(T_GRADUATION, [int(gid)])
                        st.success("Excluída.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro: {e}")

# ==============================================================
# RECEBER PAGAMENTO
# ==============================================================
elif page == "Receber Pagamento":
    require_admin()
    st.subheader("Receber Pagamentos")

    df_students = fetch_students_df()
    if df_students.empty:
        st.info("Cadastre alunos.")
    else:
        # Filtro por professor (se existir)
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
        # multi-seleção
        if "id" in dfl.columns:
            id_choices = dfl["id"].tolist()
            pick = st.multiselect("Alunos", id_choices, format_func=lambda i: f"ID {i} — {dfl.loc[dfl['id']==i,'name'].values[0]}")
        else:
            pick = []

        with st.form("form_receive_pay"):
            mref = st.text_input("Mês de referência (AAAA-MM) (opcional)", value=datetime.today().strftime("%Y-%m"))
            pdate = st.date_input("Data do pagamento", value=date.today(), min_value=TRAIN_MIN, max_value=date.today(), format="DD/MM/YYYY")
            method = st.selectbox("Forma", ["Dinheiro","PIX","Cartão","Transferência"])
            notes  = st.text_input("Observações (opcional)")
            ok = st.form_submit_button("Confirmar recebimento", type="primary", use_container_width=True)

        if ok:
            try:
                tbl_p = reflect_table(T_PAYMENT)
                if tbl_p is None:
                    st.error("Tabela de pagamentos não encontrada.")
                else:
                    ok_count = 0
                    for sid in pick:
                        row = dfl[dfl["id"]==sid].iloc[0].to_dict()
                        amount = float(row.get("monthly_fee",0.0) or 0.0)
                        master = compute_master_share(row, amount)
                        payload = {
                            "student_id": int(sid),
                            "amount": amount,
                            "master_amount": master,
                            "paid_date": pdate,
                            "method": method,
                            "notes": (notes or None)
                        }
                        if has_col(tbl_p, "month_ref") and mref:
                            payload["month_ref"] = mref
                        insert_row(T_PAYMENT, payload)
                        ok_count += 1
                    st.success(f"{ok_count} pagamento(s) registrado(s).")
                    st.rerun()
            except Exception as e:
                st.error(f"Erro: {e}")

        st.divider()
        st.markdown("#### Pagamentos do mês")
        # lista do mês atual
        month_list = st.text_input("Filtrar mês (AAAA-MM)", value=datetime.today().strftime("%Y-%m"))
        dpp = fetch_payments_df(month_list)
        if not dpp.empty:
            # join com alunos
            m = dpp.merge(df_students[["id","name"]], left_on="student_id", right_on="id", how="left", suffixes=("","_stu"))
            m["name"] = m["name"].fillna("(Aluno removido)")
            view_cols = [c for c in ["id","paid_date","name","amount","master_amount","method","notes"] if c in m.columns]
            if "paid_date" in m.columns:
                m["paid_date"] = m["paid_date"].apply(fmt_date)
            m.rename(columns={"id":"ID","name":"Aluno","paid_date":"Data","amount":"Valor (R$)","master_amount":"Repasse (R$)"}, inplace=True)
            st.dataframe(m[view_cols], use_container_width=True, hide_index=True)

            # excluir selecionados
            if "ID" in m.columns:
                del_ids = st.multiselect("Selecionar para excluir", m["ID"].tolist())
                c1, c2 = st.columns([1,1])
                with c1:
                    if st.button("🗑️ Excluir selecionados", type="secondary", use_container_width=True):
                        try:
                            n = delete_rows(T_PAYMENT, [int(i) for i in del_ids])
                            st.success(f"{n} registro(s) removido(s).")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erro: {e}")
                with c2:
                    if st.button("🧹 Excluir TODOS deste mês", type="secondary", use_container_width=True):
                        try:
                            tbl = reflect_table(T_PAYMENT)
                            if tbl is None:
                                st.error("Tabela não encontrada.")
                            else:
                                if has_col(tbl, "month_ref") and month_list:
                                    stmt = delete(tbl).where(tbl.c.month_ref == month_list)
                                else:
                                    # apaga pelo mês da data
                                    stmt = delete(tbl).where(text("strftime('%Y-%m', paid_date) = :m")).params(m=month_list)
                                with engine.begin() as conn:
                                    res = conn.execute(stmt)
                                st.success(f"{res.rowcount or 0} registro(s) removido(s).")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Erro: {e}")
        else:
            st.info("Sem pagamentos para o mês.")

# ==============================================================
# EXTRAS (REPASSE)
# ==============================================================
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
            payload = {
                "description": desc,
                "amount": float(val or 0.0),
                "month_ref": mref,
                "created_at": edate
            }
            if sid is not None:
                payload["student_id"] = int(sid)
            # flag recorrente, se existir
            tbl = reflect_table(T_EXTRA)
            if has_col(tbl,"is_recurring"):
                payload["is_recurring"] = bool(rec)
            insert_row(T_EXTRA, payload)
            st.success("Extra adicionado!")
            st.rerun()
        except Exception as e:
            st.error(f"Erro: {e}")

    st.divider()
    st.markdown("#### Lista de extras por mês")
    month_list = st.text_input("Mês (AAAA-MM)", value=datetime.today().strftime("%Y-%m"))
    dfe = fetch_extras_df(month_list)
    if dfe.empty:
        st.info("Sem extras.")
    else:
        # join nome do aluno
        if "student_id" in dfe.columns and not df_students.empty:
            dfe = dfe.merge(df_students[["id","name"]], left_on="student_id", right_on="id", how="left")
            dfe["Aluno"] = dfe["name"].fillna("Outros")
        else:
            dfe["Aluno"] = "Outros"
        view = []
        if "id" in dfe.columns: view.append("id")
        if "created_at" in dfe.columns: dfe["created_at"] = dfe["created_at"].apply(fmt_date); view.append("created_at")
        view += [c for c in ["Aluno","description","amount","is_recurring"] if c in dfe.columns]
        st.dataframe(dfe[view].rename(columns={
            "id":"ID","created_at":"Data","description":"Descrição","amount":"Valor (R$)","is_recurring":"Recorrente?"
        }), use_container_width=True, hide_index=True)

        # exclusão
        if "ID" in dfe.rename(columns={"id":"ID"}).columns:
            del_ids = st.multiselect("Selecionar extras para excluir", dfe["id"].tolist())
            if st.button("🗑️ Excluir extras selecionados", type="secondary"):
                try:
                    n = delete_rows(T_EXTRA, [int(i) for i in del_ids])
                    st.success(f"{n} removido(s).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")

# ==============================================================
# RELATÓRIOS — mensalidades + extras, detalhados e totais
# ==============================================================
elif page == "Relatórios":
    st.subheader("Relatório de repasse (mensalidades + extras)")
    month = st.text_input("Mês de referência (AAAA-MM)", value=datetime.today().strftime("%Y-%m"))

    df_students = fetch_students_df()
    dpp = fetch_payments_df(month)
    dfe = fetch_extras_df(month)

    # Filtro por professor (se existir)
    dfc = get_coaches_df()
    coach_filter = None
    if not dfc.empty and "id" in dfc.columns and "name" in dfc.columns and "coach_id" in df_students.columns:
        opts = [{"id":None,"name":"(Todos)"}] + dfc[["id","name"]].to_dict("records")
        lbl = {o["id"]: o["name"] for o in opts}
        coach_filter = st.selectbox("Professor", [o["id"] for o in opts], format_func=lambda v: lbl[v])

    # ----- Mensalidades (detalhe por aluno)
    st.markdown("### 📒 Mensalidades (alunos)")
    if dpp.empty:
        st.info("Sem pagamentos no mês.")
        pag = pd.DataFrame()
    else:
        pag = dpp.merge(df_students, left_on="student_id", right_on="id", how="left", suffixes=("","_s"))
        if coach_filter is not None and "coach_id" in pag.columns:
            pag = pag[pag["coach_id"] == coach_filter]
        if not pag.empty and "paid_date" in pag.columns:
            pag["paid_date"] = pag["paid_date"].apply(fmt_date)
        cols = [c for c in ["id","paid_date","name","Idade","Tempo de treino","Graduação","amount","master_amount","method","notes"] if c in pag.columns]
        pag = pag[cols].rename(columns={
            "id":"ID","name":"Aluno","paid_date":"Data","amount":"Valor (R$)","master_amount":"Repasse (R$)"
        })
        st.dataframe(pag, use_container_width=True, hide_index=True)

    total_pag = float(pag["Valor (R$)"].sum() if "Valor (R$)" in pag.columns else 0.0)
    total_rep_pag = float(pag["Repasse (R$)"].sum() if "Repasse (R$)" in pag.columns else 0.0)

    # ----- Extras detalhados (linha a linha)
    st.markdown("### ➕ Relatório de extras (detalhado)")
    if dfe.empty:
        st.info("Sem extras no mês (lembrando que recorrentes também aparecem).")
        ext = pd.DataFrame()
    else:
        if "student_id" in dfe.columns:
            ext = dfe.merge(df_students[["id","name"]], left_on="student_id", right_on="id", how="left")
            ext["Aluno"] = ext["name"].fillna("Outros")
        else:
            ext = dfe.copy()
            ext["Aluno"] = "Outros"
        if "created_at" in ext.columns:
            ext["created_at"] = ext["created_at"].apply(fmt_date)
        cols = [c for c in ["id","created_at","Aluno","description","amount","is_recurring"] if c in ext.columns]
        ext = ext[cols].rename(columns={
            "id":"ID","created_at":"Data","description":"Descrição","amount":"Valor (R$)","is_recurring":"Recorrente?"
        })
        if coach_filter is not None:
            # se quiser vincular extras por professor, normalmente vem via aluno; sem isso, mantemos todos
            pass
        st.dataframe(ext, use_container_width=True, hide_index=True)

    total_ext = float(ext["Valor (R$)"].sum() if "Valor (R$)" in ext.columns else 0.0)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Mensalidades (bruto)", money(total_pag))
    c2.metric("Extras", money(total_ext))
    c3.metric("Total geral", money(total_pag + total_ext))

    # export
    st.markdown("#### Exportar CSV")
    if not pag.empty:
        out_csv_pag = pag.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Exportar mensalidades", out_csv_pag, file_name=f"mensalidades_{month}.csv", mime="text/csv")
    if not ext.empty:
        out_csv_ext = ext.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Exportar extras", out_csv_ext, file_name=f"extras_{month}.csv", mime="text/csv")

# ==============================================================
# IMPORTAR / EXPORTAR
# ==============================================================
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
            if df.empty:
                st.warning("CSV vazio.")
            else:
                count = 0
                for _, r in df.iterrows():
                    payload = {k: r[k] for k in r.index}
                    if "birth_date" in payload: payload["birth_date"] = parse_date(payload["birth_date"])
                    if "start_date" in payload: payload["start_date"] = parse_date(payload["start_date"])
                    if "monthly_fee" in payload:
                        try: payload["monthly_fee"] = float(payload["monthly_fee"])
                        except: payload["monthly_fee"] = 0.0
                    nid = insert_row(T_STUDENT, payload)
                    if nid: count += 1
                st.success(f"{count} aluno(s) importado(s).")
        except Exception as e:
            st.error(f"Erro: {e}")

# ==============================================================
# CONFIGURAÇÕES
# ==============================================================
elif page == "Configurações":
    require_admin()
    st.subheader("Configurações")

    # Settings
    tbl = reflect_table(T_SETTINGS)
    if tbl is None:
        st.info("Tabela de configurações não encontrada.")
    else:
        with engine.begin() as conn:
            row = conn.execute(select(tbl)).mappings().first()
        st.markdown("#### Parâmetros gerais")
        with st.form("form_settings"):
            if row:
                sid = row.get("id")
            else:
                sid = None
            master_current = float(row.get("master_percent", 0.0) or 0.0) if row else 0.0
            master_percent = st.number_input("Percentual padrão de repasse (0.00 = 0%, 0.50 = 50%, 1.00 = 100%)", value=master_current, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
            save = st.form_submit_button("Salvar", type="primary")
        if save:
            try:
                if sid:
                    update_row(T_SETTINGS, int(sid), {"master_percent": float(master_percent)})
                else:
                    insert_row(T_SETTINGS, {"master_percent": float(master_percent)})
                st.success("Configurações salvas.")
                st.rerun()
            except Exception as e:
                st.error(f"Erro: {e}")

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
                insert_row(T_COACH, {"name": nc, "full_pass": bool(fp)})
                st.success("Professor adicionado.")
                st.rerun()
            except Exception as e:
                st.error(f"Erro: {e}")
