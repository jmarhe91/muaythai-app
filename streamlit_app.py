# JAT - Gestão de Alunos (único arquivo)
from __future__ import annotations
import os, math
import datetime as dt
from typing import Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

try:
    import plotly.express as px
except ModuleNotFoundError:
    st.error("Faltou instalar `plotly` (adicione no requirements.txt).")
    st.stop()

# --------- Aparência / Constantes ----------
st.set_page_config(page_title="JAT - Gestão de Alunos", page_icon="🥋", layout="wide")
JAT_RED    = "#D32F2F"
JAT_ORANGE = "#F57C00"
JAT_YELLOW = "#FBC02D"
JAT_BLACK  = "#000000"

def show_logo():
    if os.path.exists("logo.png"):
        st.image("logo.png", width=130)

# --------- Sessão / Login ----------
DEFAULT_SESSION = {"auth_ok": False, "role": None, "user": None}
for k, v in DEFAULT_SESSION.items(): st.session_state.setdefault(k, v)

def is_public_kpis() -> bool:
    return (os.getenv("PUBLIC_KPIS") == "1") or bool(getattr(st, "secrets", {}).get("PUBLIC_KPIS", False))

def do_login_ui():
    st.title("JAT - Gestão de Alunos")
    show_logo()
    st.subheader("Login")
    adm_u = st.secrets.get("ADMIN_USER"); adm_p = st.secrets.get("ADMIN_PASS")
    view_u = st.secrets.get("VIEW_USER"); view_p = st.secrets.get("VIEW_PASS")
    with st.form("login"):
        u = st.text_input("Usuário")
        p = st.text_input("Senha", type="password")
        ok = st.form_submit_button("Entrar ✅")
    if ok:
        role = None
        if adm_u and adm_p and u==adm_u and p==adm_p: role="admin"
        elif view_u and view_p and u==view_u and p==view_p: role="viewer"
        if role:
            st.session_state.update(auth_ok=True, role=role, user=u)
            st.success("Login OK.")
            st.rerun()
        else:
            st.error("Usuário/senha inválidos.")

def require_login():
    if not st.session_state.get("auth_ok") and not is_public_kpis():
        do_login_ui()
        st.stop()

# --------- Banco ----------
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    db_url = (st.secrets.get("DATABASE_URL") if hasattr(st, "secrets") else None) or os.getenv("DATABASE_URL")
    if not db_url:
        st.error("Configure DATABASE_URL nos secrets ou env.")
        st.stop()
    return create_engine(db_url, pool_pre_ping=True, pool_size=5, max_overflow=5, connect_args={"connect_timeout": 10})

def brl(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))): x=0.0
    s=f"R$ {x:,.2f}"; return s.replace(",", "X").replace(".", ",").replace("X",".")

def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.date

def ym(d: pd.Series) -> pd.Series:
    return pd.to_datetime(d, errors="coerce").dt.to_period("M").astype(str)

# --------- Carregamentos (cache) ----------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_all() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eng = get_engine()
    with eng.connect() as c:
        students = pd.read_sql(text("""SELECT id,name,birth_date,start_date,active,monthly_fee,coach_id,train_slot_id,
                                              master_percent_override
                                       FROM student"""), c)
        coaches  = pd.read_sql(text("""SELECT id,name,COALESCE(full_pass,false) AS full_pass FROM coach"""), c)
        slots    = pd.read_sql(text("""SELECT id,name FROM train_slot"""), c)
        pays     = pd.read_sql(text("""SELECT id,student_id,paid_date,month_ref,amount,method,notes,
                                             COALESCE(master_amount,0.0) master_amount,
                                             master_percent_used, COALESCE(master_adjustment,0.0) master_adjustment
                                      FROM payment"""), c)
        extras   = pd.read_sql(text("""SELECT id,date,month_ref,amount,description,
                                             COALESCE(is_recurring,false) is_recurring,
                                             student_id, coach_id
                                      FROM extra_repasse"""), c)
        grads    = pd.read_sql(text("""SELECT id,student_id,grade,grade_date,notes FROM graduation"""), c)
        settings = pd.read_sql(text("""SELECT id,COALESCE(master_percent,0.6) master_percent FROM settings LIMIT 1"""), c)
    # normalize
    for df, cols in [(students,["birth_date","start_date"]), (pays,["paid_date"]), (extras,["date"]), (grads,["grade_date"])]:
        for col in cols: df[col]=to_date(df[col])
    students["active"]=students["active"].astype(bool)
    extras["amount"]=pd.to_numeric(extras["amount"],errors="coerce").fillna(0.0)
    pays["amount"]=pd.to_numeric(pays["amount"],errors="coerce").fillna(0.0)
    pays["master_amount"]=pd.to_numeric(pays["master_amount"],errors="coerce").fillna(0.0)
    return students, coaches, slots, pays, extras, grads, settings

def latest_grade_map(grads: pd.DataFrame) -> pd.DataFrame:
    if grads.empty:
        return pd.DataFrame(columns=["student_id","latest_grade","latest_grade_date"])
    g=grads.dropna(subset=["grade_date"]).copy()
    g["r"]=g.groupby("student_id")["grade_date"].rank(method="first", ascending=False)
    g=g[g["r"]==1][["student_id","grade","grade_date"]].rename(columns={"grade":"latest_grade","grade_date":"latest_grade_date"})
    return g

# --------- Helpers de negócio ----------
def idade_atual(birth: Optional[dt.date]) -> Optional[int]:
    if pd.isna(birth): return None
    today=dt.date.today()
    anos=today.year-birth.year-((today.month,today.day)<(birth.month,birth.day))
    return max(0,anos)

def tempo_meses(start: Optional[dt.date]) -> Optional[int]:
    if pd.isna(start): return None
    t=dt.date.today(); m=(t.year-start.year)*12+(t.month-start.month)-(1 if t.day<start.day else 0); return m

def anos_meses_str(meses: Optional[int]) -> str:
    if meses is None: return "—"
    if meses<12: return f"{meses} meses"
    a,m=divmod(meses,12); return f"{a} anos e {m} meses" if m else f"{a} anos"

# --------- Mutations ----------
def exec_sql(sql:str, params:dict=None):
    eng=get_engine()
    with eng.begin() as c:
        c.execute(text(sql), params or {})

def add_student(name:str,birth:Optional[dt.date],start:Optional[dt.date],fee:float,active:bool,
                coach_id:Optional[int],slot_id:Optional[int], override_pct:Optional[float]):
    exec_sql("""
        INSERT INTO student(name,birth_date,start_date,monthly_fee,active,coach_id,train_slot_id,master_percent_override)
        VALUES (:n,:b,:s,:f,:a,:c,:t,:o)
    """, {"n":name,"b":birth,"s":start,"f":fee,"a":active,"c":coach_id,"t":slot_id,
          "o": (override_pct/100.0 if override_pct not in (None,"") else None)})

def update_student(id_,name,birth,start,fee,active,coach_id,slot_id, override_pct):
    exec_sql("""
      UPDATE student SET name=:n,birth_date=:b,start_date=:s,monthly_fee=:f,active=:a,
                         coach_id=:c,train_slot_id=:t, master_percent_override=:o
      WHERE id=:id
    """, {"id":id_,"n":name,"b":birth,"s":start,"f":fee,"a":active,"c":coach_id,"t":slot_id,
          "o": (override_pct/100.0 if override_pct not in (None,"") else None)})

def delete_student(id_):
    # remove dependências
    exec_sql("DELETE FROM payment WHERE student_id=:id", {"id":id_})
    exec_sql("DELETE FROM graduation WHERE student_id=:id", {"id":id_})
    exec_sql("DELETE FROM extra_repasse WHERE student_id=:id", {"id":id_})
    exec_sql("DELETE FROM student WHERE id=:id", {"id":id_})

def add_payment(student_id:int, paid_date:dt.date, month_ref:str, amount:float, method:str, notes:str,
                master_percent:float, master_adjustment:float=0.0):
    # evitar duplicidade
    eng=get_engine()
    with eng.connect() as c:
        n=pd.read_sql(text("""SELECT COUNT(*) n FROM payment WHERE student_id=:s AND month_ref=:m"""),
                      c, params={"s":student_id,"m":month_ref})["n"].iloc[0]
    if n>0: raise ValueError("Já existe pagamento deste aluno para este mês.")
    master_amount = round(amount * master_percent + master_adjustment, 2)
    exec_sql("""INSERT INTO payment(student_id,paid_date,month_ref,amount,method,notes,
                                   master_percent_used,master_adjustment,master_amount)
                VALUES(:s,:d,:m,:a,:me,:no,:mp,:adj,:ma)""",
             {"s":student_id,"d":paid_date,"m":month_ref,"a":amount,"me":method,"no":notes or None,
              "mp":master_percent,"adj":master_adjustment,"ma":master_amount})

def delete_payment(id_): exec_sql("DELETE FROM payment WHERE id=:id", {"id":id_})

def add_extra(date,month_ref,amount,desc,is_recurring,student_id,coach_id):
    exec_sql("""INSERT INTO extra_repasse(date,month_ref,amount,description,is_recurring,student_id,coach_id)
                VALUES(:d,:m,:a,:ds,:r,:s,:c)""",
             {"d":date,"m":month_ref,"a":amount,"ds":desc,"r":is_recurring,
              "s":student_id,"c":coach_id})

def delete_extra(id_): exec_sql("DELETE FROM extra_repasse WHERE id=:id", {"id":id_})

def add_grad(student_id:int, grade:str, grade_date:dt.date, notes:str):
    exec_sql("""INSERT INTO graduation(student_id,grade,grade_date,notes)
                VALUES(:s,:g,:d,:n)""", {"s":student_id,"g":grade,"d":grade_date,"n":notes})
    # atualiza grade atual no cadastro (opcional; a tela lista a atual automaticamente)

def delete_grad(id_): exec_sql("DELETE FROM graduation WHERE id=:id", {"id":id_})

def save_settings(master_percent:float):
    exec_sql("""INSERT INTO settings(id,master_percent) VALUES(1,:p)
                ON CONFLICT (id) DO UPDATE SET master_percent=EXCLUDED.master_percent""", {"p":master_percent})

def add_coach(name:str, full_pass:bool): exec_sql("INSERT INTO coach(name,full_pass) VALUES(:n,:f)",{"n":name,"f":full_pass})
def del_coach(id_:int): exec_sql("DELETE FROM coach WHERE id=:id", {"id":id_})
def add_slot(name:str): exec_sql("INSERT INTO train_slot(name) VALUES(:n)", {"n":name})
def del_slot(id_:int): exec_sql("DELETE FROM train_slot WHERE id=:id", {"id":id_})

# --------- Páginas ----------
def page_home():
    st.title("JAT - Gestão de Alunos")
    show_logo(); st.write("")
    c1,c2=st.columns([3,1])
    with c1:
        st.success(f"Bem-vindo, **{st.session_state.get('user','usuário')}**! Perfil: **{st.session_state.get('role')}**")
        st.write("Use o menu lateral para navegar. Todas as telas estão neste arquivo.")
    with c2:
        with st.container(border=True):
            st.subheader("Sessão")
            st.write(f"Usuário: **{st.session_state.get('user')}**")
            st.write(f"Perfil: **{st.session_state.get('role')}**")
            if st.button("Sair 🚪", use_container_width=True):
                for k,v in DEFAULT_SESSION.items(): st.session_state[k]=v
                st.rerun()

def page_alunos():
    st.header("👥 Alunos")
    students, coaches, slots, pays, extras, grads, settings = fetch_all()
    latest = latest_grade_map(grads)
    df = students.merge(latest, left_on="id", right_on="student_id", how="left")
    df["latest_grade"]=df["latest_grade"].fillna("Branca")
    df["Idade"]=df["birth_date"].apply(idade_atual)
    df["Tempo"]=df["start_date"].apply(tempo_meses).apply(anos_meses_str)
    df = df.merge(coaches.rename(columns={"id":"coach_id","name":"Professor"}), on="coach_id", how="left")
    df = df.merge(slots.rename(columns={"id":"train_slot_id","name":"Horário"}), on="train_slot_id", how="left")
    vis = df.rename(columns={"name":"Aluno"}).sort_values("Aluno")
    vis = vis[["Aluno","Idade","Tempo","latest_grade","Professor","Horário","monthly_fee","active"]]
    vis = vis.rename(columns={"latest_grade":"Graduação","monthly_fee":"Mensalidade (R$)","active":"Ativo"})
    st.dataframe(vis, use_container_width=True)

    st.subheader("Editar / Novo")
    names = ["(Novo)"] + df["name"].sort_values().tolist()
    pick = st.selectbox("Selecionar", names)
    is_new = (pick=="(Novo)")
    if not is_new:
        row = df[df["name"]==pick].iloc[0]
    else:
        row = pd.Series({"id":None,"name":"","birth_date":None,"start_date":dt.date.today(),
                         "monthly_fee":0.0,"active":True,"coach_id":None,"train_slot_id":None,
                         "master_percent_override":None})

    with st.form("frm_aluno"):
        c1,c2,c3 = st.columns(3)
        with c1:
            nome = st.text_input("Nome", value=row["name"])
            nasc = st.date_input("Nascimento", value=row["birth_date"] or dt.date(2000,1,1), format="DD/MM/YYYY")
            inicio= st.date_input("Início treino", value=row["start_date"] or dt.date.today(), format="DD/MM/YYYY")
        with c2:
            mensal = st.number_input("Mensalidade (R$)", min_value=0.0, step=10.0, value=float(row["monthly_fee"] or 0.0))
            ativo  = st.checkbox("Ativo", value=bool(row["active"]))
            over  = st.number_input("Repasse do aluno (%) (vazio = usar padrão)", min_value=0, max_value=100,
                                    value=int(round((row["master_percent_override"] or 0)*100)) if row["master_percent_override"] else 0)
            usar_override = st.checkbox("Usar override de % para este aluno?", value=bool(row["master_percent_override"]))
        with c3:
            coach = st.selectbox("Professor responsável",
                                 ["(selecione)"]+[c for c in coaches["name"].tolist()],
                                 index=0 if pd.isna(row["coach_id"]) else 1 + int(coaches.index[coaches["id"]==row["coach_id"]].tolist()[0]) )
            coach_id = None if coach=="(selecione)" else int(coaches.loc[coaches["name"]==coach,"id"].iloc[0])
            slot = st.selectbox("Horário de treino", ["(selecione)"]+slots["name"].tolist(),
                                index=0 if pd.isna(row["train_slot_id"]) else 1 + int(slots.index[slots["id"]==row["train_slot_id"]].tolist()[0]))
            slot_id = None if slot=="(selecione)" else int(slots.loc[slots["name"]==slot,"id"].iloc[0])
        bt1,bt2=st.columns([1,1])
        salvar = bt1.form_submit_button("💾 Salvar")
        excluir= bt2.form_submit_button("🗑️ Excluir", disabled=is_new)

    if salvar:
        try:
            if is_new:
                add_student(nome,nasc,inicio,mensal,ativo,coach_id,slot_id, over if usar_override else None)
                st.success("Aluno criado.")
            else:
                update_student(int(row["id"]),nome,nasc,inicio,mensal,ativo,coach_id,slot_id, over if usar_override else None)
                st.success("Aluno atualizado.")
            st.cache_data.clear(); st.rerun()
        except Exception as e:
            st.error(f"Erro ao salvar: {e}")

    if (not is_new) and excluir:
        try:
            delete_student(int(row["id"])); st.success("Aluno excluído.")
            st.cache_data.clear(); st.rerun()
        except Exception as e:
            st.error(f"Erro ao excluir: {e}")

def page_pagamentos():
    st.header("💵 Receber Pagamento")
    students, coaches, slots, pays, extras, grads, settings = fetch_all()
    latest = latest_grade_map(grads)
    s = students.merge(latest, left_on="id", right_on="student_id", how="left")
    s["latest_grade"]=s["latest_grade"].fillna("Branca")
    today=dt.date.today(); ym_now=str(pd.Period(today, freq="M"))
    with st.form("flt_pay"):
        c1,c2,c3=st.columns(3)
        mes = st.text_input("Mês de referência (AAAA-MM)", value=ym_now)
        prof = st.selectbox("Professor", ["(Todos)"]+coaches["name"].sort_values().tolist())
        coach_id = None if prof=="(Todos)" else int(coaches.loc[coaches["name"]==prof,"id"].iloc[0])
        ver = st.radio("Visualizar", ["Pendentes","Pagos"], horizontal=True, index=0)
        aplicar = st.form_submit_button("Aplicar filtros")
    if not aplicar: st.stop()

    def pagos_df():
        df=pays[pays["month_ref"]==mes].copy()
        df=df.merge(students[["id","name","coach_id","monthly_fee"]].rename(columns={"id":"student_id"}), on="student_id", how="left")
        if coach_id: df=df[df["coach_id"]==coach_id]
        df["Aluno"]=df["name"]; df["Data"]=pd.to_datetime(df["paid_date"]).dt.strftime("%d/%m/%Y")
        df["Valor (R$)"]=df["amount"].astype(float)
        df["Repasse (R$)"]=df["master_amount"].astype(float)
        return df[["id","Aluno","Data","Valor (R$)","Repasse (R$)","method","notes"]].rename(columns={"id":"ID","method":"Meio","notes":"Obs"})

    if ver=="Pendentes":
        # quem não pagou no mês
        paid_ids=set(pays.loc[pays["month_ref"]==mes,"student_id"].tolist())
        df=s[(s["active"]) & (~s["id"].isin(paid_ids))].copy()
        if coach_id: df=df[df["coach_id"]==coach_id]
        df=df.sort_values("name")
        st.subheader("Pendentes")
        st.caption("Selecione e confirme. Regra: **1 pagamento por aluno por mês**.")
        pick = st.multiselect("Alunos", df["name"].tolist())
        col1,col2,col3=st.columns(3)
        with col1: data = st.date_input("Data de pagamento", value=today, format="DD/MM/YYYY")
        with col2: meio = st.selectbox("Forma", ["Dinheiro","PIX","Cartão","Transferência"], index=1)
        with col3: obs  = st.text_input("Observações (opcional)")
        cfg_pct = float(settings["master_percent"].iloc[0] if not settings.empty else 0.6)
        if st.button("✅ Confirmar pagamentos selecionados"):
            try:
                for nome in pick:
                    r=df[df["name"]==nome].iloc[0]
                    pct = cfg_pct if pd.isna(r["master_percent_override"]) else float(r["master_percent_override"])
                    # full pass?
                    if r.get("coach_id") and not coaches.empty:
                        full = bool(coaches.loc[coaches["id"]==r["coach_id"],"full_pass"].values[0])
                        if full: pct=1.0
                    add_payment(int(r["id"]), data, mes, float(r["monthly_fee"] or 0.0), meio, obs, pct)
                st.success("Pagamentos registrados.")
                st.cache_data.clear(); st.rerun()
            except Exception as e:
                st.error(f"Erro: {e}")
    else:
        st.subheader("Pagos")
        df=pagos_df()
        st.dataframe(df.drop(columns=["ID"]), use_container_width=True)
        ids = df["ID"].tolist()
        if ids:
            chosen = st.selectbox("Excluir pagamento de:", df["Aluno"].tolist())
            if st.button("🗑️ Excluir pagamento selecionado"):
                try:
                    pid=int(df.loc[df["Aluno"]==chosen,"ID"].iloc[0])
                    delete_payment(pid); st.success("Excluído.")
                    st.cache_data.clear(); st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")

def page_extras():
    st.header("➕ Extras (repasse)")
    students, coaches, slots, pays, extras, grads, settings = fetch_all()
    today=dt.date.today(); ym_now=str(pd.Period(today,freq="M"))
    with st.form("frm_extra"):
        c1,c2,c3 = st.columns(3)
        dt0 = c1.date_input("Data", value=today, format="DD/MM/YYYY")
        mref= c2.text_input("Mês de referência (AAAA-MM)", value=ym_now)
        val = c3.number_input("Valor (R$) (pode ser negativo)", step=10.0, value=0.0)
        desc= st.text_input("Descrição")
        c4,c5,c6 = st.columns(3)
        vinc_stu = c4.selectbox("Vincular a aluno (opcional)", ["(Nenhum)"]+students["name"].sort_values().tolist())
        vinc_coa = c5.selectbox("Vincular a professor (opcional)", ["(Nenhum)"]+coaches["name"].sort_values().tolist())
        rec = c6.checkbox("Recorrente (fixo mês a mês)?", value=False)
        addok= st.form_submit_button("💾 Adicionar extra")
    if addok:
        try:
            sid=None if vinc_stu=="(Nenhum)" else int(students.loc[students["name"]==vinc_stu,"id"].iloc[0])
            cid=None if vinc_coa=="(Nenhum)" else int(coaches.loc[coaches["name"]==vinc_coa,"id"].iloc[0])
            add_extra(dt0, mref, float(val), desc, rec, sid, cid)
            st.success("Extra lançado.")
            st.cache_data.clear(); st.rerun()
        except Exception as e:
            st.error(f"Erro ao salvar: {e}")

    st.subheader("Lista por mês")
    with st.form("flt_ext"):
        c1,c2=st.columns(2)
        mes = c1.text_input("Mês (AAAA-MM)", value=ym_now)
        prof= c2.selectbox("Professor", ["(Todos)"]+coaches["name"].sort_values().tolist())
        cid = None if prof=="(Todos)" else int(coaches.loc[coaches["name"]==prof,"id"].iloc[0])
        ok = st.form_submit_button("Aplicar")
    dfe = extras[extras["month_ref"]==mes].copy()
    dfe=dfe.merge(students[["id","name"]].rename(columns={"id":"student_id","name":"Aluno"}), on="student_id", how="left")
    dfe=dfe.merge(coaches[["id","name"]].rename(columns={"id":"coach_id","name":"Professor"}), on="coach_id", how="left")
    if cid: dfe=dfe[(dfe["coach_id"]==cid) | (dfe["Professor"].isna())]  # mantém outros para visão geral
    if dfe.empty:
        st.info("Sem extras para o mês.")
    else:
        vis=dfe.copy()
        vis["Data"]=pd.to_datetime(vis["date"]).dt.strftime("%d/%m/%Y")
        vis["Valor (R$)"]=vis["amount"].astype(float)
        vis["Recorrente?"]=vis["is_recurring"].map({True:"Sim",False:"Não"})
        show=vis[["Data","month_ref","Aluno","Professor","description","Valor (R$)","Recorrente?"]].rename(
            columns={"month_ref":"Mês","description":"Descrição"})
        st.dataframe(show, use_container_width=True)
        # excluir
        nomes=[f"ID {i} — {a or 'Outros'} — {d}" for i,a,d in zip(dfe["id"],dfe["Aluno"],dfe["description"])]
        pick=st.selectbox("Excluir extra:", nomes) if len(nomes)>0 else None
        if pick and st.button("🗑️ Excluir selecionado"):
            try:
                eid=int(pick.split()[1])
                delete_extra(eid); st.success("Extra excluído.")
                st.cache_data.clear(); st.rerun()
            except Exception as e:
                st.error(f"Erro: {e}")

def page_graduacoes():
    st.header("🎖️ Graduações")
    students, coaches, slots, pays, extras, grads, settings = fetch_all()
    nomes = students["name"].sort_values().tolist()
    sel = st.selectbox("Aluno", nomes)
    stu = students[students["name"]==sel].iloc[0]
    h = grads[grads["student_id"]==stu["id"]].copy().sort_values("grade_date", ascending=False)
    if h.empty:
        st.info("Sem histórico.")
    else:
        show=h.copy()
        show["Data"]=pd.to_datetime(show["grade_date"]).dt.strftime("%d/%m/%Y")
        st.dataframe(show[["grade","Data","notes"]].rename(columns={"grade":"Graduação","notes":"Observações"}),
                     use_container_width=True)
        ids = {f"{g} — {d}":i for g,i,d in zip(h["grade"],h["id"],pd.to_datetime(h["grade_date"]).dt.strftime("%d/%m/%Y"))}
        lab = st.selectbox("Excluir graduação:", ["(nenhuma)"]+list(ids.keys()))
        if lab!="(nenhuma)" and st.button("🗑️ Excluir graduação"):
            try: delete_grad(ids[lab]); st.success("Excluída."); st.cache_data.clear(); st.rerun()
            except Exception as e: st.error(f"Erro: {e}")

    st.subheader("Adicionar graduação")
    GRADES=["Branca","Amarelo","Amarelo e Branca","Verde","Verde e Branca",
            "Azul","Azul e Branca","Marrom","Marrom e Branca","Vermelha","Vermelha e Branca","Preta"]
    with st.form("frm_grad"):
        g = st.selectbox("Graduação", GRADES)
        d = st.date_input("Data", value=dt.date.today(), format="DD/MM/YYYY")
        n = st.text_input("Observações (opcional)")
        ok = st.form_submit_button("💾 Salvar")
    if ok:
        try: add_grad(int(stu["id"]), g, d, n or None); st.success("Graduação adicionada."); st.cache_data.clear(); st.rerun()
        except Exception as e: st.error(f"Erro: {e}")

def page_relatorios():
    st.header("📑 Relatórios")
    students, coaches, slots, pays, extras, grads, settings = fetch_all()
    latest = latest_grade_map(grads)
    s = students.merge(latest, left_on="id", right_on="student_id", how="left")
    s["latest_grade"]=s["latest_grade"].fillna("Branca")
    with st.form("flt_rep"):
        c1,c2,c3=st.columns(3)
        ini=c1.date_input("De", value=dt.date.today().replace(day=1), format="DD/MM/YYYY")
        fim=c2.date_input("Até", value=dt.date.today(), format="DD/MM/YYYY")
        prof=c3.selectbox("Professor", ["(Todos)"]+coaches["name"].sort_values().tolist())
        cid=None if prof=="(Todos)" else int(coaches.loc[coaches["name"]==prof,"id"].iloc[0])
        ok=st.form_submit_button("Aplicar")
    if not ok: st.stop()

    p=pays.copy()
    p["paid_date"]=pd.to_datetime(p["paid_date"])
    p=p[(p["paid_date"]>=pd.Timestamp(ini))&(p["paid_date"]<=pd.Timestamp(fim))]
    p=p.merge(students[["id","name","coach_id"]].rename(columns={"id":"student_id"}), on="student_id", how="left")
    if cid: p=p[p["coach_id"]==cid]
    e=extras.copy()
    e["date"]=pd.to_datetime(e["date"])
    e=e[(e["date"]>=pd.Timestamp(ini))&(e["date"]<=pd.Timestamp(fim))]
    e=e.merge(students[["id","name","coach_id"]].rename(columns={"id":"student_id","name":"Aluno"}), on="student_id", how="left")
    e=e.merge(coaches[["id","name"]].rename(columns={"id":"coach_id","name":"Professor"}), on="coach_id", how="left")
    if cid: e=e[(e["coach_id"]==cid) | (e["Professor"].isna())]

    st.subheader("Mensalidades")
    if p.empty:
        st.info("Sem pagamentos no período.")
    else:
        vis=p.copy()
        vis["Aluno"]=vis["name"]
        vis["Data"]=vis["paid_date"].dt.strftime("%d/%m/%Y")
        vis["Valor (R$)"]=vis["amount"].astype(float)
        vis["Repasse (R$)"]=vis["master_amount"].astype(float)
        show=vis[["Aluno","Data","Valor (R$)","Repasse (R$)","method","notes"]].rename(columns={"method":"Meio","notes":"Obs"})
        st.dataframe(show,use_container_width=True)

    st.subheader("Extras (detalhado)")
    if e.empty:
        st.info("Sem extras no período.")
        total_extras=0.0
    else:
        e2=e.copy()
        e2["Data"]=e2["date"].dt.strftime("%d/%m/%Y")
        e2["Valor (R$)"]=e2["amount"].astype(float)
        st.dataframe(e2[["Data","Aluno","Professor","description","Valor (R$)","is_recurring"]].rename(
            columns={"description":"Descrição","is_recurring":"Recorrente?"}), use_container_width=True)
        total_extras=float(e2["Valor (R$)"].sum())

    total_receita=float(p["amount"].sum()) if not p.empty else 0.0
    total_repasse=float(p["master_amount"].sum()) if not p.empty else 0.0
    lucro = total_receita - total_repasse - total_extras

    st.divider()
    k1,k2,k3,k4=st.columns(4)
    k1.metric("Receita (R$)", brl(total_receita))
    k2.metric("Repasse (R$)", brl(total_repasse))
    k3.metric("Extras (R$)", brl(total_extras))
    k4.metric("Lucro (R$)", brl(lucro))

    # Export
    c1,c2=st.columns(2)
    with c1:
        if not p.empty:
            out=p.copy(); out["paid_date"]=pd.to_datetime(out["paid_date"]).dt.strftime("%d/%m/%Y")
            st.download_button("⬇️ Exportar pagamentos (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                               file_name="rel_pagamentos.csv", mime="text/csv")
    with c2:
        if not e.empty:
            out=e.copy(); out["date"]=pd.to_datetime(out["date"]).dt.strftime("%d/%m/%Y")
            st.download_button("⬇️ Exportar extras (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                               file_name="rel_extras.csv", mime="text/csv")

def page_config():
    st.header("⚙️ Configurações")
    students, coaches, slots, pays, extras, grads, settings = fetch_all()
    pct = float(settings["master_percent"].iloc[0] if not settings.empty else 0.6)
    with st.form("cfg"):
        pct_ui = st.number_input("Percentual padrão de repasse (%)", min_value=0, max_value=100, value=int(round(pct*100)))
        ok=st.form_submit_button("💾 Salvar")
    if ok:
        try: save_settings(pct_ui/100.0); st.success("Configurações salvas."); st.cache_data.clear()
        except Exception as e: st.error(f"Erro ao salvar: {e}")

    st.subheader("Professores")
    st.dataframe(coaches.rename(columns={"name":"Nome","full_pass":"Repasse 100%"}).drop(columns=["id"]),
                 use_container_width=True)
    with st.form("newc"):
        c1,c2=st.columns([3,1])
        nm=c1.text_input("Nome do professor")
        fp=c2.checkbox("Repasse 100%?")
        ok=st.form_submit_button("Adicionar")
    if ok and nm:
        try: add_coach(nm, fp); st.success("Professor adicionado."); st.cache_data.clear(); st.rerun()
        except Exception as e: st.error(f"Erro: {e}")
    if not coaches.empty:
        who=st.selectbox("Excluir professor:", ["(nenhum)"]+coaches["name"].tolist())
        if who!="(nenhum)" and st.button("🗑️ Excluir professor"):
            try: del_coach(int(coaches.loc[coaches["name"]==who,"id"].iloc[0])); st.success("Excluído."); st.cache_data.clear(); st.rerun()
            except Exception as e: st.error(f"Erro: {e}")

    st.subheader("Horários de treino")
    st.dataframe(slots.rename(columns={"name":"Nome"}).drop(columns=["id"]), use_container_width=True)
    with st.form("newslot"):
        nm=st.text_input("Novo horário (ex.: Ter/Qui 19h)")
        ok=st.form_submit_button("Adicionar")
    if ok and nm:
        try: add_slot(nm); st.success("Horário adicionado."); st.cache_data.clear(); st.rerun()
        except Exception as e: st.error(f"Erro: {e}")
    if not slots.empty:
        who=st.selectbox("Excluir horário:", ["(nenhum)"]+slots["name"].tolist())
        if who!="(nenhum)" and st.button("🗑️ Excluir horário"):
            try: del_slot(int(slots.loc[slots["name"]==who,"id"].iloc[0])); st.success("Excluído."); st.cache_data.clear(); st.rerun()
            except Exception as e: st.error(f"Erro: {e}")

def page_kpis():
    st.header("📊 KPIs")
    # Reutiliza funções do relatório, mas com projeções
    students, coaches, slots, pays, extras, grads, settings = fetch_all()
    with st.form("kpi_flt"):
        c1,c2,c3=st.columns(3)
        prof=c1.selectbox("Professor", ["(Todos)"]+coaches["name"].sort_values().tolist())
        cid=None if prof=="(Todos)" else int(coaches.loc[coaches["name"]==prof,"id"].iloc[0])
        ini=c2.date_input("De", value=dt.date.today().replace(day=1)-dt.timedelta(days=180), format="DD/MM/YYYY")
        fim=c3.date_input("Até", value=dt.date.today(), format="DD/MM/YYYY")
        ok=st.form_submit_button("✅ Aplicar filtros")
    if not ok: st.stop()

    p=pays.copy(); p["paid_date"]=pd.to_datetime(p["paid_date"])
    p=p[(p["paid_date"]>=pd.Timestamp(ini))&(p["paid_date"]<=pd.Timestamp(fim))]
    p=p.merge(students[["id","name","coach_id"]].rename(columns={"id":"student_id"}), on="student_id", how="left")
    if cid: p=p[p["coach_id"]==cid]
    e=extras.copy(); e["date"]=pd.to_datetime(e["date"])
    e=e[(e["date"]>=pd.Timestamp(ini))&(e["date"]<=pd.Timestamp(fim))]
    e=e.merge(students[["id","coach_id"]].rename(columns={"id":"student_id"}), on="student_id", how="left")
    if cid: e=e[(e["coach_id"]==cid) | (e["coach_id"].isna())]

    def monthly_projection(series: pd.Series) -> pd.Series:
        s=series.copy()
        if s.dropna().empty: return s.fillna(0.0)
        return s.fillna(s.dropna().mean())

    cal = pd.period_range(pd.Period(ini, "M"), pd.Period(fim, "M"), freq="M").astype(str)
    ativos = (p.assign(M=ym(p["paid_date"])) .groupby("M")["student_id"].nunique()
              if not p.empty else pd.Series(dtype=float)).reindex(cal).fillna(0).astype(int)
    receita = (p.assign(M=ym(p["paid_date"])) .groupby("M")["amount"].sum()
               if not p.empty else pd.Series(dtype=float)).reindex(cal).astype(float)
    repasse = (p.assign(M=ym(p["paid_date"])) .groupby("M")["master_amount"].sum()
               if not p.empty else pd.Series(dtype=float)).reindex(cal).astype(float)
    lucro = (receita.fillna(0)-repasse.fillna(0)).astype(float)
    extras_m = (e.assign(M=ym(e["date"])) .groupby("M")["amount"].sum()
                if not e.empty else pd.Series(dtype=float)).reindex(cal).astype(float)

    receita_proj = monthly_projection(receita)
    lucro_proj   = monthly_projection(lucro)

    k1,k2,k3,k4=st.columns(4)
    k1.metric("Receita total", brl(float(receita.sum())))
    k2.metric("Repasse", brl(float(repasse.sum())))
    k3.metric("Extras", brl(float(extras_m.sum())))
    k4.metric("Lucro", brl(float(lucro.sum()-extras_m.sum())))

    st.subheader("Alunos ativos (pagaram no mês)")
    fig=px.bar(pd.DataFrame({"Mês":cal,"Ativos":ativos.values}), x="Mês", y="Ativos",
               color_discrete_sequence=[JAT_RED])
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    cA,cB=st.columns(2)
    with cA:
        st.subheader("Receita Real × Projetada")
        d=pd.DataFrame({"Mês":cal,"Receita (Real)":receita.fillna(0).values,"Receita (Proj.)":receita_proj.fillna(0).values})
        f=px.bar(d, x="Mês", y="Receita (Real)", color_discrete_sequence=[JAT_BLACK])
        f.add_scatter(x=d["Mês"], y=d["Receita (Proj.)"], name="Projetada", mode="lines+markers",
                      line=dict(color=JAT_ORANGE))
        f.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis_title=None, yaxis_title=None, legend_title=None)
        f.update_yaxes(tickprefix="R$ ")
        st.plotly_chart(f, use_container_width=True)
    with cB:
        st.subheader("Lucro Real × Projetado")
        d=pd.DataFrame({"Mês":cal,"Lucro (Real)":lucro.fillna(0).values,"Lucro (Proj.)":lucro_proj.fillna(0).values})
        f=px.bar(d, x="Mês", y="Lucro (Real)", color_discrete_sequence=[JAT_RED])
        f.add_scatter(x=d["Mês"], y=d["Lucro (Proj.)"], name="Projetado", mode="lines+markers",
                      line=dict(color=JAT_YELLOW))
        f.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis_title=None, yaxis_title=None, legend_title=None)
        f.update_yaxes(tickprefix="R$ ")
        st.plotly_chart(f, use_container_width=True)

def sidebar_menu() -> str:
    st.sidebar.title("JAT")
    show_logo()
    st.sidebar.markdown("---")
    return st.sidebar.radio(
        "Navegação",
        ["🏠 Home","👥 Alunos","💵 Receber Pagamento","➕ Extras","🎖️ Graduações","📑 Relatórios","⚙️ Configurações","📊 KPIs","🚪 Sair"],
        index=0
    )

def main():
    require_login()
    page = sidebar_menu()
    if page=="🏠 Home": page_home()
    elif page=="👥 Alunos": page_alunos()
    elif page=="💵 Receber Pagamento": page_pagamentos()
    elif page=="➕ Extras": page_extras()
    elif page=="🎖️ Graduações": page_graduacoes()
    elif page=="📑 Relatórios": page_relatorios()
    elif page=="⚙️ Configurações": page_config()
    elif page=="📊 KPIs": page_kpis()
    elif page=="🚪 Sair":
        for k,v in DEFAULT_SESSION.items(): st.session_state[k]=v
        st.success("Sessão encerrada."); st.rerun()

if __name__ == "__main__":
    main()
