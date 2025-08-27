# pages/03_📊_KPIs.py
# JAT - KPIs e gráficos (página independente, sem alterar telas existentes)

from __future__ import annotations
import os
import math
import datetime as dt
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# -----------------------------
# Configuração básica da página
# -----------------------------
st.set_page_config(page_title="JAT - KPIs", page_icon="📊", layout="wide")

# Paleta sugerida (tema JAT)
JAT_RED = "#D32F2F"
JAT_ORANGE = "#F57C00"
JAT_YELLOW = "#FBC02D"
JAT_BLACK = "#000000"
JAT_GREY = "#F5F5F5"

# -----------------------------
# Conexão com o banco
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    # Tenta pegar do secrets, depois da env var; se nada, erro claro
    db_url = (
        st.secrets.get("DATABASE_URL", None)
        if hasattr(st, "secrets") else None
    ) or os.getenv("DATABASE_URL", None)

    if not db_url:
        st.error(
            "DATABASE_URL não configurado. Defina em `.streamlit/secrets.toml` "
            "ou na variável de ambiente `DATABASE_URL`."
        )
        st.stop()

    # Psycopg v3 (postgresql+psycopg) ou qualquer outro driver compatível
    connect_args = {}
    if db_url.startswith("postgresql"):
        connect_args["connect_timeout"] = 10

    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        connect_args=connect_args
    )
    return engine


# -----------------------------
# Utilitários
# -----------------------------
def brl(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "R$ 0,00"
    try:
        s = f"R$ {x:,.2f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0,00"


def year_month(d: pd.Series | pd.DatetimeIndex) -> pd.Series:
    return pd.to_datetime(d).dt.to_period("M").astype(str)  # "YYYY-MM"


def ensure_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_all(engine: Engine) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega tabelas mínimas necessárias para KPIs.
    Retorna: students, coaches, payments, extras, graduations, train_slots
    """
    with engine.connect() as con:
        # Tabelas principais (ajuste nomes se diferente no seu banco)
        students = pd.read_sql(text("""
            SELECT id, name, birth_date, start_date, active, monthly_fee, coach_id, train_slot_id
            FROM student
        """), con)

        coaches = pd.read_sql(text("""
            SELECT id, name, COALESCE(full_pass, FALSE) AS full_pass
            FROM coach
        """), con)

        payments = pd.read_sql(text("""
            SELECT id, student_id, paid_date, month_ref, amount, method, notes,
                   COALESCE(master_amount,0.0) AS master_amount,
                   COALESCE(master_percent_used, NULL) AS master_percent_used,
                   COALESCE(master_adjustment, 0.0) AS master_adjustment
            FROM payment
        """), con)

        extras = pd.read_sql(text("""
            SELECT id, date, month_ref, amount, description, COALESCE(is_recurring, FALSE) AS is_recurring,
                   COALESCE(student_id, NULL) AS student_id,
                   COALESCE(coach_id, NULL) AS coach_id
            FROM extra_repasse
        """), con)

        graduations = pd.read_sql(text("""
            SELECT id, student_id, grade, grade_date, notes
            FROM graduation
        """), con)

        train_slots = pd.read_sql(text("""
            SELECT id, name
            FROM train_slot
        """), con)

    # Normalizações de tipos e colunas úteis
    if not students.empty:
        students["birth_date"] = ensure_datetime(students["birth_date"]).dt.date
        students["start_date"] = ensure_datetime(students["start_date"]).dt.date
        students["active"] = students["active"].astype(bool)

    if not payments.empty:
        payments["paid_date"] = ensure_datetime(payments["paid_date"]).dt.date
        payments["amount"] = pd.to_numeric(payments["amount"], errors="coerce").fillna(0.0)
        payments["master_amount"] = pd.to_numeric(payments["master_amount"], errors="coerce").fillna(0.0)
        payments["month_ref"] = payments["month_ref"].fillna("").astype(str)

    if not extras.empty:
        extras["date"] = ensure_datetime(extras["date"]).dt.date
        extras["amount"] = pd.to_numeric(extras["amount"], errors="coerce").fillna(0.0)
        extras["is_recurring"] = extras["is_recurring"].astype(bool)
        extras["month_ref"] = extras["month_ref"].fillna("").astype(str)

    if not graduations.empty:
        graduations["grade_date"] = ensure_datetime(graduations["grade_date"]).dt.date

    return students, coaches, payments, extras, graduations, train_slots


def attach_latest_grade(students: pd.DataFrame, graduations: pd.DataFrame) -> pd.DataFrame:
    """Anexa graduação mais recente ao DF de alunos."""
    if graduations.empty or students.empty:
        students["latest_grade"] = "Branca"
        students["latest_grade_date"] = pd.NaT
        return students

    g = graduations.dropna(subset=["grade_date"]).copy()
    g["rank"] = g.groupby("student_id")["grade_date"].rank(method="first", ascending=False)
    g_latest = g[g["rank"] == 1][["student_id", "grade", "grade_date"]].rename(
        columns={"grade": "latest_grade", "grade_date": "latest_grade_date"}
    )
    out = students.merge(g_latest, left_on="id", right_on="student_id", how="left")
    out["latest_grade"] = out["latest_grade"].fillna("Branca")
    out["latest_grade_date"] = out["latest_grade_date"]
    out.drop(columns=["student_id"], inplace=True, errors="ignore")
    return out


def enrich_dims(students: pd.DataFrame, coaches: pd.DataFrame, train_slots: pd.DataFrame) -> pd.DataFrame:
    """Adiciona nomes de coach/slot e campos de idade/tempo de treino (hoje)."""
    df = students.copy()
    df = df.merge(coaches.rename(columns={"id": "coach_id", "name": "coach_name"}), on="coach_id", how="left")
    df = df.merge(train_slots.rename(columns={"id": "train_slot_id", "name": "train_slot_name"}), on="train_slot_id", how="left")

    # Idade hoje (anos) e tempo de treino (meses/anosemeses)
    today = dt.date.today()

    def idade_anos(nasc: Optional[dt.date]) -> Optional[int]:
        if pd.isna(nasc):
            return None
        anos = today.year - nasc.year - ((today.month, today.day) < (nasc.month, nasc.day))
        return max(0, anos)

    def tempo_meses(start: Optional[dt.date]) -> Optional[int]:
        if pd.isna(start):
            return None
        return (today.year - start.year) * 12 + (today.month - start.month) - (1 if today.day < start.day else 0)

    def anos_meses_str(meses: Optional[int]) -> str:
        if meses is None:
            return "—"
        if meses < 12:
            return f"{meses} meses"
        a, m = divmod(meses, 12)
        return f"{a} anos e {m} meses" if m else f"{a} anos"

    df["idade"] = df["birth_date"].apply(idade_anos)
    df["tempo_meses"] = df["start_date"].apply(tempo_meses)
    df["tempo_str"] = df["tempo_meses"].apply(anos_meses_str)
    return df


def apply_filters(
    students: pd.DataFrame,
    payments: pd.DataFrame,
    extras: pd.DataFrame,
    coach_id: Optional[int],
    date_start: dt.date,
    date_end: dt.date
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filtra pagamentos e extras por data e professor."""
    # Pagamentos: filtro por faixa de data e por professor (via aluno)
    p = payments.copy()
    if not p.empty:
        p = p[(p["paid_date"] >= date_start) & (p["paid_date"] <= date_end)]
        if coach_id is not None:
            stus = students.loc[students["coach_id"] == coach_id, "id"].tolist()
            p = p[p["student_id"].isin(stus)]

    # Extras: se tiver coach_id na própria linha, usa direto; caso contrário, filtra via aluno
    e = extras.copy()
    if not e.empty:
        e = e[(e["date"] >= date_start) & (e["date"] <= date_end)]
        if coach_id is not None:
            if "coach_id" in e.columns and e["coach_id"].notna().any():
                e = e[e["coach_id"] == coach_id]
            else:
                stus = students.loc[students["coach_id"] == coach_id, "id"].tolist()
                if "student_id" in e.columns:
                    e = e[e["student_id"].isin(stus)]

    return p, e


def monthly_projection(series: pd.Series) -> pd.Series:
    """
    Projeta valores ausentes usando média dos meses que têm valor.
    Entrada/saída: série indexada por YearMonth (string) com valores numéricos ou NaN.
    """
    s = series.copy()
    if s.dropna().empty:
        return s.fillna(0.0)  # sem histórico: projeta 0
    media = s.dropna().mean()
    return s.fillna(media)


# -----------------------------
# UI - Filtros
# -----------------------------
st.title("📊 KPIs — JAT (Gestão de Alunos)")

engine = get_engine()
students, coaches, payments, extras, graduations, train_slots = fetch_all(engine)
students = attach_latest_grade(students, graduations)
students = enrich_dims(students, coaches, train_slots)

# Range de datas padrão
min_date_candidates = []
if not payments.empty:
    min_date_candidates.append(pd.to_datetime(payments["paid_date"]).min())
if not extras.empty:
    min_date_candidates.append(pd.to_datetime(extras["date"]).min())
DEFAULT_MIN = (min(min_date_candidates) if min_date_candidates else pd.Timestamp(dt.date.today().replace(day=1))).date()
DEFAULT_MAX = dt.date.today()

with st.form("filtros_kpi"):
    c1, c2, c3 = st.columns([2,2,2])
    with c1:
        prof_opcoes = ["(Todos)"] + coaches["name"].sort_values().tolist()
        prof_sel = st.selectbox("Professor", prof_opcoes, index=0)
        coach_id = None
        if prof_sel != "(Todos)":
            coach_id = int(coaches.loc[coaches["name"] == prof_sel, "id"].iloc[0])

    with c2:
        date_start = st.date_input("De (data)", value=DEFAULT_MIN, format="DD/MM/YYYY")
    with c3:
        date_end = st.date_input("Até (data)", value=DEFAULT_MAX, format="DD/MM/YYYY")

    st.caption("Clique em **Aplicar filtros** para recalcular os KPIs.")
    submitted = st.form_submit_button("✅ Aplicar filtros")

if not submitted:
    st.stop()

# Aplica filtros aos fatos
p_fil, e_fil = apply_filters(students, payments, extras, coach_id, date_start, date_end)

# -----------------------------
# KPIs (cards)
# -----------------------------
receita = float(p_fil["amount"].sum()) if not p_fil.empty else 0.0
repasse = float(p_fil["master_amount"].sum()) if not p_fil.empty else 0.0
extras_liq = float(e_fil["amount"].sum()) if not e_fil.empty else 0.0
lucro = receita - repasse - extras_liq

k1, k2, k3, k4 = st.columns(4)
k1.metric("Receita (Real)", brl(receita))
k2.metric("Repasse (pagamentos)", brl(repasse))
k3.metric("Extras (líquido)", brl(extras_liq))
k4.metric("Lucro (Real)", brl(lucro))

st.divider()

# -----------------------------
# Séries mensais: Ativos, Receita vs Projetada, Lucro vs Projetado
# -----------------------------
# Monta calendário mensal no intervalo
cal = pd.period_range(pd.Period(date_start, freq="M"), pd.Period(date_end, freq="M"), freq="M").astype(str)

# 1) Alunos ativos (pagaram no mês)
ativos_mes = (
    p_fil.assign(YearMonth=year_month(pd.to_datetime(p_fil["paid_date"])))
    .groupby("YearMonth")["student_id"].nunique()
    if not p_fil.empty else pd.Series(dtype=float)
).reindex(cal).fillna(0).astype(int)

# 2) Receita real mensal
receita_mes = (
    p_fil.assign(YearMonth=year_month(pd.to_datetime(p_fil["paid_date"])))
    .groupby("YearMonth")["amount"].sum()
    if not p_fil.empty else pd.Series(dtype=float)
).reindex(cal).astype(float)

# 3) Lucro real mensal
lucro_mes = (
    p_fil.assign(YearMonth=year_month(pd.to_datetime(p_fil["paid_date"])))
       .groupby("YearMonth")
       .apply(lambda df: float(df["amount"].sum() - df["master_amount"].sum()))
    if not p_fil.empty else pd.Series(dtype=float)
).reindex(cal).astype(float)

# 4) Extras mensais
extras_mes = (
    e_fil.assign(YearMonth=year_month(pd.to_datetime(e_fil["date"])))
    .groupby("YearMonth")["amount"].sum()
    if not e_fil.empty else pd.Series(dtype=float)
).reindex(cal).astype(float)

# Receita projetada: preenche meses sem receita pela média dos meses com valor
receita_proj = monthly_projection(receita_mes)

# Lucro projetado: usa lucro real se houver, senão média dos meses com valor
lucro_proj = monthly_projection(lucro_mes)

# -----------------------------
# Gráficos
# -----------------------------
st.subheader("Alunos ativos (pagaram no mês)")

df_ativos = pd.DataFrame({"Mês": cal, "Ativos": ativos_mes.values})
fig1 = px.bar(
    df_ativos, x="Mês", y="Ativos",
    title=None, color_discrete_sequence=[JAT_RED]
)
fig1.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis_title=None, yaxis_title=None)
st.plotly_chart(fig1, use_container_width=True)

cA, cB = st.columns(2)

with cA:
    st.subheader("Receita Real × Projetada")
    df_rec = pd.DataFrame({
        "Mês": cal,
        "Receita (Real)": receita_mes.fillna(0).values,
        "Receita (Projetada)": receita_proj.fillna(0).values
    })
    fig2 = px.bar(df_rec, x="Mês", y="Receita (Real)", color_discrete_sequence=[JAT_BLACK])
    fig2.add_scatter(x=df_rec["Mês"], y=df_rec["Receita (Projetada)"], name="Projetada", mode="lines+markers",
                     line=dict(color=JAT_ORANGE))
    fig2.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis_title=None, yaxis_title=None, legend_title=None)
    fig2.update_yaxes(tickprefix="R$ ")
    st.plotly_chart(fig2, use_container_width=True)

with cB:
    st.subheader("Lucro Real × Projetado")
    df_luc = pd.DataFrame({
        "Mês": cal,
        "Lucro (Real)": lucro_mes.fillna(0).values,
        "Lucro (Projetado)": lucro_proj.fillna(0).values
    })
    fig3 = px.bar(df_luc, x="Mês", y="Lucro (Real)", color_discrete_sequence=[JAT_RED])
    fig3.add_scatter(x=df_luc["Mês"], y=df_luc["Lucro (Projetado)"], name="Projetado", mode="lines+markers",
                     line=dict(color=JAT_YELLOW))
    fig3.update_layout(margin=dict(l=10,r=10,t=10,b=10), xaxis_title=None, yaxis_title=None, legend_title=None)
    fig3.update_yaxes(tickprefix="R$ ")
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# -----------------------------
# Tabelas detalhadas + export
# -----------------------------
st.subheader("Detalhes (tabelas)")

t1, t2 = st.tabs(["💳 Pagamentos (mês a mês)", "🧾 Extras (mês a mês)"])

with t1:
    df_pag = pd.DataFrame({"Mês": cal, "Receita (R$)": receita_mes.fillna(0).values,
                           "Repasse (R$)": (
                               p_fil.assign(YearMonth=year_month(pd.to_datetime(p_fil["paid_date"])))
                                  .groupby("YearMonth")["master_amount"].sum()
                               if not p_fil.empty else pd.Series(dtype=float)
                           ).reindex(cal).fillna(0).values
                          })
    df_pag["Lucro (R$)"] = df_pag["Receita (R$)"] - df_pag["Repasse (R$)"]
    st.dataframe(df_pag, use_container_width=True)
    st.download_button(
        "⬇️ Exportar CSV de Pagamentos",
        df_pag.to_csv(index=False).encode("utf-8-sig"),
        file_name="kpi_pagamentos.csv",
        mime="text/csv"
    )

with t2:
    df_ext = pd.DataFrame({"Mês": cal, "Extras (R$)": extras_mes.fillna(0).values})
    st.dataframe(df_ext, use_container_width=True)
    st.download_button(
        "⬇️ Exportar CSV de Extras",
        df_ext.to_csv(index=False).encode("utf-8-sig"),
        file_name="kpi_extras.csv",
        mime="text/csv"
    )
