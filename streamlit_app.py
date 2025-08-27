if cid: q = q.where(student.c.coach_id==cid)
p_rows = conn.execute(q.order_by(student.c.name)).mappings().all()

        # Extras (filtro só por extra_repasse.coach_id — já backfilled)
        # Extras (filtro apenas por extra_repasse.coach_id — já backfilled)
q_extra = (
select(
extra_repasse.c.id,
@@ -791,7 +791,7 @@ def page_relatorios(role: str):
.select_from(
extra_repasse
.outerjoin(student, extra_repasse.c.student_id == student.c.id)
                .outerjoin(coach, extra_repasse.c.coach_id == coach.c.id)
                .outerjoin(coach,   extra_repasse.c.coach_id   == coach.c.id)
)
.where(extra_repasse.c.month_ref == month)
.order_by(extra_repasse.c.date)
@@ -808,20 +808,20 @@ def page_relatorios(role: str):
"id","date","month_ref","description","amount","is_recurring","student_id","coach_id","aluno","professor_name"
])

    # Mensalidades
    # -------- Mensalidades
st.subheader("Mensalidades (alunos)")
if pag.empty:
st.info("Sem pagamentos no mês."); total_rep = 0.0
else:
        def idade(dt: Optional[date]) -> str:
        def idade(dt):
if not dt or pd.isna(dt): return "-"
if isinstance(dt, str):
try: dt = pd.to_datetime(dt).date()
except Exception: return "-"
today = TODAY
y = today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day))
return f"{y} anos"
        def tempo_treino(sd: Optional[date]) -> str:
        def tempo_treino(sd):
if not sd or pd.isna(sd): return "-"
if isinstance(sd, str):
try: sd = pd.to_datetime(sd).date()
@@ -835,31 +835,42 @@ def tempo_treino(sd: Optional[date]) -> str:
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
    # -------- Extras (detalhado)  <<< AQUI ESTAVA O PROBLEMA: usar LISTA em vez de SET
st.subheader("Extras (detalhado)")
if ext.empty:
st.info("Sem extras no mês."); total_ext = 0.0
else:
ext["Data"] = pd.to_datetime(ext["date"]).dt.strftime("%d/%m/%Y")
ext["Valor (R$)"] = ext["amount"].apply(fmt_money)
        st.dataframe(ext[{"Data","month_ref","description","Valor (R$)","is_recurring","aluno","professor_name"}] \
                     .rename(columns={"month_ref":"Ref.","description":"Descrição","is_recurring":"Recorrente?","aluno":"Aluno","professor_name":"Professor"}),
                     use_container_width=True, hide_index=True)

        cols = ["Data","month_ref","description","Valor (R$)","is_recurring","aluno","professor_name"]
        tabela = ext[cols].rename(columns={
            "month_ref":"Ref.",
            "description":"Descrição",
            "is_recurring":"Recorrente?",
            "aluno":"Aluno",
            "professor_name":"Professor"
        })
        st.dataframe(tabela, use_container_width=True, hide_index=True)

total_ext = float(ext["amount"].astype(float).sum())
st.metric("Total (extras)", fmt_money(total_ext))

    # Total geral
    # -------- Total geral
st.subheader("Total geral (repasse + extras)")
total_geral = (locals().get("total_rep",0.0) or 0.0) + (locals().get("total_ext",0.0) or 0.0)
st.metric("Total geral (a repassar)", fmt_money(total_geral))


def page_config(role: str):
st.header("Configurações")
if role != "admin":
