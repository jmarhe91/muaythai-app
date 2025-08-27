# streamlit_app.py
# JAT - Gestão de Alunos (Home + Login)
from __future__ import annotations
import os
import streamlit as st

st.set_page_config(page_title="JAT - Gestão de Alunos", page_icon="🏷️", layout="wide")

# ---------------------------
# Estado inicial (login)
# ---------------------------
DEFAULT_SESSION = {
    "auth_ok": False,     # foi autenticado?
    "role": None,         # "admin" ou "viewer"
    "user": None,         # login do usuário
}
for k, v in DEFAULT_SESSION.items():
    st.session_state.setdefault(k, v)

# ---------------------------
# Helper: logo (se existir)
# ---------------------------
def show_logo():
    # se tiver logo.png na raiz, mostra
    if os.path.exists("logo.png"):
        st.image("logo.png", width=160)

# ---------------------------
# Login
# ---------------------------
def do_login_ui():
    st.title("JAT - Gestão de Alunos")
    show_logo()
    st.subheader("Login")

    # Lê credenciais do secrets
    adm_user = st.secrets.get("ADMIN_USER")
    adm_pass = st.secrets.get("ADMIN_PASS")
    view_user = st.secrets.get("VIEW_USER")
    view_pass = st.secrets.get("VIEW_PASS")

    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Usuário", placeholder="admin ou viewer")
        p = st.text_input("Senha", type="password", placeholder="••••••••")
        entrar = st.form_submit_button("Entrar ✅")
        if entrar:
            role = None
            if adm_user and adm_pass and u == adm_user and p == adm_pass:
                role = "admin"
            elif view_user and view_pass and u == view_user and p == view_pass:
                role = "viewer"

            if role:
                st.session_state["auth_ok"] = True
                st.session_state["role"] = role
                st.session_state["user"] = u
                st.success("Login OK! Você já pode acessar as páginas.")
                # Link direto para KPIs
                st.page_link("pages/03_📊_KPIs.py", label="Ir para KPIs 📊", icon="📊")
                st.experimental_rerun()
            else:
                st.error("Usuário ou senha inválidos.")

    st.caption("Dica: defina ADMIN_USER/ADMIN_PASS e VIEW_USER/VIEW_PASS em `secrets`.")

# ---------------------------
# Home (pós-login)
# ---------------------------
def home_ui():
    st.title("JAT - Gestão de Alunos")
    show_logo()

    c1, c2 = st.columns([3,1])
    with c1:
        st.success(f"Bem-vindo, **{st.session_state.get('user', 'usuário')}**! Perfil: **{st.session_state.get('role')}**")
        st.write("Use o menu lateral para navegar pelas páginas do sistema.")
        # Atalhos úteis (os nomes devem corresponder aos seus arquivos em pages/)
        st.page_link("pages/03_📊_KPIs.py", label="Abrir KPIs 📊", icon="📊")
        # Ex.: st.page_link("pages/01_Alunos.py", label="Alunos", icon="👥")
        # Ex.: st.page_link("pages/02_Recebimentos.py", label="Receber Pagamentos", icon="💳")

    with c2:
        with st.container(border=True):
            st.subheader("Sessão")
            st.write(f"Usuário: **{st.session_state.get('user')}**")
            st.write(f"Perfil: **{st.session_state.get('role')}**")
            if st.button("Sair 🚪", use_container_width=True):
                for k in list(DEFAULT_SESSION.keys()):
                    st.session_state[k] = DEFAULT_SESSION[k]
                st.experimental_rerun()

    st.info("⚠️ Este arquivo não altera suas outras páginas. Ele apenas cuida do login e da página inicial.")

# ---------------------------
# Main
# ---------------------------
def main():
    if not st.session_state.get("auth_ok"):
        do_login_ui()
    else:
        home_ui()

if __name__ == "__main__":
    main()
