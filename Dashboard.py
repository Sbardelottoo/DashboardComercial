import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
import textwrap
from openai import OpenAI

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================== CONFIG ==================
st.set_page_config(layout='wide', page_title="Dashboard de Vendas")

# Meses (pt-BR)
MAP_PT = {1:"janeiro",2:"fevereiro",3:"março",4:"abril",5:"maio",6:"junho",
          7:"julho",8:"agosto",9:"setembro",10:"outubro",11:"novembro",12:"dezembro"}
MESES_PT = ["janeiro","fevereiro","março","abril","maio","junho","julho",
            "agosto","setembro","outubro","novembro","dezembro"]

# ================== HELPERS ==================
def formata_numero(valor, prefixo=''):
    try:
        valor = float(valor)
    except:
        return f'{prefixo} 0,00'
    for unidade in ['', 'mil']:
        if valor < 1000:
            return f'{prefixo} {valor:.2f} {unidade}'.strip()
        valor /= 1000
    return f'{prefixo} {valor:.2f} milhões'.strip()

def mes_ordem(df, col="Data da Compra"):
    tmp = df.set_index(col).copy()
    tmp["Ano"] = tmp.index.year
    tmp["Mes"] = tmp.index.month
    return tmp

def kpi(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else 0.0
    except Exception:
        return 0.0

def zscores(series):
    if series.std(ddof=0) == 0 or len(series) < 3:
        return pd.Series([0]*len(series), index=series.index)
    return (series - series.mean())/series.std(ddof=0)

def pareto_share(df, chave, valor):
    base = df.groupby(chave)[valor].sum().sort_values(ascending=False)
    if base.sum() == 0 or len(base) == 0:
        return base, 0, None
    cumul = base.cumsum()/base.sum()
    idx80 = np.argmax(cumul.values >= 0.8) + 1
    return base, idx80, cumul

def resumo_regras(dados):
    linhas = []
    if dados.empty:
        return ["Sem dados para os filtros atuais."]

    total_receita = kpi(dados["Preço"].sum())
    qtd_vendas = int(dados.shape[0])
    ticket_medio = total_receita/qtd_vendas if qtd_vendas else 0

    linhas += [
        f"Receita total: R$ {total_receita:,.2f}",
        f"Qtde de vendas: {qtd_vendas:,}",
        f"Ticket médio: R$ {ticket_medio:,.2f}",
    ]

    m = mes_ordem(dados)
    mensal = m.resample("M")["Preço"].sum()
    if len(mensal) >= 2 and mensal.iloc[-2] != 0:
        mom = (mensal.iloc[-1] - mensal.iloc[-2]) / mensal.iloc[-2] * 100
        linhas.append(f"Tendência MoM: {mom:+.1f}% (último mês vs. anterior)")
    if len(mensal) >= 3:
        z = zscores(mensal)
        outliers = mensal[z.abs() >= 2]
        if not outliers.empty:
            linhas.append(
                "⚠️ Possíveis anomalias mensais (|z|≥2): " +
                ", ".join([f"{idx.strftime('%b/%Y')}: R$ {val:,.0f}" for idx, val in outliers.items()])
            )

    cat_base, idx80, _ = pareto_share(dados, "Categoria do Produto", "Preço")
    if len(cat_base):
        top_cat = cat_base.index[:idx80].tolist()
        if idx80 > 0:
            linhas.append(f"Concentração 80/20 por categoria: ~{idx80} categorias explicam ~80% da receita.")
        if top_cat:
            linhas.append("Top categorias (até 80%): " + ", ".join(top_cat[:5]) + ("..." if len(top_cat) > 5 else ""))

    est_receita = dados.groupby("Local da compra")["Preço"].sum().sort_values(ascending=False).head(3)
    est_qtd = dados["Local da compra"].value_counts().head(3)
    if len(est_receita):
        linhas.append("🏆 Estados por receita: " + " | ".join([f"{k}: R$ {v:,.0f}" for k, v in est_receita.items()]))
    if len(est_qtd):
        linhas.append("📦 Estados por volume: " + " | ".join([f"{k}: {v}" for k, v in est_qtd.items()]))

    vend = dados.groupby("Vendedor")["Preço"].sum().sort_values(ascending=False)
    if len(vend) >= 3:
        zv = zscores(vend)
        tops = zv[zv >= 1].index.tolist()
        lows = zv[zv <= -1].index.tolist()
        if tops:
            linhas.append("🔥 Vendedores acima da média: " + ", ".join(tops[:5]) + ("..." if len(tops) > 5 else ""))
        if lows:
            linhas.append("🥶 Vendedores abaixo da média: " + ", ".join(lows[:5]) + ("..." if len(lows) > 5 else ""))

    return linhas

def responder_pergunta_simples(pergunta, df):
    p = pergunta.lower().strip()
    if "melhor mês" in p or "melhor mes" in p:
        m = mes_ordem(df).resample("M")["Preço"].sum()
        if m.empty: return "Sem dados."
        idx = m.idxmax()
        return f"Melhor mês em receita foi {idx.strftime('%B/%Y')} com R$ {m.max():,.2f}."
    if "pior mês" in p or "pior mes" in p:
        m = mes_ordem(df).resample("M")["Preço"].sum()
        if m.empty: return "Sem dados."
        idx = m.idxmin()
        return f"Pior mês em receita foi {idx.strftime('%B/%Y')} com R$ {m.min():,.2f}."
    if "top estados" in p or "top estados por receita" in p:
        est = df.groupby("Local da compra")["Preço"].sum().sort_values(ascending=False).head(5)
        if est.empty: return "Sem dados."
        return "Top estados por receita: " + " | ".join([f"{k}: R$ {v:,.0f}" for k, v in est.items()])
    if "ticket médio" in p or "ticket medio" in p:
        tm = df["Preço"].mean() if not df.empty else 0
        return f"Ticket médio: R$ {tm:,.2f}."
    return "Não entendi. Exemplos: 'melhor mês', 'pior mês', 'top estados', 'ticket médio'."

# ---------- OpenAI client cache ----------
@st.cache_resource
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# ---------- Requests session with retry ----------
def make_session():
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

@st.cache_data(ttl=600)
def fetch_dados(url, params):
    try:
        session = make_session()
        resp = session.get(
            url, params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15
        )
        if not resp.ok:
            return None, f"HTTP {resp.status_code} ao acessar a API."
        try:
            return pd.DataFrame(resp.json()), None
        except ValueError:
            return None, "A resposta não está em JSON (possível indisponibilidade ou bloqueio)."
    except requests.RequestException as e:
        return None, f"Erro de conexão: {e}"

# ================== APP ==================
st.title('DASHBOARD DE VENDAS :shopping_trolley:')

# ----- Filtros -----
url = 'https://labdados.com/produtos'
regioes = ['Brasil', 'Centro-Oeste', 'Nordeste', 'Norte', 'Sudeste', 'Sul']

st.sidebar.title('Filtros')
regiao = st.sidebar.selectbox('Região', regioes)
regiao_param = '' if regiao == 'Brasil' else regiao.lower()

todos_anos = st.sidebar.checkbox('Dados de todo o período', value=True)
ano = '' if todos_anos else st.sidebar.slider('Ano', 2020, 2023)

query_string = {'regiao': regiao_param, 'ano': ano}

with st.spinner("Carregando dados..."):
    dados, erro_api = fetch_dados(url, query_string)

if erro_api or dados is None or dados.empty:
    st.error("Não foi possível carregar os dados da API.")
    if erro_api:
        with st.expander("Detalhes técnicos"):
            st.code(erro_api)
    st.stop()

# Pré-processamento
dados['Data da Compra'] = pd.to_datetime(dados['Data da Compra'], format='%d/%m/%Y', errors='coerce')
dados["Preço"] = pd.to_numeric(dados["Preço"], errors="coerce").fillna(0.0)

filtro_vendedores = st.sidebar.multiselect('Vendedores', sorted(dados['Vendedor'].dropna().unique()))
if filtro_vendedores:
    dados = dados[dados['Vendedor'].isin(filtro_vendedores)]

# Botão para exportar dados filtrados
st.download_button(
    "Baixar dados filtrados (CSV)",
    data=dados.to_csv(index=False).encode("utf-8"),
    file_name="dados_filtrados.csv",
    mime="text/csv"
)

# ===== Tabelas =====
# Receita
receita_estados = dados.groupby('Local da compra')[['Preço']].sum()
receita_estados = dados.drop_duplicates(subset='Local da compra')[['Local da compra', 'lat', 'lon']]\
    .merge(receita_estados, left_on='Local da compra', right_index=True)\
    .sort_values('Preço', ascending=False)

receita_mensal = dados.set_index('Data da Compra').groupby(pd.Grouper(freq='M'))['Preço'].sum().reset_index()
receita_mensal['Ano'] = receita_mensal['Data da Compra'].dt.year
receita_mensal['MesNum'] = receita_mensal['Data da Compra'].dt.month
receita_mensal['Mes'] = receita_mensal['MesNum'].map(MAP_PT)

receita_categorias = dados.groupby('Categoria do Produto')[['Preço']].sum().sort_values('Preço', ascending=False)

# Vendas
vendas_estados = pd.DataFrame(dados.groupby('Local da compra')['Preço'].count())
vendas_estados = dados.drop_duplicates(subset='Local da compra')[['Local da compra', 'lat', 'lon']]\
    .merge(vendas_estados, left_on='Local da compra', right_index=True)\
    .sort_values('Preço', ascending=False)

vendas_mensal = pd.DataFrame(dados.set_index('Data da Compra').groupby(pd.Grouper(freq='M'))['Preço'].count()).reset_index()
vendas_mensal['Ano'] = vendas_mensal['Data da Compra'].dt.year
vendas_mensal['MesNum'] = vendas_mensal['Data da Compra'].dt.month
vendas_mensal['Mes'] = vendas_mensal['MesNum'].map(MAP_PT)

vendas_categorias = pd.DataFrame(dados.groupby('Categoria do Produto')['Preço'].count().sort_values(ascending=False))

# Vendedores
vendedores = pd.DataFrame(dados.groupby('Vendedor')['Preço'].agg(['sum', 'count']))

# ===== Gráficos =====
# Receita
fig_mapa_receita = px.scatter_geo(
    receita_estados,
    lat='lat',
    lon='lon',
    scope='south america',
    size='Preço',
    template='seaborn',
    hover_name='Local da compra',
    hover_data={'lat': False, 'lon': False},
    title='Receita por estado'
)

fig_receita_mensal = px.line(
    receita_mensal.sort_values(["Ano","MesNum"]),
    x='Mes', y='Preço', color='Ano', line_dash='Ano', markers=True,
    category_orders={"Mes": MESES_PT},
    range_y=(0, float(receita_mensal['Preço'].max()) if len(receita_mensal) else 0),
    title='Receita mensal'
)
fig_receita_mensal.update_layout(xaxis_title="", yaxis_title='Receita')

fig_receita_estados = px.bar(
    receita_estados.head(),
    x='Local da compra',
    y='Preço',
    text_auto=True,
    title='Top estados (receita)'
)
fig_receita_estados.update_layout(xaxis_title="", yaxis_title='Receita')

fig_receita_categorias = px.bar(
    receita_categorias,
    text_auto=True,
    title='Receita por categoria'
)
fig_receita_categorias.update_layout(xaxis_title="", yaxis_title='Receita')

# Vendas
fig_mapa_vendas = px.scatter_geo(
    vendas_estados,
    lat='lat',
    lon='lon',
    scope='south america',
    template='seaborn',
    size='Preço',
    hover_name='Local da compra',
    hover_data={'lat': False, 'lon': False},
    title='Vendas por estado',
)

fig_vendas_estados = px.bar(
    vendas_estados.head(),
    x='Local da compra',
    y='Preço',
    text_auto=True,
    title='Top 5 estados'
)
fig_vendas_estados.update_layout(xaxis_title="", yaxis_title='Quantidade')

fig_vendas_mensal = px.line(
    vendas_mensal.sort_values(["Ano","MesNum"]),
    x='Mes', y='Preço', color='Ano', line_dash='Ano', markers=True,
    category_orders={"Mes": MESES_PT},
    range_y=(0, float(vendas_mensal['Preço'].max()) if len(vendas_mensal) else 0),
    title='Quantidade de vendas mensal'
)
fig_vendas_mensal.update_layout(xaxis_title="", yaxis_title='Quantidade')

fig_vendas_categorias = px.bar(
    vendas_categorias,
    text_auto=True,
    title='Vendas por categoria'
)
fig_vendas_categorias.update_layout(xaxis_title="", yaxis_title='Quantidade', showlegend=False)

# ===== Visualização =====
aba1, aba2, aba3, abaIA = st.tabs(['Receita', 'Quantidade de vendas', 'Vendedores', 'Insights (IA)'])

with aba1:
    coluna1, coluna2 = st.columns(2)
    with coluna1:
        st.metric('Receita', formata_numero(dados['Preço'].sum(), 'R$'))
        st.plotly_chart(fig_mapa_receita, use_container_width=True)
        st.plotly_chart(fig_receita_estados, use_container_width=True)
    with coluna2:
        st.metric('Quantidade de vendas', formata_numero(dados.shape[0]))
        st.plotly_chart(fig_receita_mensal, use_container_width=True)
        st.plotly_chart(fig_receita_categorias, use_container_width=True)

with aba2:
    coluna1, coluna2 = st.columns(2)
    with coluna1:
        st.metric('Receita', formata_numero(dados['Preço'].sum(), 'R$'))
        st.plotly_chart(fig_mapa_vendas, use_container_width=True)
        st.plotly_chart(fig_vendas_estados, use_container_width=True)
    with coluna2:
        st.metric('Quantidade de vendas', formata_numero(dados.shape[0]))
        st.plotly_chart(fig_vendas_mensal, use_container_width=True)
        st.plotly_chart(fig_vendas_categorias, use_container_width=True)

with aba3:
    qtd_vendedores = st.number_input('Quantidade de vendedores', 2, 10, 5)
    coluna1, coluna2 = st.columns(2)

    top_sum = vendedores["sum"].nlargest(qtd_vendedores)
    top_cnt = vendedores["count"].nlargest(qtd_vendedores)

    with coluna1:
        st.metric('Receita', formata_numero(dados['Preço'].sum(), 'R$'))
        fig_receita_vendedores = px.bar(
            top_sum.sort_values(ascending=True),
            x=top_sum.values, y=top_sum.index,
            text_auto=True, title=f'Top {qtd_vendedores} vendedores (receita)'
        )
        fig_receita_vendedores.update_layout(xaxis_title="Receita", yaxis_title="")
        st.plotly_chart(fig_receita_vendedores, use_container_width=True)

    with coluna2:
        st.metric('Quantidade de vendas', formata_numero(dados.shape[0]))
        fig_vendas_vendedores = px.bar(
            top_cnt.sort_values(ascending=True),
            x=top_cnt.values, y=top_cnt.index,
            text_auto=True, title=f'Top {qtd_vendedores} vendedores (quantidade de vendas)'
        )
        fig_vendas_vendedores.update_layout(xaxis_title="Vendas", yaxis_title="")
        st.plotly_chart(fig_vendas_vendedores, use_container_width=True)

# ===== Insights (IA) =====
with abaIA:
    st.subheader("Resumo inteligente dos dados filtrados")
    insights = resumo_regras(dados)
    for linha in insights:
        st.write("• " + linha)

    st.divider()
    st.subheader("Resumo em linguagem natural (LLM)")

    colA, colB = st.columns(2)
    with colA:
        usar_llm = st.toggle("Gerar com LLM", value=False)
    with colB:
        temp = st.slider("Criatividade (temperature)", 0.0, 1.0, 0.2, 0.05)

    if usar_llm:
        client = get_openai_client()
        if not client:
            st.info("Adicione OPENAI_API_KEY em .streamlit/secrets.toml")
        else:
            try:
                prompt = (
                    "Você é um analista de RevOps. Escreva um resumo curto (4–6 linhas), "
                    "com insights e recomendações acionáveis, usando estes fatos:\n- "
                    + "\n- ".join(insights[:12])
                )
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",  # pode trocar por "gpt-4o"
                    messages=[{"role": "user", "content": prompt}],
                    temperature=float(temp),
                    max_tokens=300,
                )
                st.success(textwrap.fill(resp.choices[0].message.content, 110))
            except Exception as e:
                st.warning(f"LLM indisponível: {e}")

    st.divider()
    st.subheader("Pergunte aos dados")
    q = st.chat_input("Ex.: 'melhor mês', 'top estados', 'ticket médio'...")
    if q:
        st.chat_message("user").write(q)
        resposta = responder_pergunta_simples(q, dados)
        st.chat_message("assistant").write(resposta)