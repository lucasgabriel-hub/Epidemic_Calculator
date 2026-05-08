"""
========================================================
  CALCULADORA EPIDÊMICA DA DENGUE — Modelo SEIR-SEI
========================================================
Modelo compartimental para dengue com população humana
(SEIR) e população de mosquitos Aedes aegypti (SEI).

Compartimentos:
  Humanos  → S_h, E_h, I_h, R_h
  Mosquito → S_m, E_m, I_m

Autor: Calculadora Epidêmica de Dengue
Referência: Esteva & Vargas (1998), Chowell et al. (2007)
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# Pasta onde os gráficos serão salvos (mesma pasta do script)
PASTA_SAIDA = os.path.dirname(os.path.abspath(__file__))

def caminho_saida(nome_arquivo):
    return os.path.join(PASTA_SAIDA, nome_arquivo)

# ──────────────────────────────────────────────────────
#  PARÂMETROS PADRÃO (biologicamente embasados)
# ──────────────────────────────────────────────────────
PARAMETROS_PADRAO = {
    # ── Humanos ──────────────────────────────────────
    "N_h"       : 100_000,   # população humana total
    "mu_h"      : 1/70/365,  # taxa de mortalidade humana (1/vida média em dias)
    "sigma_h"   : 1/5,       # taxa de progressão E→I (1/período de incubação ~5 dias)
    "gamma_h"   : 1/7,       # taxa de recuperação (1/duração infecção ~7 dias)
    "b_h"       : 0.5,       # probabilidade de infecção (mosquito→humano) por picada
    # ── Mosquitos ────────────────────────────────────
    "N_m"       : 200_000,   # população de mosquitos (razão ~2:1 com humanos)
    "mu_m"      : 1/14,      # taxa de mortalidade do mosquito (vida média ~14 dias)
    "sigma_m"   : 1/10,      # taxa de progressão E→I no mosquito (~10 dias de incubação)
    "b_m"       : 0.3,       # probabilidade de infecção (humano→mosquito) por picada
    # ── Transmissão ──────────────────────────────────
    "beta"      : 0.75,      # taxa de picadas por mosquito por dia
    # ── Condições iniciais ───────────────────────────
    "I_h0"      : 10,        # humanos infectados inicialmente
    "I_m0"      : 50,        # mosquitos infectados inicialmente
    # ── Simulação ────────────────────────────────────
    "dias"      : 365,       # duração da simulação (dias)
}


# ──────────────────────────────────────────────────────
#  SISTEMA DE EQUAÇÕES DIFERENCIAIS
# ──────────────────────────────────────────────────────
def sistema_dengue(t, y, params):
    """
    Sistema de EDOs do modelo SEIR (humanos) + SEI (mosquitos).

    Estado: [S_h, E_h, I_h, R_h, S_m, E_m, I_m]
    """
    S_h, E_h, I_h, R_h, S_m, E_m, I_m = y

    N_h    = params["N_h"]
    mu_h   = params["mu_h"]
    sigma_h= params["sigma_h"]
    gamma_h= params["gamma_h"]
    b_h    = params["b_h"]

    N_m    = params["N_m"]
    mu_m   = params["mu_m"]
    sigma_m= params["sigma_m"]
    b_m    = params["b_m"]

    beta   = params["beta"]

    # Forças de infecção
    lambda_h = beta * b_h * (I_m / N_h)   # humano infectado por mosquito
    lambda_m = beta * b_m * (I_h / N_h)   # mosquito infectado por humano

    # Equações — Humanos
    dS_h = mu_h * N_h - lambda_h * S_h - mu_h * S_h
    dE_h = lambda_h * S_h - sigma_h * E_h - mu_h * E_h
    dI_h = sigma_h * E_h - gamma_h * I_h - mu_h * I_h
    dR_h = gamma_h * I_h - mu_h * R_h

    # Equações — Mosquitos
    dS_m = mu_m * N_m - lambda_m * S_m - mu_m * S_m
    dE_m = lambda_m * S_m - sigma_m * E_m - mu_m * E_m
    dI_m = sigma_m * E_m - mu_m * I_m

    return [dS_h, dE_h, dI_h, dR_h, dS_m, dE_m, dI_m]


# ──────────────────────────────────────────────────────
#  NÚMERO REPRODUTIVO BÁSICO (R₀)
# ──────────────────────────────────────────────────────
def calcular_R0(params):
    """
    R₀ para o modelo dengue de dois hospedeiros.

    R₀ = sqrt( (beta² · b_h · b_m · N_m · sigma_h · sigma_m) /
               (N_h · (sigma_h+mu_h)(gamma_h+mu_h)(mu_m)(sigma_m+mu_m)) )
    """
    p = params
    numerador   = (p["beta"]**2 * p["b_h"] * p["b_m"] * p["N_m"]
                   * p["sigma_h"] * p["sigma_m"])
    denominador = (p["N_h"]
                   * (p["sigma_h"] + p["mu_h"])
                   * (p["gamma_h"] + p["mu_h"])
                   * p["mu_m"]
                   * (p["sigma_m"] + p["mu_m"]))
    return np.sqrt(numerador / denominador)


# ──────────────────────────────────────────────────────
#  SIMULAÇÃO
# ──────────────────────────────────────────────────────
def simular(params=None):
    if params is None:
        params = PARAMETROS_PADRAO.copy()

    N_h = params["N_h"]
    N_m = params["N_m"]
    I_h0 = params["I_h0"]
    I_m0 = params["I_m0"]

    # Condições iniciais
    S_h0 = N_h - I_h0
    E_h0 = 0
    R_h0 = 0
    S_m0 = N_m - I_m0
    E_m0 = 0
    y0 = [S_h0, E_h0, I_h0, R_h0, S_m0, E_m0, I_m0]

    t_span = (0, params["dias"])
    t_eval = np.linspace(0, params["dias"], params["dias"] * 4)

    sol = solve_ivp(
        fun=lambda t, y: sistema_dengue(t, y, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    return sol, params


# ──────────────────────────────────────────────────────
#  INDICADORES EPIDEMIOLÓGICOS
# ──────────────────────────────────────────────────────
def calcular_indicadores(sol, params):
    t  = sol.t
    Ih = sol.y[2]
    Rh = sol.y[3]
    N_h = params["N_h"]

    pico_infectados   = Ih.max()
    dia_pico          = t[Ih.argmax()]
    total_infectados  = Rh[-1]          # acumulado de recuperados ≈ total de casos
    ataque            = total_infectados / N_h * 100
    duracao_epidemia  = estimar_duracao(t, Ih, N_h)

    return {
        "R0"                   : calcular_R0(params),
        "pico_infectados"      : pico_infectados,
        "dia_pico"             : dia_pico,
        "total_casos"          : total_infectados,
        "taxa_ataque_pct"      : ataque,
        "duracao_epidemia_dias": duracao_epidemia,
    }


def estimar_duracao(t, Ih, N_h, limiar=0.001):
    """Duração da epidemia: intervalo com I_h > limiar * N_h."""
    threshold = limiar * N_h
    ativo = Ih > threshold
    if ativo.any():
        inicio = t[np.argmax(ativo)]
        fim    = t[len(ativo) - 1 - np.argmax(ativo[::-1])]
        return fim - inicio
    return 0


# ──────────────────────────────────────────────────────
#  VISUALIZAÇÃO
# ──────────────────────────────────────────────────────
CORES = {
    "S_h": "#2196F3",
    "E_h": "#FF9800",
    "I_h": "#F44336",
    "R_h": "#4CAF50",
    "S_m": "#90CAF9",
    "E_m": "#FFCC80",
    "I_m": "#EF9A9A",
    "bg" : "#0F1117",
    "fg" : "#E8EAF6",
    "grid": "#1E2230",
    "painel": "#161B27",
}


def plotar_resultado(sol, params, indicadores):
    t  = sol.t
    Sh, Eh, Ih, Rh, Sm, Em, Im = sol.y

    fig = plt.figure(figsize=(18, 12), facecolor=CORES["bg"])
    fig.suptitle(
        "CALCULADORA EPIDÊMICA DA DENGUE — Modelo SEIR-SEI",
        fontsize=18, fontweight="bold", color=CORES["fg"],
        y=0.97, family="monospace"
    )

    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35,
                  left=0.07, right=0.97, top=0.92, bottom=0.07)

    # ── Painel 1: Dinâmica humana ─────────────────────
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.set_facecolor(CORES["painel"])
    ax1.plot(t, Sh, color=CORES["S_h"], lw=2, label="Suscetíveis (S_h)")
    ax1.plot(t, Eh, color=CORES["E_h"], lw=2, label="Expostos (E_h)")
    ax1.plot(t, Ih, color=CORES["I_h"], lw=2.5, label="Infectados (I_h)")
    ax1.plot(t, Rh, color=CORES["R_h"], lw=2, label="Recuperados (R_h)")
    ax1.axvline(indicadores["dia_pico"], color=CORES["I_h"],
                ls="--", lw=1.2, alpha=0.6)
    ax1.text(indicadores["dia_pico"] + 3, Ih.max() * 0.97,
             f"  Pico: dia {int(indicadores['dia_pico'])}",
             color=CORES["I_h"], fontsize=9, family="monospace")
    ax1.set_title("Dinâmica Humana", color=CORES["fg"], fontsize=12, pad=8)
    ax1.set_xlabel("Tempo (dias)", color=CORES["fg"])
    ax1.set_ylabel("Indivíduos", color=CORES["fg"])
    ax1.tick_params(colors=CORES["fg"])
    ax1.grid(color=CORES["grid"], lw=0.5)
    for spine in ax1.spines.values():
        spine.set_edgecolor(CORES["grid"])
    ax1.legend(facecolor=CORES["painel"], edgecolor=CORES["grid"],
               labelcolor=CORES["fg"], fontsize=9)

    # ── Painel 2: Dinâmica dos mosquitos ─────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(CORES["painel"])
    ax2.plot(t, Sm, color=CORES["S_m"], lw=2, label="S_m")
    ax2.plot(t, Em, color=CORES["E_m"], lw=2, label="E_m")
    ax2.plot(t, Im, color=CORES["I_m"], lw=2, label="I_m")
    ax2.set_title("Dinâmica dos Mosquitos", color=CORES["fg"], fontsize=11, pad=8)
    ax2.set_xlabel("Dias", color=CORES["fg"])
    ax2.set_ylabel("Mosquitos", color=CORES["fg"])
    ax2.tick_params(colors=CORES["fg"])
    ax2.grid(color=CORES["grid"], lw=0.5)
    for spine in ax2.spines.values():
        spine.set_edgecolor(CORES["grid"])
    ax2.legend(facecolor=CORES["painel"], edgecolor=CORES["grid"],
               labelcolor=CORES["fg"], fontsize=9)

    # ── Painel 3: Curva epidêmica (novos casos/dia) ──
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor(CORES["painel"])
    novos_casos = np.gradient(Rh, t)
    novos_casos = np.clip(novos_casos, 0, None)
    ax3.fill_between(t, novos_casos, color=CORES["I_h"], alpha=0.7)
    ax3.plot(t, novos_casos, color=CORES["I_h"], lw=1.5)
    ax3.set_title("Novos Casos / Dia", color=CORES["fg"], fontsize=11, pad=8)
    ax3.set_xlabel("Dias", color=CORES["fg"])
    ax3.set_ylabel("Casos", color=CORES["fg"])
    ax3.tick_params(colors=CORES["fg"])
    ax3.grid(color=CORES["grid"], lw=0.5)
    for spine in ax3.spines.values():
        spine.set_edgecolor(CORES["grid"])

    # ── Painel 4: Prevalência (%) ────────────────────
    ax4 = fig.add_subplot(gs[2, 0:2])
    ax4.set_facecolor(CORES["painel"])
    N_h = params["N_h"]
    ax4.stackplot(t,
                  Eh / N_h * 100,
                  Ih / N_h * 100,
                  Rh / N_h * 100,
                  colors=[CORES["E_h"], CORES["I_h"], CORES["R_h"]],
                  alpha=0.75,
                  labels=["Expostos %", "Infectados %", "Recuperados %"])
    ax4.set_title("Composição da População Humana (%)", color=CORES["fg"],
                  fontsize=11, pad=8)
    ax4.set_xlabel("Dias", color=CORES["fg"])
    ax4.set_ylabel("% da População", color=CORES["fg"])
    ax4.tick_params(colors=CORES["fg"])
    ax4.grid(color=CORES["grid"], lw=0.5, axis="y")
    ax4.set_xlim(0, params["dias"])
    ax4.set_ylim(0, 100)
    for spine in ax4.spines.values():
        spine.set_edgecolor(CORES["grid"])
    ax4.legend(facecolor=CORES["painel"], edgecolor=CORES["grid"],
               labelcolor=CORES["fg"], fontsize=9, loc="upper right")

    # ── Painel 5: Indicadores-chave ──────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_facecolor(CORES["painel"])
    ax5.axis("off")

    R0 = indicadores["R0"]
    cor_R0 = "#F44336" if R0 > 1 else "#4CAF50"
    status  = "EPIDEMIA" if R0 > 1 else "CONTROLADO"

    linhas = [
        ("R₀", f"{R0:.3f}", cor_R0),
        ("Status", status, cor_R0),
        ("Pico infectados", f"{int(indicadores['pico_infectados']):,}".replace(",", "."), CORES["I_h"]),
        ("Dia do pico", f"dia {int(indicadores['dia_pico'])}", CORES["fg"]),
        ("Total de casos", f"{int(indicadores['total_casos']):,}".replace(",", "."), CORES["R_h"]),
        ("Taxa de ataque", f"{indicadores['taxa_ataque_pct']:.1f}%", CORES["E_h"]),
        ("Duração", f"{int(indicadores['duracao_epidemia_dias'])} dias", CORES["S_h"]),
    ]

    ax5.set_title("Indicadores", color=CORES["fg"], fontsize=11, pad=8)
    for i, (label, valor, cor) in enumerate(linhas):
        y = 0.90 - i * 0.13
        ax5.text(0.05, y, label + ":", color=CORES["fg"], fontsize=9,
                 transform=ax5.transAxes, va="top", family="monospace")
        ax5.text(0.95, y, valor, color=cor, fontsize=10, fontweight="bold",
                 transform=ax5.transAxes, va="top", ha="right", family="monospace")
        ax5.plot([0.03, 0.97], [y - 0.01, y - 0.01], color=CORES["grid"],
                 lw=0.5, transform=ax5.transAxes, clip_on=False)

    plt.savefig(caminho_saida("dengue_epidemia.png"),
                dpi=150, bbox_inches="tight", facecolor=CORES["bg"])
    plt.show()
    print("\n  Gráfico salvo em: dengue_epidemia.png")


# ──────────────────────────────────────────────────────
#  ANÁLISE DE SENSIBILIDADE DO R₀
# ──────────────────────────────────────────────────────
def analise_sensibilidade():
    """Mostra como o R₀ varia com beta e razão mosquito/humano."""
    base = PARAMETROS_PADRAO.copy()
    betas  = np.linspace(0.1, 1.5, 50)
    razoes = [0.5, 1.0, 2.0, 5.0]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=CORES["bg"])
    ax.set_facecolor(CORES["painel"])

    cores_linha = ["#64B5F6", "#FFB74D", "#EF5350", "#AB47BC"]
    for razao, cor in zip(razoes, cores_linha):
        R0s = []
        for b in betas:
            p = base.copy()
            p["beta"] = b
            p["N_m"]  = int(p["N_h"] * razao)
            R0s.append(calcular_R0(p))
        ax.plot(betas, R0s, lw=2.5, color=cor,
                label=f"Mosquitos/Humanos = {razao}×")

    ax.axhline(1, color="white", ls="--", lw=1, alpha=0.5, label="R₀ = 1 (limiar)")
    ax.set_title("Sensibilidade do R₀ à Taxa de Picadas (β)",
                 color=CORES["fg"], fontsize=13)
    ax.set_xlabel("β — picadas/mosquito/dia", color=CORES["fg"])
    ax.set_ylabel("R₀", color=CORES["fg"])
    ax.tick_params(colors=CORES["fg"])
    ax.grid(color=CORES["grid"], lw=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(CORES["grid"])
    ax.legend(facecolor=CORES["painel"], edgecolor=CORES["grid"],
              labelcolor=CORES["fg"])

    plt.tight_layout()
    plt.savefig(caminho_saida("dengue_sensibilidade.png"),
                dpi=150, bbox_inches="tight", facecolor=CORES["bg"])
    plt.show()
    print("  Gráfico salvo em: dengue_sensibilidade.png")


# ──────────────────────────────────────────────────────
#  COMPARAÇÃO DE CENÁRIOS
# ──────────────────────────────────────────────────────
def comparar_cenarios():
    """Compara três cenários: sem controle, controle moderado, controle intenso."""
    cenarios = {
        "Sem controle"      : PARAMETROS_PADRAO.copy(),
        "Controle moderado" : {**PARAMETROS_PADRAO, "beta": 0.4, "N_m": 100_000},
        "Controle intenso"  : {**PARAMETROS_PADRAO, "beta": 0.2, "N_m":  50_000},
    }
    cores_cen = ["#F44336", "#FF9800", "#4CAF50"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=CORES["bg"])
    for ax in axes:
        ax.set_facecolor(CORES["painel"])

    print("\n" + "═"*62)
    print(f"  {'CENÁRIO':<22} {'R₀':>8}  {'PICO':>10}  {'TOTAL CASOS':>12}")
    print("═"*62)

    for (nome, params), cor in zip(cenarios.items(), cores_cen):
        sol, _ = simular(params)
        ind = calcular_indicadores(sol, params)
        t  = sol.t
        Ih = sol.y[2]
        Rh = sol.y[3]

        axes[0].plot(t, Ih, color=cor, lw=2.5, label=nome)
        axes[1].plot(t, Rh, color=cor, lw=2.5, label=nome)

        print(f"  {nome:<22} {ind['R0']:>8.3f}  "
              f"{int(ind['pico_infectados']):>10,}  "
              f"{int(ind['total_casos']):>12,}".replace(",", "."))

    print("═"*62)

    for ax, titulo, ylabel in zip(
        axes,
        ["Infectados Simultâneos", "Casos Acumulados (Recuperados)"],
        ["Indivíduos", "Indivíduos"]
    ):
        ax.set_title(titulo, color=CORES["fg"], fontsize=12)
        ax.set_xlabel("Dias", color=CORES["fg"])
        ax.set_ylabel(ylabel, color=CORES["fg"])
        ax.tick_params(colors=CORES["fg"])
        ax.grid(color=CORES["grid"], lw=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(CORES["grid"])
        ax.legend(facecolor=CORES["painel"], edgecolor=CORES["grid"],
                  labelcolor=CORES["fg"])

    fig.suptitle("Comparação de Cenários de Controle da Dengue",
                 color=CORES["fg"], fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(caminho_saida("dengue_cenarios.png"),
                dpi=150, bbox_inches="tight", facecolor=CORES["bg"])
    plt.show()
    print("  Gráfico salvo em: dengue_cenarios.png")


# ──────────────────────────────────────────────────────
#  RELATÓRIO NO TERMINAL
# ──────────────────────────────────────────────────────
def imprimir_relatorio(params, indicadores):
    R0 = indicadores["R0"]
    status = "🔴 EPIDEMIA" if R0 > 1 else "🟢 CONTROLADO"

    print("\n" + "═"*60)
    print("  RELATÓRIO EPIDEMIOLÓGICO — DENGUE")
    print("═"*60)
    print(f"\n  PARÂMETROS DO MODELO")
    print(f"  {'População humana':<30} {params['N_h']:>12,}".replace(",", "."))
    print(f"  {'Pop. de mosquitos':<30} {params['N_m']:>12,}".replace(",", "."))
    print(f"  {'Taxa de picadas (β)':<30} {params['beta']:>12.2f} picadas/dia")
    print(f"  {'Prob. infecção mosq→hum':<30} {params['b_h']:>12.2f}")
    print(f"  {'Prob. infecção hum→mosq':<30} {params['b_m']:>12.2f}")
    print(f"  {'Período de incubação (h)':<30} {1/params['sigma_h']:>12.1f} dias")
    print(f"  {'Período infeccioso (h)':<30} {1/params['gamma_h']:>12.1f} dias")

    print(f"\n  INDICADORES CALCULADOS")
    print(f"  {'R₀ (número reprodutivo básico)':<30} {R0:>12.4f}")
    print(f"  {'Status':<30} {status:>12}")
    print(f"  {'Pico de infectados':<30} {int(indicadores['pico_infectados']):>12,}".replace(",", "."))
    print(f"  {'Dia do pico':<30} {int(indicadores['dia_pico']):>12}")
    print(f"  {'Total de casos':<30} {int(indicadores['total_casos']):>12,}".replace(",", "."))
    print(f"  {'Taxa de ataque':<30} {indicadores['taxa_ataque_pct']:>11.1f}%")
    print(f"  {'Duração da epidemia':<30} {int(indicadores['duracao_epidemia_dias']):>11} dias")
    print("═"*60)


# ──────────────────────────────────────────────────────
#  ENTRADA INTERATIVA
# ──────────────────────────────────────────────────────
def menu_interativo():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║   CALCULADORA EPIDÊMICA DA DENGUE — Modelo SEIR-SEI   ║")
    print("╚" + "═"*58 + "╝")
    print("""
  Escolha o modo de execução:

    [1]  Simulação com parâmetros padrão
    [2]  Simulação com parâmetros personalizados
    [3]  Comparação de cenários de controle
    [4]  Análise de sensibilidade do R₀
    [5]  Executar tudo
    [0]  Sair
""")
    return input("  → Opção: ").strip()


def obter_float(mensagem, padrao):
    try:
        val = input(f"  {mensagem} [{padrao}]: ").strip()
        return float(val) if val else padrao
    except ValueError:
        print("    Valor inválido. Usando padrão.")
        return padrao


def obter_int(mensagem, padrao):
    try:
        val = input(f"  {mensagem} [{padrao}]: ").strip()
        return int(val) if val else padrao
    except ValueError:
        print("    Valor inválido. Usando padrão.")
        return padrao


def personalizar_parametros():
    print("\n  ── Parâmetros personalizados (Enter = valor padrão) ──")
    p = PARAMETROS_PADRAO.copy()
    p["N_h"]   = obter_int  ("População humana                    ", p["N_h"])
    p["N_m"]   = obter_int  ("População de mosquitos              ", p["N_m"])
    p["beta"]  = obter_float("Taxa de picadas β (picadas/dia)     ", p["beta"])
    p["b_h"]   = obter_float("Prob. infecção mosquito→humano      ", p["b_h"])
    p["b_m"]   = obter_float("Prob. infecção humano→mosquito      ", p["b_m"])
    p["gamma_h"]= obter_float("Taxa de recuperação (1/dias)        ", p["gamma_h"])
    p["sigma_h"]= obter_float("Taxa incubação humana (1/dias)      ", p["sigma_h"])
    p["I_h0"]  = obter_int  ("Humanos infectados inicialmente     ", p["I_h0"])
    p["I_m0"]  = obter_int  ("Mosquitos infectados inicialmente   ", p["I_m0"])
    p["dias"]  = obter_int  ("Duração da simulação (dias)         ", p["dias"])
    return p


# ──────────────────────────────────────────────────────
#  UTILITÁRIO: LIMPAR TELA
# ──────────────────────────────────────────────────────
def limpar_tela():
    import os
    os.system("cls" if os.name == "nt" else "clear")


def aguardar_enter(mensagem="  Pressione Enter para voltar ao menu..."):
    input(f"\n{mensagem}")
    limpar_tela()


# ──────────────────────────────────────────────────────
#  PONTO DE ENTRADA
# ──────────────────────────────────────────────────────
def main():
    limpar_tela()
    while True:
        opcao = menu_interativo()

        if opcao == "0":
            limpar_tela()
            print("\n  Encerrando. Até mais!\n")
            break

        elif opcao in ("1", "2", "5"):
            limpar_tela()
            params = personalizar_parametros() if opcao == "2" else PARAMETROS_PADRAO.copy()
            print("\n  Executando simulação...")
            sol, params = simular(params)
            ind = calcular_indicadores(sol, params)
            imprimir_relatorio(params, ind)
            plotar_resultado(sol, params, ind)

            if opcao == "5":
                aguardar_enter("  Pressione Enter para ver a comparação de cenários...")
                comparar_cenarios()
                aguardar_enter("  Pressione Enter para ver a análise de sensibilidade...")
                analise_sensibilidade()

            aguardar_enter()

        elif opcao == "3":
            limpar_tela()
            comparar_cenarios()
            aguardar_enter()

        elif opcao == "4":
            limpar_tela()
            analise_sensibilidade()
            aguardar_enter()

        else:
            print("\n  Opção inválida. Tente novamente.")
            aguardar_enter()


if __name__ == "__main__":
    main()