"""Simulazione numerica di un pendolo semplice con tre metodi di integrazione."""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass(frozen=True)
class Config:
    durata: float = 12.0
    passo: float = 0.01
    angolo_iniziale: float = math.radians(69.0)
    velocita_iniziale: float = 0.0
    lunghezza: float = 1.0
    gravita: float = 9.81
    smorzamento: float = 0.05
    angolo_secondo: float = math.radians(-30.0)
    lunghezza_seconda: float = 1.0
    massa_1: float = 1.0
    massa_2: float = 1.0


@dataclass
class Risultato:
    tempi: list[float]
    angoli: list[float]
    velocita: list[float]
    energie: list[float]


@dataclass
class RisultatoDoppio:
    tempi: list[float]
    theta1: list[float]
    theta2: list[float]
    omega1: list[float]
    omega2: list[float]
    energie: list[float]


def accelerazione(angolo: float, velocita: float, config: Config) -> float:
    """Accelerazione angolare del pendolo semplice con smorzamento viscoso."""
    return (
        -(config.gravita / config.lunghezza) * math.sin(angolo)
        - config.smorzamento * velocita
    )


def passo_eulero(angolo: float, velocita: float, dt: float, config: Config):
    nuova_accelerazione = accelerazione(angolo, velocita, config)
    return angolo + dt * velocita, velocita + dt * nuova_accelerazione


def passo_eulero_semi_implicito(
    angolo: float, velocita: float, dt: float, config: Config
):
    nuova_velocita = velocita + dt * accelerazione(angolo, velocita, config)
    return angolo + dt * nuova_velocita, nuova_velocita


def passo_rk4(angolo: float, velocita: float, dt: float, config: Config):
    def derivata(theta: float, omega: float):
        return omega, accelerazione(theta, omega, config)

    k1_theta, k1_omega = derivata(angolo, velocita)
    k2_theta, k2_omega = derivata(
        angolo + dt * k1_theta / 2, velocita + dt * k1_omega / 2
    )
    k3_theta, k3_omega = derivata(
        angolo + dt * k2_theta / 2, velocita + dt * k2_omega / 2
    )
    k4_theta, k4_omega = derivata(
        angolo + dt * k3_theta, velocita + dt * k3_omega
    )

    theta_nuovo = angolo + dt * (
        k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta
    ) / 6
    omega_nuova = velocita + dt * (
        k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega
    ) / 6
    return theta_nuovo, omega_nuova


def derivata_doppio(
    theta1: float, theta2: float, omega1: float, omega2: float, config: Config
):
    """Derivate del pendolo a due braccia in coordinate angolari."""
    m1 = config.massa_1
    m2 = config.massa_2
    l1 = config.lunghezza
    l2 = config.lunghezza_seconda
    g = config.gravita
    denominatore = 2 * m1 + m2 - m2 * math.cos(2 * theta1 - 2 * theta2)

    alpha1 = (
        -g * (2 * m1 + m2) * math.sin(theta1)
        - m2 * g * math.sin(theta1 - 2 * theta2)
        - 2
        * math.sin(theta1 - theta2)
        * m2
        * (omega2**2 * l2 + omega1**2 * l1 * math.cos(theta1 - theta2))
    ) / (l1 * denominatore)

    alpha2 = (
        2
        * math.sin(theta1 - theta2)
        * (
            omega1**2 * l1 * (m1 + m2)
            + g * (m1 + m2) * math.cos(theta1)
            + omega2**2 * l2 * m2 * math.cos(theta1 - theta2)
        )
    ) / (l2 * denominatore)

    return omega1, alpha1, omega2, alpha2


def passo_eulero_doppio(
    theta1: float, theta2: float, omega1: float, omega2: float, dt: float, config: Config
):
    omega1_nuovo, alpha1, omega2_nuovo, alpha2 = derivata_doppio(
        theta1, theta2, omega1, omega2, config
    )
    theta1_nuovo = theta1 + dt * omega1_nuovo
    theta2_nuovo = theta2 + dt * omega2_nuovo
    omega1_nuovo = omega1 + dt * alpha1
    omega2_nuovo = omega2 + dt * alpha2
    return theta1_nuovo, theta2_nuovo, omega1_nuovo, omega2_nuovo


def passo_eulero_semi_implicito_doppio(
    theta1: float, theta2: float, omega1: float, omega2: float, dt: float, config: Config
):
    omega1_nuovo, alpha1, omega2_nuovo, alpha2 = derivata_doppio(
        theta1, theta2, omega1, omega2, config
    )
    omega1_nuovo = omega1 + dt * alpha1
    omega2_nuovo = omega2 + dt * alpha2
    theta1_nuovo = theta1 + dt * omega1_nuovo
    theta2_nuovo = theta2 + dt * omega2_nuovo
    return theta1_nuovo, theta2_nuovo, omega1_nuovo, omega2_nuovo


def passo_rk4_doppio(
    theta1: float, theta2: float, omega1: float, omega2: float, dt: float, config: Config
):
    def derivata(state):
        t1, t2, o1, o2 = state
        _, alpha1, _, alpha2 = derivata_doppio(t1, t2, o1, o2, config)
        return [o1, o2, alpha1, alpha2]

    stato = [theta1, theta2, omega1, omega2]
    k1 = derivata(stato)
    k2 = derivata([s + dt * k / 2 for s, k in zip(stato, k1)])
    k3 = derivata([s + dt * k / 2 for s, k in zip(stato, k2)])
    k4 = derivata([s + dt * k for s, k in zip(stato, k3)])

    stato_nuovo = [
        s + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6
        for i, s in enumerate(stato)
    ]
    return tuple(stato_nuovo)


METODI = {
    "Euler esplicito": passo_eulero,
    "Euler semi-implicito": passo_eulero_semi_implicito,
    "Runge-Kutta 4": passo_rk4,
}
METODI_DOPPIO = {
    "Euler esplicito": passo_eulero_doppio,
    "Euler semi-implicito": passo_eulero_semi_implicito_doppio,
    "Runge-Kutta 4": passo_rk4_doppio,
}
COLORI = {
    "Euler esplicito": "#e76f51",
    "Euler semi-implicito": "#2a9d8f",
    "Runge-Kutta 4": "#4169e1",
}


def energia(angolo: float, velocita: float, config: Config) -> float:
    """Energia specifica del pendolo semplice."""
    cinetica = 0.5 * config.lunghezza**2 * velocita**2
    potenziale = config.gravita * config.lunghezza * (1 - math.cos(angolo))
    return cinetica + potenziale


def energia_doppio(
    theta1: float, theta2: float, omega1: float, omega2: float, config: Config
) -> float:
    """Energia meccanica del pendolo a due braccia."""
    l1 = config.lunghezza
    l2 = config.lunghezza_seconda
    m1 = config.massa_1
    m2 = config.massa_2
    g = config.gravita

    cinetica = 0.5 * (m1 + m2) * (l1 * omega1) ** 2 + 0.5 * m2 * (
        l2 * omega2
    ) ** 2 + m2 * l1 * l2 * omega1 * omega2 * math.cos(theta1 - theta2)
    potenziale = -(m1 + m2) * g * l1 * math.cos(theta1) - m2 * g * l2 * math.cos(theta2)
    return cinetica + potenziale


def simula(metodo: str, config: Config) -> Risultato:
    if metodo not in METODI:
        raise ValueError(f"Metodo sconosciuto: {metodo}")
    if config.durata <= 0 or config.passo <= 0 or config.lunghezza <= 0:
        raise ValueError("durata, passo e lunghezza devono essere maggiori di zero.")

    numero_passi = round(config.durata / config.passo)
    if numero_passi == 0:
        raise ValueError("durata e passo devono produrre almeno un passo di integrazione.")
    dt = config.durata / numero_passi
    tempi = [0.0]
    angoli = [config.angolo_iniziale]
    velocita = [config.velocita_iniziale]
    energie = [energia(angoli[0], velocita[0], config)]
    integra = METODI[metodo]

    for indice in range(numero_passi):
        theta, omega = integra(angoli[-1], velocita[-1], dt, config)
        tempi.append((indice + 1) * dt)
        angoli.append(theta)
        velocita.append(omega)
        energie.append(energia(theta, omega, config))

    return Risultato(tempi, angoli, velocita, energie)


def simula_doppio(metodo: str, config: Config) -> RisultatoDoppio:
    if metodo not in METODI_DOPPIO:
        raise ValueError(f"Metodo sconosciuto: {metodo}")
    if config.durata <= 0 or config.passo <= 0:
        raise ValueError("durata e passo devono essere maggiori di zero.")
    if config.lunghezza <= 0 or config.lunghezza_seconda <= 0:
        raise ValueError("le lunghezze devono essere positive.")
    if config.massa_1 <= 0 or config.massa_2 <= 0:
        raise ValueError("le masse devono essere positive.")

    numero_passi = round(config.durata / config.passo)
    if numero_passi == 0:
        raise ValueError("durata e passo devono produrre almeno un passo di integrazione.")
    dt = config.durata / numero_passi
    tempi = [0.0]
    theta1 = [config.angolo_iniziale]
    theta2 = [config.angolo_secondo]
    omega1 = [config.velocita_iniziale]
    omega2 = [config.velocita_iniziale]
    energie = [energia_doppio(theta1[0], theta2[0], omega1[0], omega2[0], config)]
    integra = METODI_DOPPIO[metodo]

    for indice in range(numero_passi):
        t1_nuovo, t2_nuovo, o1_nuovo, o2_nuovo = integra(
            theta1[-1], theta2[-1], omega1[-1], omega2[-1], dt, config
        )
        tempi.append((indice + 1) * dt)
        theta1.append(t1_nuovo)
        theta2.append(t2_nuovo)
        omega1.append(o1_nuovo)
        omega2.append(o2_nuovo)
        energie.append(energia_doppio(t1_nuovo, t2_nuovo, o1_nuovo, o2_nuovo, config))

    return RisultatoDoppio(tempi, theta1, theta2, omega1, omega2, energie)


def crea_grafico(risultati: dict[str, Risultato], percorso: Path) -> None:
    riferimento = risultati["Runge-Kutta 4"]
    energia_iniziale = riferimento.energie[0]
    fig, assi = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    fig.suptitle("Il pendolo semplice — confronto tra metodi numerici", fontsize=17)

    for nome, dati in risultati.items():
        colore = COLORI[nome]
        assi[0].plot(
            dati.tempi,
            [math.degrees(theta) for theta in dati.angoli],
            label=nome,
            color=colore,
            linewidth=1.8,
        )
        assi[1].plot(
            dati.angoli,
            dati.velocita,
            label=nome,
            color=colore,
            linewidth=1.5,
            alpha=0.9,
        )
        errore_energia = [
            (value - energia_iniziale) / energia_iniziale * 100
            for value in dati.energie
        ]
        assi[2].plot(
            dati.tempi,
            errore_energia,
            label=nome,
            color=colore,
            linewidth=1.5,
        )

    assi[0].set_title("Angolo nel tempo")
    assi[0].set_ylabel("Angolo (gradi)")
    assi[1].set_title("Spazio delle fasi")
    assi[1].set_ylabel("Velocità angolare (rad/s)")
    assi[2].set_title("Variazione dell'energia rispetto all'inizio")
    assi[2].set_ylabel("Variazione energia (%)")
    assi[2].set_xlabel("Tempo (s)")
    for asse in assi:
        asse.grid(True, alpha=0.25)
        asse.legend(loc="best")
    fig.tight_layout()
    fig.savefig(percorso, dpi=160, bbox_inches="tight")
    plt.close(fig)


def crea_grafico_doppio(risultati: dict[str, RisultatoDoppio], percorso: Path) -> None:
    riferimento = risultati["Runge-Kutta 4"]
    energia_iniziale = riferimento.energie[0]
    fig, assi = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    fig.suptitle("Pendolo a due braccia — confronto tra metodi numerici", fontsize=17)

    for nome, dati in risultati.items():
        colore = COLORI[nome]
        assi[0].plot(
            dati.tempi,
            [math.degrees(theta) for theta in dati.theta1],
            label=f"{nome} θ1",
            color=colore,
            linewidth=1.8,
        )
        assi[0].plot(
            dati.tempi,
            [math.degrees(theta) for theta in dati.theta2],
            label=f"{nome} θ2",
            color=colore,
            linewidth=1.8,
            linestyle="--",
            alpha=0.7,
        )
        assi[1].plot(
            [math.degrees(theta) for theta in dati.theta1],
            [math.degrees(theta) for theta in dati.theta2],
            label=nome,
            color=colore,
            linewidth=1.5,
            alpha=0.9,
        )
        errore_energia = []
        for value in dati.energie:
            if abs(energia_iniziale) < 1e-12:
                errore_energia.append(0.0)
            else:
                errore_energia.append((value - energia_iniziale) / energia_iniziale * 100)
        assi[2].plot(dati.tempi, errore_energia, label=nome, color=colore, linewidth=1.5)

    assi[0].set_title("Angoli nel tempo")
    assi[0].set_ylabel("Angolo (gradi)")
    assi[1].set_title("Traiettoria degli angoli")
    assi[1].set_xlabel("θ1 (gradi)")
    assi[1].set_ylabel("θ2 (gradi)")
    assi[2].set_title("Variazione dell'energia rispetto all'inizio")
    assi[2].set_ylabel("Variazione energia (%)")
    assi[2].set_xlabel("Tempo (s)")
    for asse in assi:
        asse.grid(True, alpha=0.25)
        asse.legend(loc="best")
    fig.tight_layout()
    fig.savefig(percorso, dpi=160, bbox_inches="tight")
    plt.close(fig)


def crea_animazione(risultato: Risultato, config: Config, percorso: Path) -> None:
    lunghezza = config.lunghezza
    altezza = 1.5 * lunghezza
    fig, asse = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#101827")
    asse.set_facecolor("#101827")
    asse.set_xlim(-altezza, altezza)
    asse.set_ylim(-1.2 * lunghezza, 0.35 * lunghezza)
    asse.set_aspect("equal")
    asse.axis("off")
    asse.set_title(
        "PENDOLO · RUNGE–KUTTA 4",
        color="white",
        fontsize=15,
        pad=18,
    )

    asse.plot(0, 0, marker="o", color="#f4a261", markersize=10, zorder=4)
    asta, = asse.plot([], [], color="#a8dadc", linewidth=3, zorder=2)
    massa, = asse.plot([], [], marker="o", color="#e76f51", markersize=22, zorder=3)
    traccia, = asse.plot([], [], color="#f4a261", alpha=0.5, linewidth=1.5)
    testo = asse.text(
        0.03,
        0.96,
        "",
        transform=asse.transAxes,
        color="white",
        va="top",
        fontsize=11,
    )

    salto = max(1, len(risultato.tempi) // 300)
    fotogrammi = list(range(0, len(risultato.tempi), salto))
    if fotogrammi[-1] != len(risultato.tempi) - 1:
        fotogrammi.append(len(risultato.tempi) - 1)

    def aggiorna(indice: int):
        theta = risultato.angoli[indice]
        x = lunghezza * math.sin(theta)
        y = -lunghezza * math.cos(theta)
        asta.set_data([0, x], [0, y])
        massa.set_data([x], [y])
        punti_traccia = fotogrammi[: fotogrammi.index(indice) + 1]
        traccia.set_data(
            [lunghezza * math.sin(risultato.angoli[i]) for i in punti_traccia],
            [-lunghezza * math.cos(risultato.angoli[i]) for i in punti_traccia],
        )
        testo.set_text(
            f"t = {risultato.tempi[indice]:.2f} s\n"
            f"θ = {math.degrees(theta):.1f}°"
        )
        return asta, massa, traccia, testo

    animazione = FuncAnimation(
        fig,
        aggiorna,
        frames=fotogrammi,
        interval=1000 / 25,
        blit=True,
        repeat=True,
    )
    animazione.save(percorso, writer=PillowWriter(fps=25))
    plt.close(fig)


def crea_animazione_doppio(risultato: RisultatoDoppio, config: Config, percorso: Path) -> None:
    l1 = config.lunghezza
    l2 = config.lunghezza_seconda
    scala = 1.5 * (l1 + l2)
    fig, asse = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#101827")
    asse.set_facecolor("#101827")
    asse.set_xlim(-scala, scala)
    asse.set_ylim(-1.4 * (l1 + l2), 0.5 * (l1 + l2))
    asse.set_aspect("equal")
    asse.axis("off")
    asse.set_title(
        "PENDOLO A DUE BRACCIA · RUNGE–KUTTA 4",
        color="white",
        fontsize=15,
        pad=18,
    )

    asse.plot(0, 0, marker="o", color="#f4a261", markersize=10, zorder=4)
    asta1, = asse.plot([], [], color="#a8dadc", linewidth=3, zorder=2)
    asta2, = asse.plot([], [], color="#90e0ef", linewidth=3, zorder=2)
    massa1, = asse.plot([], [], marker="o", color="#e76f51", markersize=18, zorder=3)
    massa2, = asse.plot([], [], marker="o", color="#f4a261", markersize=16, zorder=3)
    traccia, = asse.plot([], [], color="#f4a261", alpha=0.35, linewidth=1.5)
    testo = asse.text(
        0.03,
        0.96,
        "",
        transform=asse.transAxes,
        color="white",
        va="top",
        fontsize=11,
    )

    salto = max(1, len(risultato.tempi) // 300)
    fotogrammi = list(range(0, len(risultato.tempi), salto))
    if fotogrammi[-1] != len(risultato.tempi) - 1:
        fotogrammi.append(len(risultato.tempi) - 1)

    def aggiorna(indice: int):
        theta1 = risultato.theta1[indice]
        theta2 = risultato.theta2[indice]
        x1 = l1 * math.sin(theta1)
        y1 = -l1 * math.cos(theta1)
        x2 = x1 + l2 * math.sin(theta2)
        y2 = y1 - l2 * math.cos(theta2)
        asta1.set_data([0, x1], [0, y1])
        asta2.set_data([x1, x2], [y1, y2])
        massa1.set_data([x1], [y1])
        massa2.set_data([x2], [y2])
        punti_traccia = fotogrammi[: fotogrammi.index(indice) + 1]
        traccia_x = [
            l1 * math.sin(risultato.theta1[i]) + l2 * math.sin(risultato.theta2[i])
            for i in punti_traccia
        ]
        traccia_y = [
            -l1 * math.cos(risultato.theta1[i]) - l2 * math.cos(risultato.theta2[i])
            for i in punti_traccia
        ]
        traccia.set_data(traccia_x, traccia_y)
        testo.set_text(
            f"t = {risultato.tempi[indice]:.2f} s\n"
            f"θ1 = {math.degrees(theta1):.1f}°\n"
            f"θ2 = {math.degrees(theta2):.1f}°"
        )
        return asta1, asta2, massa1, massa2, traccia, testo

    animazione = FuncAnimation(
        fig,
        aggiorna,
        frames=fotogrammi,
        interval=1000 / 25,
        blit=True,
        repeat=True,
    )
    animazione.save(percorso, writer=PillowWriter(fps=25))
    plt.close(fig)


def argomenti() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simula un pendolo semplice o a due braccia e confronta Euler, Euler semi-implicito e RK4."
    )
    parser.add_argument("--modello", choices=["singolo", "doppio"], default="singolo", help="Tipo di pendolo da simulare.")
    parser.add_argument("--durata", type=float, default=12.0, help="Durata in secondi.")
    parser.add_argument("--passo", type=float, default=0.01, help="Passo temporale in secondi.")
    parser.add_argument(
        "--angolo",
        type=float,
        default=69.0,
        help="Angolo iniziale in gradi (predefinito: 69°).",
    )
    parser.add_argument(
        "--angolo2",
        type=float,
        default=-30.0,
        help="Secondo angolo iniziale del pendolo doppio in gradi.",
    )
    parser.add_argument("--lunghezza", type=float, default=1.0, help="Lunghezza della prima braccia in metri.")
    parser.add_argument(
        "--lunghezza-seconda",
        dest="lunghezza_seconda",
        type=float,
        default=1.0,
        help="Lunghezza della seconda braccio in metri.",
    )
    parser.add_argument("--massa-1", type=float, default=1.0, help="Massa della prima braccio.")
    parser.add_argument("--massa-2", type=float, default=1.0, help="Massa della seconda braccio.")
    parser.add_argument(
        "--smorzamento",
        type=float,
        default=0.05,
        help="Smorzamento viscoso (0 = pendolo ideale).",
    )
    parser.add_argument(
        "--cartella",
        type=Path,
        default=Path("."),
        help="Cartella in cui salvare grafico e animazione.",
    )
    parser.add_argument(
        "--solo-grafico",
        action="store_true",
        help="Salva il grafico senza generare la GIF.",
    )
    args = parser.parse_args()
    if args.durata <= 0 or args.passo <= 0 or args.lunghezza <= 0:
        parser.error("durata, passo e lunghezza devono essere maggiori di zero.")
    if args.lunghezza_seconda <= 0 or args.massa_1 <= 0 or args.massa_2 <= 0:
        parser.error("lunghezza_seconda, massa_1 e massa_2 devono essere positive.")
    if args.smorzamento < 0:
        parser.error("lo smorzamento non può essere negativo.")
    if not -180 <= args.angolo <= 180:
        parser.error("l'angolo iniziale deve essere compreso tra -180 e 180 gradi.")
    if args.modello == "doppio" and not -180 <= args.angolo2 <= 180:
        parser.error("il secondo angolo iniziale deve essere compreso tra -180 e 180 gradi.")
    if args.passo > args.durata:
        parser.error("il passo temporale non può essere maggiore della durata.")
    return args


def main() -> None:
    args = argomenti()
    args.cartella.mkdir(parents=True, exist_ok=True)

    if args.modello == "singolo":
        config = Config(
            durata=args.durata,
            passo=args.passo,
            angolo_iniziale=math.radians(args.angolo),
            lunghezza=args.lunghezza,
            smorzamento=args.smorzamento,
        )
        risultati = {nome: simula(nome, config) for nome in METODI}
        grafico = args.cartella / "pendolo_confronto.png"
        crea_grafico(risultati, grafico)
        print(f"Grafico salvato in: {grafico.resolve()}")
        if not args.solo_grafico:
            animazione = args.cartella / "pendolo_animation.gif"
            crea_animazione(risultati["Runge-Kutta 4"], config, animazione)
            print(f"Animazione salvata in: {animazione.resolve()}")
        print(
            f"Simulati {len(risultati['Runge-Kutta 4'].tempi) - 1} passi "
            f"da {config.passo:g} s per {config.durata:g} s."
        )
        return

    config = Config(
        durata=args.durata,
        passo=args.passo,
        angolo_iniziale=math.radians(args.angolo),
        angolo_secondo=math.radians(args.angolo2),
        lunghezza=args.lunghezza,
        lunghezza_seconda=args.lunghezza_seconda,
        massa_1=args.massa_1,
        massa_2=args.massa_2,
        smorzamento=args.smorzamento,
    )
    risultati = {nome: simula_doppio(nome, config) for nome in METODI_DOPPIO}
    grafico = args.cartella / "pendolo_doppio_confronto.png"
    crea_grafico_doppio(risultati, grafico)

    print(f"Grafico pendolo doppio salvato in: {grafico.resolve()}")
    if not args.solo_grafico:
        animazione = args.cartella / "pendolo_doppio_animation.gif"
        crea_animazione_doppio(risultati["Runge-Kutta 4"], config, animazione)
        print(f"Animazione pendolo doppio salvata in: {animazione.resolve()}")
    print(
        f"Simulati {len(risultati['Runge-Kutta 4'].tempi) - 1} passi "
        f"da {config.passo:g} s per {config.durata:g} s."
    )


if __name__ == "__main__":
    main()
