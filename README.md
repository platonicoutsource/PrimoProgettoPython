# Simulazione del pendolo

Un pendolo semplice simulato in Python con tre metodi numerici:

- **Euler esplicito**: facile da implementare, ma può accumulare errori di energia.
- **Euler semi-implicito**: aggiorna prima la velocità; spesso è più stabile per i sistemi oscillanti.
- **Runge-Kutta del quarto ordine (RK4)**: più preciso a parità di passo; usato per l'animazione.

Il modello usa l'equazione

```text
θ' = ω
ω' = -(g / L) sin(θ) - cω
```

dove `θ` è l'angolo in radianti, `ω` la velocità angolare, `g` la gravità,
`L` la lunghezza e `c` lo smorzamento viscoso. La massa è omessa perché si
semplifica nell'equazione del moto.

## Avvio

Installa le dipendenze e avvia il programma:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Vengono creati `pendolo_confronto.png`, con l'angolo, lo spazio delle fasi e la
variazione dell'energia, e `pendolo_animation.gif`, con il pendolo animato.

## Personalizzazione

```powershell
python main.py --durata 20 --passo 0.005 --angolo 90 --lunghezza 1.5 --smorzamento 0.1
python main.py --solo-grafico --cartella risultati
python -m unittest discover -s tests -v
```

L'angolo iniziale è espresso in gradi; il passo temporale e la durata in secondi,
la lunghezza in metri. Diminuendo il passo si aumenta la precisione e il tempo
di calcolo. Con attrito, l'energia meccanica diminuisce: il grafico mostra
questa variazione e anche l'eventuale errore numerico dei metodi.
