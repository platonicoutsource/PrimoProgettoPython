import math
import sys
import unittest
from unittest.mock import patch

import main
from main import Config, accelerazione, simula


class TestSimulazionePendolo(unittest.TestCase):
    def test_accelerazione_gravita_richiama_il_pendolo_verso_il_basso(self):
        config = Config(smorzamento=0)
        self.assertLess(accelerazione(math.pi / 2, 0, config), 0)

    def test_rk4_converge_a_un_risultato_di_riferimento(self):
        configurazione_grossolana = Config(durata=2, passo=0.05, smorzamento=0)
        configurazione_precisa = Config(durata=2, passo=0.0005, smorzamento=0)

        grossolano = simula("Runge-Kutta 4", configurazione_grossolana)
        riferimento = simula("Runge-Kutta 4", configurazione_precisa)

        self.assertAlmostEqual(grossolano.angoli[-1], riferimento.angoli[-1], places=4)
        self.assertAlmostEqual(grossolano.velocita[-1], riferimento.velocita[-1], places=4)

    def test_euler_esplicito_mostra_la_deriva_numerica_di_energia(self):
        config = Config(durata=5, passo=0.01, smorzamento=0)
        euler = simula("Euler esplicito", config)
        rk4 = simula("Runge-Kutta 4", config)

        self.assertGreater(euler.energie[-1], euler.energie[0])
        self.assertAlmostEqual(rk4.energie[-1], rk4.energie[0], places=7)

    def test_durata_e_passo_determinano_numero_campioni(self):
        risultato = simula("Runge-Kutta 4", Config(durata=1, passo=0.1))
        self.assertEqual(len(risultato.tempi), 11)
        self.assertAlmostEqual(risultato.tempi[-1], 1)

    def test_rifiuta_metodo_sconosciuto(self):
        with self.assertRaises(ValueError):
            simula("metodo inesistente", Config())

    def test_argomenti_predefinisce_il_pendolo_singolo(self):
        with patch.object(sys, "argv", ["main.py"]):
            args = main.argomenti()
        self.assertEqual(args.modello, "singolo")

    def test_accetta_angoli_ai_limiti_del_range_ammesso(self):
        with patch.object(sys, "argv", ["main.py", "--angolo", "0", "--angolo2", "180"]):
            args = main.argomenti()
        self.assertEqual(args.angolo, 0)
        self.assertEqual(args.angolo2, 180)


if __name__ == "__main__":
    unittest.main()
