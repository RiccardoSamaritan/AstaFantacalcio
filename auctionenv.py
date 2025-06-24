import numpy as np
import gym
from gym import spaces
import pandas as pd
from typing import Dict, List, Tuple, Optional

class FantacalcioAstaEnv(gym.Env):
    """
    Ambiente che simula un'asta del fantacalcio.
    """
    
    def __init__(self, 
                 giocatori_df: pd.DataFrame,
                 n_agenti: int = 8,
                 budget_iniziale: int = 500,
                 n_portieri: int = 3,
                 n_difensori: int = 8, 
                 n_centrocampisti: int = 8,
                 n_attaccanti: int = 6):
        """
        Inizializza l'ambiente dell'asta del fantacalcio.
        
        Args:
            giocatori_df: DataFrame contenente i dati dei giocatori 
                          (deve contenere almeno 'nome', 'ruolo', 'valore')
            n_agenti: Numero di partecipanti all'asta
            budget_iniziale: Budget iniziale per ogni partecipante
            n_portieri: Numero di portieri da acquistare
            n_difensori: Numero di difensori da acquistare
            n_centrocampisti: Numero di centrocampisti da acquistare
            n_attaccanti: Numero di attaccanti da acquistare
        """
        super().__init__()
        
        # Parametri dell'asta
        self.n_agenti = n_agenti
        self.budget_iniziale = budget_iniziale
        self.n_portieri = n_portieri
        self.n_difensori = n_difensori
        self.n_centrocampisti = n_centrocampisti
        self.n_attaccanti = n_attaccanti
        
        # Dataframe con tutti i giocatori
        self.giocatori_df = giocatori_df
        
        # Spazio delle azioni: [offerta] dove offerta è compresa tra 0 (passa) e il budget rimanente
        self.action_space = spaces.Box(low=0, high=budget_iniziale, shape=(1,), dtype=np.int32)
        
        # Calcola il numero massimo di giocatori per costruire lo spazio di osservazione
        self.max_giocatori = len(giocatori_df)
        
        # Spazio delle osservazioni:
        # - budget_rimanente (1 valore)
        # - giocatore corrente (1 valore per indice, 1 per ruolo, 1 per valore stimato)
        # - giocatori in rosa (4 valori per il conteggio dei ruoli)
        # - offerta corrente e chi l'ha fatta (2 valori)
        # - budgets degli avversari (n_agenti-1 valori)
        # - rose degli avversari (4*(n_agenti-1) valori per i conteggi dei ruoli)
        obs_shape = 1 + 3 + 4 + 2 + (n_agenti-1) + 4*(n_agenti-1)
        self.observation_space = spaces.Box(low=0, high=budget_iniziale, shape=(obs_shape,), dtype=np.float32)
        
        # Variabili di stato dell'ambiente
        self.budgets = None  # Budget rimanente per ogni agente
        self.roster = None   # Rosa di ogni agente (dict di liste di indici giocatori)
        self.giocatori_disponibili = None  # Indici dei giocatori ancora disponibili
        self.giocatore_corrente = None  # Indice del giocatore attualmente in asta
        self.offerta_corrente = None  # Valore dell'offerta corrente
        self.agente_offerta = None  # Chi ha fatto l'offerta attuale
        self.turno_agente = None  # Di chi è il turno di rilanciare
        self.turno = None  # Turno globale della partita
        self.posizioni_per_ruolo = {"P": 0, "D": 0, "C": 0, "A": 0}  # Conteggio posizioni per ruolo
        self.ruoli_dict = {"P": n_portieri, "D": n_difensori, "C": n_centrocampisti, "A": n_attaccanti}
        
    def reset(self):
        """Reimposta l'ambiente all'inizio di una nuova asta."""
        # Resetta budget e rose
        self.budgets = [self.budget_iniziale] * self.n_agenti
        self.roster = {i: [] for i in range(self.n_agenti)}
        self.roster_ruoli = {i: {"P": [], "D": [], "C": [], "A": []} for i in range(self.n_agenti)}
        
        # Resetta i giocatori disponibili (tutti all'inizio)
        self.giocatori_disponibili = list(self.giocatori_df.index)
        np.random.shuffle(self.giocatori_disponibili)
        
        # Inizia con il primo giocatore
        self._next_giocatore()
        
        # Azzera offerta corrente
        self.offerta_corrente = 0
        self.agente_offerta = None
        
        # Sceglie casualmente chi inizia
        self.turno_agente = np.random.randint(0, self.n_agenti)
        self.turno = 0
        
        return self._get_observation()
        
    def step(self, action):
        """
        Esegue un'azione nell'ambiente.
        
        Args:
            action: Offerta fatta dall'agente (0 significa passa)
            
        Returns:
            observation: Stato aggiornato dell'ambiente
            reward: Reward ottenuto dall'azione
            done: Se l'episodio è terminato
            info: Informazioni addizionali
        """
        # Estrai l'offerta dall'azione
        offerta = int(action[0])
        agente_corrente = self.turno_agente
        
        # Controllo se l'azione è valida
        valid_action = self._is_valid_action(offerta, agente_corrente)
        
        reward = 0
        done = False
        info = {"valid_action": valid_action}
        
        if valid_action:
            if offerta > 0:  # L'agente ha fatto un'offerta
                self.offerta_corrente = offerta
                self.agente_offerta = agente_corrente
                
                # Passa al prossimo agente
                self._next_turno()
                reward = 0  # Neutrale per ora, solo offerta
            else:  # L'agente passa
                # Se tutti hanno passato tranne uno, assegna il giocatore
                if self.agente_offerta is not None:
                    # Controlla se questo è l'ultimo agente a passare
                    passi_consecutivi = 0
                    for i in range(self.n_agenti):
                        next_agent = (agente_corrente + i) % self.n_agenti
                        if next_agent != self.agente_offerta:
                            passi_consecutivi += 1
                        else:
                            break
                            
                    if passi_consecutivi == self.n_agenti - 1:
                        # Tutti tranne l'offerente hanno passato
                        self._assegna_giocatore(self.agente_offerta, self.offerta_corrente)
                        
                        # Se non ci sono più giocatori, termina
                        if len(self.giocatori_disponibili) == 0:
                            done = True
                        else:
                            # Passa al prossimo giocatore da mettere in asta
                            self._next_giocatore()
                            
                            # Resetta offerta
                            self.offerta_corrente = 0
                            self.agente_offerta = None
                            
                            # L'agente che ha preso l'ultimo giocatore inizia il turno
                            self.turno_agente = self.agente_offerta
                            
                    else:
                        # Passa al prossimo agente
                        self._next_turno()
                else:
                    # Nessuno ha offerto, si passa al prossimo giocatore
                    self._next_giocatore()
                    
        else:
            # Azione non valida, penalità
            reward = -10
            
        # Incrementa turno globale
        self.turno += 1
        
        # Verifica se l'asta è terminata (tutti i ruoli soddisfatti o nessun budget)
        remaining_budgets = sum(1 for b in self.budgets if b > 0)
        if remaining_budgets == 0:
            done = True
            
        # Verifica se un agente ha completato tutti i ruoli
        for agente in range(self.n_agenti):
            all_fulfilled = True
            for ruolo, n_required in self.ruoli_dict.items():
                if len(self.roster_ruoli[agente][ruolo]) < n_required:
                    all_fulfilled = False
                    break
            if all_fulfilled:
                info["roster_completo_agente"] = agente
                # Non terminiamo qui perché vogliamo che tutti completino la rosa

        # Ritorna stato, reward, done e info
        return self._get_observation(), reward, done, info
    
    def _is_valid_action(self, offerta, agente):
        """Controlla se l'azione è valida."""
        # Se offerta è 0, significa che l'agente passa
        if offerta == 0:
            return True
            
        # L'offerta deve essere maggiore dell'offerta corrente
        if offerta <= self.offerta_corrente:
            return False
            
        # L'agente deve avere abbastanza budget
        if offerta > self.budgets[agente]:
            return False
            
        # L'agente deve poter comprare ancora quel ruolo
        ruolo = self.giocatori_df.loc[self.giocatore_corrente, "ruolo"]
        if len(self.roster_ruoli[agente][ruolo]) >= self.ruoli_dict[ruolo]:
            return False
            
        return True
    
    def _next_turno(self):
        """Passa al prossimo agente."""
        self.turno_agente = (self.turno_agente + 1) % self.n_agenti
    
    def _next_giocatore(self):
        """Seleziona il prossimo giocatore da mettere all'asta."""
        if len(self.giocatori_disponibili) > 0:
            self.giocatore_corrente = self.giocatori_disponibili.pop(0)
        else:
            self.giocatore_corrente = None
    
    def _assegna_giocatore(self, agente, offerta):
        """Assegna un giocatore a un agente."""
        # Aggiorna budget
        self.budgets[agente] -= offerta
        
        # Aggiungi giocatore alla rosa
        self.roster[agente].append(self.giocatore_corrente)
        
        # Aggiorna il conteggio per ruolo
        ruolo = self.giocatori_df.loc[self.giocatore_corrente, "ruolo"]
        self.roster_ruoli[agente][ruolo].append(self.giocatore_corrente)
        
    def _get_observation(self):
        """Ritorna l'osservazione corrente dell'ambiente."""
        # Inizializza array di osservazione con zeri
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Budget rimasto dell'agente corrente
        obs[0] = self.budgets[self.turno_agente]
        
        # Informazioni sul giocatore corrente
        if self.giocatore_corrente is not None:
            giocatore = self.giocatori_df.loc[self.giocatore_corrente]
            obs[1] = self.giocatore_corrente  # Indice
            # Codifica one-hot del ruolo (P=0, D=1, C=2, A=3)
            ruolo_idx = {"P": 0, "D": 1, "C": 2, "A": 3}.get(giocatore["ruolo"], 0)
            obs[2] = ruolo_idx
            obs[3] = giocatore["valore"]  # Valore stimato
        
        # Conteggio giocatori per ruolo nell'agente corrente
        for i, ruolo in enumerate(["P", "D", "C", "A"]):
            obs[4 + i] = len(self.roster_ruoli[self.turno_agente][ruolo])
        
        # Offerta corrente e chi l'ha fatta
        obs[8] = self.offerta_corrente
        if self.agente_offerta is not None:
            obs[9] = self.agente_offerta
        
        # Budget degli avversari
        idx = 10
        for agente in range(self.n_agenti):
            if agente != self.turno_agente:
                obs[idx] = self.budgets[agente]
                idx += 1
        
        # Conteggio ruoli degli avversari
        for agente in range(self.n_agenti):
            if agente != self.turno_agente:
                for ruolo in ["P", "D", "C", "A"]:
                    obs[idx] = len(self.roster_ruoli[agente][ruolo])
                    idx += 1
        
        return obs
    
    def render(self, mode="human"):
        """Visualizza lo stato corrente dell'asta."""
        if mode != "human":
            return
            
        print(f"\n=== TURNO {self.turno} ===")
        print(f"Agente di turno: {self.turno_agente}")
        
        if self.giocatore_corrente is not None:
            giocatore = self.giocatori_df.loc[self.giocatore_corrente]
            print(f"Giocatore all'asta: {giocatore['nome']} ({giocatore['ruolo']}) - Valore: {giocatore['valore']}")
        
        print(f"Offerta corrente: {self.offerta_corrente} crediti", end="")
        if self.agente_offerta is not None:
            print(f" (Agente {self.agente_offerta})")
        else:
            print()
            
        print("\nStato degli agenti:")
        for i in range(self.n_agenti):
            print(f"Agente {i}: Budget {self.budgets[i]} crediti")
            for ruolo in ["P", "D", "C", "A"]:
                print(f"  {ruolo}: {len(self.roster_ruoli[i][ruolo])}/{self.ruoli_dict[ruolo]}", end=" ")
                if self.roster_ruoli[i][ruolo]:
                    nomi = [self.giocatori_df.loc[idx, 'nome'] for idx in self.roster_ruoli[i][ruolo]]
                    print(f"({', '.join(nomi)})")
                else:
                    print()
            print()
            
        print(f"Giocatori rimanenti: {len(self.giocatori_disponibili)}")
        print("=" * 50)

# Esempio di utilizzo
if __name__ == "__main__":
    # Crea un DataFrame di esempio con alcuni giocatori
    giocatori_data = {
        'nome': [
            # Portieri
            'Maignan', 'Sommer', 'Di Gregorio', 'Szczesny', 'Provedel', 'Carnesecchi',
            # Difensori
            'Bastoni', 'Bremer', 'Di Lorenzo', 'Dimarco', 'Pavard', 'Calafiori', 'Buongiorno', 'Theo Hernandez',
            # Centrocampisti
            'Barella', 'Calhanoglu', 'Koopmeiners', 'Pulisic', 'Zaccagni', 'Zielinski', 'Pellegrini', 'Chiesa',
            # Attaccanti
            'Lautaro', 'Vlahovic', 'Osimhen', 'Kvaratskhelia', 'Dybala', 'Lukaku', 'Leao', 'Thuram'
        ],
        'ruolo': [
            # Portieri
            'P', 'P', 'P', 'P', 'P', 'P',
            # Difensori
            'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D',
            # Centrocampisti
            'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
            # Attaccanti
            'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'
        ],
        'valore': [
            # Portieri
            30, 20, 18, 15, 15, 13,
            # Difensori
            25, 22, 20, 30, 20, 18, 22, 32,
            # Centrocampisti
            35, 30, 40, 28, 25, 20, 22, 30,
            # Attaccanti
            50, 45, 35, 40, 38, 30, 35, 28,
        ]
    }
    
    giocatori_df = pd.DataFrame(giocatori_data)
    
    # Crea l'ambiente
    env = FantacalcioAstaEnv(giocatori_df, n_agenti=4, budget_iniziale=300)
    
    # Reset dell'ambiente
    observation = env.reset()
    
    # Esempio di qualche step di asta
    for _ in range(5):
        env.render()
        
        # Genera un'azione casuale per il test
        action = np.array([np.random.randint(0, 20)])
        if np.random.random() < 0.3:  # 30% probabilità di passare
            action[0] = 0
            
        observation, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        print(f"Info: {info}")
        
        if done:
            break
    
    env.render()