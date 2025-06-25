import numpy as np
import gym
from gym import spaces
import pandas as pd
from typing import Dict, List, Tuple, Optional

class FantacalcioAstaChiamataEnv(gym.Env):
    """
    Ambiente che simula un'asta del fantacalcio a chiamata.
    In questo tipo di asta, ogni agente nel suo turno chiama un giocatore
    e tutti gli altri possono rilanciare.
    """
    
    def __init__(self, 
                 giocatori_df: pd.DataFrame,
                 n_agenti: int = 8,
                 budget_iniziale: int = 500,
                 n_portieri: int = 3,
                 n_difensori: int = 8, 
                 n_centrocampisti: int = 8,
                 n_attaccanti: int = 6,
                 rialzo_minimo: int = 1):
        """
        Inizializza l'ambiente dell'asta del fantacalcio a chiamata.
        
        Args:
            giocatori_df: DataFrame contenente i dati dei giocatori 
                          (deve contenere almeno 'nome', 'ruolo', 'valore')
            n_agenti: Numero di partecipanti all'asta
            budget_iniziale: Budget iniziale per ogni partecipante
            n_portieri: Numero di portieri da acquistare
            n_difensori: Numero di difensori da acquistare
            n_centrocampisti: Numero di centrocampisti da acquistare
            n_attaccanti: Numero di attaccanti da acquistare
            rialzo_minimo: Rialzo minimo per le offerte (default 1)
        """
        super().__init__()
        
        # Parametri dell'asta
        self.n_agenti = n_agenti
        self.budget_iniziale = budget_iniziale
        self.n_portieri = n_portieri
        self.n_difensori = n_difensori
        self.n_centrocampisti = n_centrocampisti
        self.n_attaccanti = n_attaccanti
        self.rialzo_minimo = rialzo_minimo
        
        # Dataframe con tutti i giocatori
        self.giocatori_df = giocatori_df.reset_index(drop=True)
        self.max_giocatori = len(giocatori_df)
        
        # Spazio delle azioni: 
        # - Se è il turno di chiamare: [giocatore_idx] (0 a num_giocatori-1)
        # - Se è il turno di rilancio: [offerta] (0 = passa, >0 = rilancia)
        self.action_space = spaces.Box(low=0, high=max(budget_iniziale, self.max_giocatori), 
                                     shape=(1,), dtype=np.int32)
        
        # Spazio delle osservazioni:
        # - budget dell'agente corrente (1)
        # - fase dell'asta: 0=chiamata, 1=rilancio (1)
        # - giocatore attualmente in asta (-1 se nessuno) (1)
        # - offerta corrente (1)
        # - chi ha fatto l'offerta corrente (-1 se nessuno) (1)
        # - giocatori in rosa dell'agente corrente per ruolo (4)
        # - giocatori disponibili per ruolo (4) 
        # - budgets degli altri agenti (n_agenti-1)
        # - conteggio giocatori per ruolo degli altri agenti (4*(n_agenti-1))
        obs_shape = 1 + 1 + 1 + 1 + 1 + 4 + 4 + (n_agenti-1) + 4*(n_agenti-1)
        self.observation_space = spaces.Box(low=-1, high=budget_iniziale, 
                                          shape=(obs_shape,), dtype=np.float32)
        
        # Dizionario dei ruoli
        self.ruoli_dict = {"P": n_portieri, "D": n_difensori, 
                          "C": n_centrocampisti, "A": n_attaccanti}
        
        # Stati dell'ambiente
        self.reset()
        
    def reset(self):
        """Reimposta l'ambiente all'inizio di una nuova asta."""
        # Reset budget e rose
        self.budgets = [self.budget_iniziale] * self.n_agenti
        self.roster = {i: [] for i in range(self.n_agenti)}
        self.roster_ruoli = {i: {"P": [], "D": [], "C": [], "A": []} for i in range(self.n_agenti)}
        
        # Giocatori disponibili (tutti all'inizio)
        self.giocatori_disponibili = set(range(len(self.giocatori_df)))
        
        # Stati dell'asta
        self.fase = "chiamata"  # "chiamata" o "rilancio"
        self.agente_di_turno = 0  # Chi deve chiamare o chi ha chiamato
        self.agente_rilancio = 0  # Chi deve decidere se rilanciare
        self.giocatore_in_asta = -1  # Indice del giocatore attualmente in asta
        self.offerta_corrente = 0
        self.agente_offerta = -1  # Chi ha fatto l'offerta corrente
        self.turno_globale = 0
        self.giri_senza_rilanci = 0  # Per tracciare quando tutti hanno passato
        
        return self._get_observation()
        
    def step(self, action):
        """
        Esegue un'azione nell'ambiente.
        
        Args:
            action: Array con l'azione (chiamata giocatore o offerta)
            
        Returns:
            observation: Stato aggiornato dell'ambiente
            reward: Reward ottenuto dall'azione
            done: Se l'episodio è terminato
            info: Informazioni addizionali
        """
        azione = int(action[0])
        reward = 0
        done = False
        info = {"valid_action": True, "fase": self.fase}
        
        if self.fase == "chiamata":
            # L'agente deve chiamare un giocatore
            if self._is_valid_chiamata(azione):
                self._esegui_chiamata(azione)
                reward = 0  # Neutrale per la chiamata
            else:
                reward = -10  # Penalità per chiamata non valida
                info["valid_action"] = False
                
        elif self.fase == "rilancio":
            # L'agente deve decidere se rilanciare o passare
            if self._is_valid_rilancio(azione):
                if azione == 0:  # Passa
                    reward = self._esegui_passa()
                else:  # Rilancia
                    reward = self._esegui_rilancio(azione)
            else:
                reward = -10  # Penalità per rilancio non valido
                info["valid_action"] = False
        
        # Controlla se l'asta è terminata
        done = self._is_asta_terminata()
        
        # Aggiorna turno globale
        self.turno_globale += 1
        
        return self._get_observation(), reward, done, info
    
    def _is_valid_chiamata(self, giocatore_idx):
        """Controlla se la chiamata del giocatore è valida."""
        # Il giocatore deve esistere ed essere disponibile
        if giocatore_idx < 0 or giocatore_idx >= len(self.giocatori_df):
            return False
        if giocatore_idx not in self.giocatori_disponibili:
            return False
            
        # L'agente deve poter ancora comprare quel ruolo
        ruolo = self.giocatori_df.loc[giocatore_idx, "ruolo"]
        if len(self.roster_ruoli[self.agente_di_turno][ruolo]) >= self.ruoli_dict[ruolo]:
            return False
            
        # L'agente deve avere almeno 1 credito
        if self.budgets[self.agente_di_turno] < 1:
            return False
            
        return True
    
    def _is_valid_rilancio(self, offerta):
        """Controlla se il rilancio è valido."""
        if offerta == 0:  # Passa sempre valido
            return True
            
        # L'offerta deve essere almeno rialzo_minimo sopra l'offerta corrente
        if offerta < self.offerta_corrente + self.rialzo_minimo:
            return False
            
        # L'agente deve avere abbastanza budget
        if offerta > self.budgets[self.agente_rilancio]:
            return False
            
        # L'agente deve poter ancora comprare quel ruolo
        ruolo = self.giocatori_df.loc[self.giocatore_in_asta, "ruolo"]
        if len(self.roster_ruoli[self.agente_rilancio][ruolo]) >= self.ruoli_dict[ruolo]:
            return False
            
        return True
    
    def _esegui_chiamata(self, giocatore_idx):
        """Esegue la chiamata di un giocatore."""
        self.giocatore_in_asta = giocatore_idx
        self.offerta_corrente = 1  # Offerta base di 1 credito
        self.agente_offerta = self.agente_di_turno
        
        # Passa alla fase di rilancio
        self.fase = "rilancio"
        self.agente_rilancio = (self.agente_di_turno + 1) % self.n_agenti
        self.giri_senza_rilanci = 0
    
    def _esegui_passa(self):
        """Esegue il passaggio durante la fase di rilancio."""
        # Passa al prossimo agente
        self.agente_rilancio = (self.agente_rilancio + 1) % self.n_agenti
        
        # Se siamo tornati all'agente che ha l'offerta corrente, tutti hanno passato
        if self.agente_rilancio == self.agente_offerta:
            self._assegna_giocatore()
            return 0
        
        return 0  # Reward neutrale per il passaggio
    
    def _esegui_rilancio(self, offerta):
        """Esegue un rilancio."""
        self.offerta_corrente = offerta
        self.agente_offerta = self.agente_rilancio
        
        # Passa al prossimo agente
        self.agente_rilancio = (self.agente_rilancio + 1) % self.n_agenti
        
        # Se siamo tornati all'agente che ha rilanciato, tutti hanno passato
        if self.agente_rilancio == self.agente_offerta:
            self._assegna_giocatore()
            
        return 0  # Reward neutrale per il rilancio
    
    def _assegna_giocatore(self):
        """Assegna il giocatore all'agente che ha fatto l'offerta più alta."""
        # Aggiorna budget
        self.budgets[self.agente_offerta] -= self.offerta_corrente
        
        # Aggiungi giocatore alla rosa
        self.roster[self.agente_offerta].append(self.giocatore_in_asta)
        
        # Aggiorna il conteggio per ruolo
        ruolo = self.giocatori_df.loc[self.giocatore_in_asta, "ruolo"]
        self.roster_ruoli[self.agente_offerta][ruolo].append(self.giocatore_in_asta)
        
        # Rimuovi il giocatore dai disponibili
        self.giocatori_disponibili.remove(self.giocatore_in_asta)
        
        # Torna alla fase di chiamata con il prossimo agente
        self.fase = "chiamata"
        self.agente_di_turno = (self.agente_di_turno + 1) % self.n_agenti
        self.giocatore_in_asta = -1
        self.offerta_corrente = 0
        self.agente_offerta = -1
    
    def _is_asta_terminata(self):
        """Controlla se l'asta è terminata."""
        # Nessun giocatore disponibile
        if len(self.giocatori_disponibili) == 0:
            return True
            
        # Nessun agente ha budget sufficiente
        if all(budget < 1 for budget in self.budgets):
            return True
            
        # Tutti gli agenti hanno completato le rose richieste
        for agente in range(self.n_agenti):
            for ruolo, n_required in self.ruoli_dict.items():
                if len(self.roster_ruoli[agente][ruolo]) < n_required:
                    # Controlla se ci sono ancora giocatori di quel ruolo disponibili
                    # e se l'agente ha budget
                    giocatori_ruolo = [i for i in self.giocatori_disponibili 
                                     if self.giocatori_df.loc[i, "ruolo"] == ruolo]
                    if giocatori_ruolo and self.budgets[agente] >= 1:
                        return False
        
        return True
    
    def _get_observation(self):
        """Ritorna l'osservazione corrente dell'ambiente."""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        idx = 0
        
        # Budget dell'agente corrente
        agente_corrente = self.agente_di_turno if self.fase == "chiamata" else self.agente_rilancio
        obs[idx] = self.budgets[agente_corrente]
        idx += 1
        
        # Fase dell'asta (0=chiamata, 1=rilancio)
        obs[idx] = 0 if self.fase == "chiamata" else 1
        idx += 1
        
        # Giocatore in asta (-1 se nessuno)
        obs[idx] = self.giocatore_in_asta
        idx += 1
        
        # Offerta corrente
        obs[idx] = self.offerta_corrente
        idx += 1
        
        # Chi ha fatto l'offerta corrente
        obs[idx] = self.agente_offerta
        idx += 1
        
        # Giocatori in rosa dell'agente corrente per ruolo
        for ruolo in ["P", "D", "C", "A"]:
            obs[idx] = len(self.roster_ruoli[agente_corrente][ruolo])
            idx += 1
        
        # Giocatori disponibili per ruolo
        for ruolo in ["P", "D", "C", "A"]:
            count = sum(1 for i in self.giocatori_disponibili 
                       if self.giocatori_df.loc[i, "ruolo"] == ruolo)
            obs[idx] = count
            idx += 1
        
        # Budget degli altri agenti
        for agente in range(self.n_agenti):
            if agente != agente_corrente:
                obs[idx] = self.budgets[agente]
                idx += 1
        
        # Conteggio giocatori per ruolo degli altri agenti
        for agente in range(self.n_agenti):
            if agente != agente_corrente:
                for ruolo in ["P", "D", "C", "A"]:
                    obs[idx] = len(self.roster_ruoli[agente][ruolo])
                    idx += 1
        
        return obs
    
    def render(self, mode="human"):
        """Visualizza lo stato corrente dell'asta."""
        if mode != "human":
            return
            
        print(f"\n=== TURNO {self.turno_globale} - FASE: {self.fase.upper()} ===")
        
        if self.fase == "chiamata":
            print(f"Agente {self.agente_di_turno} deve chiamare un giocatore")
        else:
            print(f"Agente {self.agente_rilancio} deve decidere se rilanciare")
            if self.giocatore_in_asta >= 0:
                giocatore = self.giocatori_df.loc[self.giocatore_in_asta]
                print(f"Giocatore in asta: {giocatore['nome']} ({giocatore['ruolo']}) - Valore stimato: {giocatore['valore']}")
                print(f"Offerta corrente: {self.offerta_corrente} crediti (Agente {self.agente_offerta})")
        
        print(f"\nGiocatori disponibili: {len(self.giocatori_disponibili)}")
        
        print("\nStato degli agenti:")
        for i in range(self.n_agenti):
            marker = ">>> " if (i == self.agente_di_turno and self.fase == "chiamata") or \
                              (i == self.agente_rilancio and self.fase == "rilancio") else "    "
            print(f"{marker}Agente {i}: Budget {self.budgets[i]} crediti")
            for ruolo in ["P", "D", "C", "A"]:
                count = len(self.roster_ruoli[i][ruolo])
                required = self.ruoli_dict[ruolo]
                print(f"      {ruolo}: {count}/{required}", end=" ")
                if self.roster_ruoli[i][ruolo]:
                    nomi = [self.giocatori_df.loc[idx, 'nome'] for idx in self.roster_ruoli[i][ruolo]]
                    print(f"({', '.join(nomi)})")
                else:
                    print()
        
        print("=" * 60)

    def get_valid_actions(self):
        """Ritorna le azioni valide per l'agente corrente."""
        if self.fase == "chiamata":
            # Ritorna tutti i giocatori che possono essere chiamati
            valid_players = []
            for giocatore_idx in self.giocatori_disponibili:
                if self._is_valid_chiamata(giocatore_idx):
                    valid_players.append(giocatore_idx)
            return valid_players
        else:
            # Fase di rilancio: può sempre passare (0) o fare offerte valide
            valid_offers = [0]  # Può sempre passare
            
            # Aggiunge tutte le offerte possibili dal minimo al budget
            min_offer = self.offerta_corrente + self.rialzo_minimo
            max_offer = self.budgets[self.agente_rilancio]
            
            if min_offer <= max_offer:
                # Controlla se può comprare il ruolo
                ruolo = self.giocatori_df.loc[self.giocatore_in_asta, "ruolo"]
                if len(self.roster_ruoli[self.agente_rilancio][ruolo]) < self.ruoli_dict[ruolo]:
                    valid_offers.extend(range(min_offer, max_offer + 1))
            
            return valid_offers