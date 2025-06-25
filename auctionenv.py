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
                 rialzo_minimo: int = 1,
                 max_offerta_iniziale: int = 50):
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
            max_offerta_iniziale: Massima offerta iniziale consentita
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
        self.max_offerta_iniziale = max_offerta_iniziale
        
        # Dataframe con tutti i giocatori
        self.giocatori_df = giocatori_df.reset_index(drop=True)
        self.max_giocatori = len(giocatori_df)

        # Calcola statistiche per normalizzazione dei reward
        self.valore_medio = giocatori_df['Valore'].mean()
        self.valore_std = giocatori_df['Valore'].std()
        
        # Spazio delle azioni: 
        # - Se è il turno di chiamare: [giocatore_idx] (0 a num_giocatori-1)
        # - Se è il turno di rilancio: [offerta] (0 = passa, >0 = rilancia)
        self.action_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([self.max_giocatori, budget_iniziale]), 
            dtype=np.int32
        )
        
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
        obs_shape = self._calculate_obs_shape()
        self.observation_space = spaces.Box(
            low=-1, 
            high=max(budget_iniziale, self.max_giocatori), 
            shape=(obs_shape,), 
            dtype=np.float32
        )
        
        # Dizionario dei ruoli
        self.ruoli_dict = {"P": n_portieri, "D": n_difensori, 
                          "C": n_centrocampisti, "A": n_attaccanti}
        
        # Stati dell'ambiente
        self.reset()

    def _calculate_obs_shape(self):
        """Calcola la dimensione dello spazio delle osservazioni."""
        base_obs = 7  # budget, fase, giocatore_in_asta, offerta, agente_offerta, turno, priorità
        ruoli_obs = 8  # 4 ruoli rosa corrente + 4 ruoli disponibili
        altri_agenti = (self.n_agenti - 1) * 5  # budget + 4 ruoli per ogni altro agente
        giocatore_features = 3  # valore, ruolo_encoded, desiderabilità del giocatore in asta
        return base_obs + ruoli_obs + altri_agenti + giocatore_features
        
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
        
        # Statistiche per i reward
        self.valore_totale_rose = [0.0] * self.n_agenti
        self.storia_acquisti = []  # Per analisi post-asta

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
        giocatore_idx = int(action[0])
        offerta = int(action[1])
        reward = 0
        done = False
        info = {"valid_action": True, "fase": self.fase}
        
        reward = 0
        done = False
        info = {"valid_action": True, "fase": self.fase}
        
        if self.fase == "chiamata":
            # L'agente deve chiamare un giocatore con offerta iniziale
            if self._is_valid_chiamata(giocatore_idx, offerta):
                reward = self._esegui_chiamata(giocatore_idx, offerta)
                self.ultima_chiamata_agente = self.agente_di_turno
                info["azione"] = f"Chiamata {self.giocatori_df.loc[giocatore_idx, 'Nome']} per {offerta}"
            else:
                reward = -10
                info["valid_action"] = False
                info["errore"] = "Chiamata non valida"
                
        elif self.fase == "rilancio":
            # L'agente deve decidere se rilanciare o passare
            if self._is_valid_rilancio(offerta):
                if offerta == 0:  # Passa
                    reward = self._esegui_passa()
                    info["azione"] = "Passa"
                else:  # Rilancia
                    reward = self._esegui_rilancio(offerta)
                    info["azione"] = f"Rilancia a {offerta}"
            else:
                reward = -10
                info["valid_action"] = False
                info["errore"] = "Rilancio non valido"
        
        # Controlla se l'asta è terminata
        done = self._is_asta_terminata()
        
        # Reward finale se l'asta è terminata
        if done:
            reward += self._calculate_final_reward()
            info["final_scores"] = self._get_final_scores()
        
        self.turno_globale += 1
        
        return self._get_observation(), reward, done, info
    
    def _is_valid_chiamata(self, giocatore_idx, offerta_iniziale):
        """Controlla se la chiamata è valida."""
        # Controlli base
        if giocatore_idx < 0 or giocatore_idx >= len(self.giocatori_df):
            return False
        if giocatore_idx not in self.giocatori_disponibili:
            return False
            
        # Controlli sull'offerta
        if offerta_iniziale < 1:
            return False
        if offerta_iniziale > self.budgets[self.agente_di_turno]:
            return False
        if offerta_iniziale > self.max_offerta_iniziale:
            return False
            
        # Controlli sul ruolo
        ruolo = self.giocatori_df.loc[giocatore_idx, "Ruolo"]
        if len(self.roster_ruoli[self.agente_di_turno][ruolo]) >= self.ruoli_dict[ruolo]:
            return False
            
        return True
    
    def _is_valid_rilancio(self, offerta):
        """Controlla se il rilancio è valido."""
        if offerta == 0:  # Passa sempre valido
            return True
            
        # Controlli sull'offerta
        if offerta < self.offerta_corrente + self.rialzo_minimo:
            return False
        if offerta > self.budgets[self.agente_rilancio]:
            return False
            
        # Controlli sul ruolo
        if self.giocatore_in_asta >= 0:
            ruolo = self.giocatori_df.loc[self.giocatore_in_asta, "Ruolo"]
            if len(self.roster_ruoli[self.agente_rilancio][ruolo]) >= self.ruoli_dict[ruolo]:
                return False
            
        return True
    
    def _esegui_chiamata(self, giocatore_idx, offerta_iniziale):
        """Esegue la chiamata di un giocatore con offerta iniziale."""
        self.giocatore_in_asta = giocatore_idx
        self.offerta_corrente = offerta_iniziale
        self.agente_offerta = self.agente_di_turno
        
        # Calcola reward strategico per la chiamata
        giocatore = self.giocatori_df.loc[giocatore_idx]
        valore_giocatore = giocatore['Valore']
        
        # Reward basato su:
        # 1. Rapporto valore/prezzo
        # 2. Urgenza del ruolo
        # 3. Aggressività dell'offerta
        
        rapporto_valore = valore_giocatore / offerta_iniziale
        urgenza_ruolo = self._calculate_role_urgency(giocatore['Ruolo'], self.agente_di_turno)
        aggressivita = min(offerta_iniziale / valore_giocatore, 2.0)  # Cap a 2x il valore
        
        reward = rapporto_valore * 2 + urgenza_ruolo * 3 - aggressivita * 1
        
        # Passa alla fase di rilancio
        self.fase = "rilancio"
        self.agente_rilancio = (self.agente_di_turno + 1) % self.n_agenti
        
        return reward
    
    def _esegui_passa(self):
        """Esegue il passaggio durante la fase di rilancio."""
        # Passa al prossimo agente
        self.agente_rilancio = (self.agente_rilancio + 1) % self.n_agenti
        
        # Se siamo tornati all'agente che ha l'offerta corrente, tutti hanno passato
        if self.agente_rilancio == self.agente_offerta:
            return self._assegna_giocatore()
        
        return 0  # Reward neutrale per il passaggio
    
    def _esegui_rilancio(self, offerta):
        """Esegue un rilancio."""
        old_offerta = self.offerta_corrente
        self.offerta_corrente = offerta
        self.agente_offerta = self.agente_rilancio
        
        # Calcola reward per il rilancio
        giocatore = self.giocatori_df.loc[self.giocatore_in_asta]
        valore_giocatore = giocatore['Valore']
        
        # Reward negativo se si rilancia troppo sopra il valore
        if offerta > valore_giocatore * 1.2:
            reward = -2
        else:
            reward = 1  # Reward positivo per rilancio ragionevole
        
        # Passa al prossimo agente
        self.agente_rilancio = (self.agente_rilancio + 1) % self.n_agenti
        
        # Se siamo tornati all'agente che ha rilanciato, tutti hanno passato
        if self.agente_rilancio == self.agente_offerta:
            reward += self._assegna_giocatore()
            
        return reward
    
    def _assegna_giocatore(self):
        """Assegna il giocatore all'agente che ha fatto l'offerta più alta."""
        # Aggiorna budget
        self.budgets[self.agente_offerta] -= self.offerta_corrente
        
        # Aggiungi giocatore alla rosa
        self.roster[self.agente_offerta].append(self.giocatore_in_asta)
        
        # Aggiorna il conteggio per ruolo
        giocatore = self.giocatori_df.loc[self.giocatore_in_asta]
        ruolo = giocatore["Ruolo"]
        self.roster_ruoli[self.agente_offerta][ruolo].append(self.giocatore_in_asta)
        
        # Aggiorna valore totale della rosa
        self.valore_totale_rose[self.agente_offerta] += giocatore['Valore']
        
        # Salva nella storia
        self.storia_acquisti.append({
            'agente': self.agente_offerta,
            'giocatore': self.giocatore_in_asta,
            'nome': giocatore['Nome'],
            'ruolo': ruolo,
            'valore': giocatore['Valore'],
            'prezzo_pagato': self.offerta_corrente,
            'turno': self.turno_globale
        })
        
        # Calcola reward per l'acquisto
        affare_reward = max(0, giocatore['Valore'] - self.offerta_corrente) / 10
        
        # Rimuovi il giocatore dai disponibili
        self.giocatori_disponibili.remove(self.giocatore_in_asta)
        
        # Torna alla fase di chiamata con il prossimo agente
        self.fase = "chiamata"
        self.agente_di_turno = (self.agente_di_turno + 1) % self.n_agenti
        self.giocatore_in_asta = -1
        self.offerta_corrente = 0
        self.agente_offerta = -1
        
        return affare_reward
    
    def _calculate_role_urgency(self, ruolo, agente):
        """Calcola l'urgenza di acquistare un giocatore di un certo ruolo."""
        posseduti = len(self.roster_ruoli[agente][ruolo])
        richiesti = self.ruoli_dict[ruolo]
        disponibili = sum(1 for i in self.giocatori_disponibili 
                         if self.giocatori_df.loc[i, "Ruolo"] == ruolo)
        
        if posseduti >= richiesti:
            return -5  # Non serve
        
        # Urgenza cresce se ho pochi slot rimasti rispetto ai giocatori disponibili
        slots_rimanenti = richiesti - posseduti
        if disponibili <= slots_rimanenti:
            return 5  # Molto urgente
        elif disponibili <= slots_rimanenti * 2:
            return 3  # Abbastanza urgente
        else:
            return 1  # Poco urgente
    
    def _calculate_final_reward(self):
        """Calcola il reward finale basato sulla qualità della rosa assemblata."""
        agente_corrente = self.agente_di_turno if self.fase == "chiamata" else self.agente_rilancio
        
        # Valore totale della rosa
        valore_rosa = self.valore_totale_rose[agente_corrente]
        
        # Penalità per rosa incompleta
        penalita_completezza = 0
        for ruolo, richiesti in self.ruoli_dict.items():
            posseduti = len(self.roster_ruoli[agente_corrente][ruolo])
            if posseduti < richiesti:
                penalita_completezza -= (richiesti - posseduti) * 20
        
        # Budget rimanente (bonus se ben gestito)
        budget_rimasto = self.budgets[agente_corrente]
        bonus_budget = min(budget_rimasto / 10, 5)  # Massimo 5 punti
        
        return (valore_rosa / 10) + penalita_completezza + bonus_budget
    
    def _get_final_scores(self):
        """Ritorna i punteggi finali di tutti gli agenti."""
        scores = {}
        for agente in range(self.n_agenti):
            valore_rosa = sum(self.giocatori_df.loc[i, 'Valore'] for i in self.roster[agente])
            budget_speso = self.budget_iniziale - self.budgets[agente]
            efficienza = valore_rosa / max(budget_speso, 1)
            
            scores[f'agente_{agente}'] = {
                'valore_rosa': valore_rosa,
                'budget_speso': budget_speso,
                'budget_rimasto': self.budgets[agente],
                'efficienza': efficienza,
                'rosa_completa': self._is_rosa_completa(agente)
            }
        
        return scores
    
    def _is_rosa_completa(self, agente):
        """Controlla se la rosa dell'agente è completa."""
        for ruolo, richiesti in self.ruoli_dict.items():
            if len(self.roster_ruoli[agente][ruolo]) < richiesti:
                return False
        return True
    
    def _is_asta_terminata(self):
        """Controlla se l'asta è terminata."""
        if len(self.giocatori_disponibili) == 0:
            return True
            
        if all(budget < 1 for budget in self.budgets):
            return True
            
        # Controlla se tutti possono completare le rose
        for agente in range(self.n_agenti):
            if not self._can_complete_roster(agente):
                continue
            return False
        
        return True
    
    def _can_complete_roster(self, agente):
        """Controlla se l'agente può ancora completare la sua rosa."""
        if self.budgets[agente] < 1:
            return False
            
        for ruolo, richiesti in self.ruoli_dict.items():
            posseduti = len(self.roster_ruoli[agente][ruolo])
            if posseduti < richiesti:
                # Controlla se ci sono giocatori disponibili per questo ruolo
                giocatori_ruolo = [i for i in self.giocatori_disponibili 
                                 if self.giocatori_df.loc[i, "Ruolo"] == ruolo]
                if giocatori_ruolo:
                    return True
        
        return False
    
    def _get_observation(self):
        """Ritorna l'osservazione corrente dell'ambiente."""
        obs = []
        
        # Agente corrente
        agente_corrente = self.agente_di_turno if self.fase == "chiamata" else self.agente_rilancio
        
        # Informazioni base (7 elementi)
        obs.extend([
            self.budgets[agente_corrente] / self.budget_iniziale,  # Budget normalizzato
            0 if self.fase == "chiamata" else 1,  # Fase
            self.giocatore_in_asta,  # Giocatore in asta
            self.offerta_corrente / self.budget_iniziale,  # Offerta normalizzata
            self.agente_offerta,  # Chi ha fatto l'offerta
            self.turno_globale / 100,  # Turno normalizzato
            1 if agente_corrente == self.ultima_chiamata_agente else 0  # Priorità turno
        ])
        
        # Rosa dell'agente corrente (4 elementi)
        for ruolo in ["P", "D", "C", "A"]:
            count = len(self.roster_ruoli[agente_corrente][ruolo])
            required = self.ruoli_dict[ruolo]
            obs.append(count / required)
        
        # Giocatori disponibili per ruolo (4 elementi)
        total_available = len(self.giocatori_disponibili)
        for ruolo in ["P", "D", "C", "A"]:
            count = sum(1 for i in self.giocatori_disponibili 
                       if self.giocatori_df.loc[i, "Ruolo"] == ruolo)
            obs.append(count / max(total_available, 1))
        
        # Informazioni sugli altri agenti
        for agente in range(self.n_agenti):
            if agente != agente_corrente:
                # Budget dell'agente
                obs.append(self.budgets[agente] / self.budget_iniziale)
                
                # Rosa per ruolo
                for ruolo in ["P", "D", "C", "A"]:
                    count = len(self.roster_ruoli[agente][ruolo])
                    required = self.ruoli_dict[ruolo]
                    obs.append(count / required)
        
        # Informazioni sul giocatore in asta (3 elementi)
        if self.giocatore_in_asta >= 0:
            giocatore = self.giocatori_df.loc[self.giocatore_in_asta]
            obs.extend([
                giocatore['Valore'] / 100,  # Valore normalizzato
                ["P", "D", "C", "A"].index(giocatore["Ruolo"]) / 3,  # Ruolo codificato
                self._calculate_role_urgency(giocatore["Ruolo"], agente_corrente) / 5  # Urgenza
            ])
        else:
            obs.extend([0, 0, 0])
        
        return np.array(obs, dtype=np.float32)
    
    def get_valid_actions(self):
        """Ritorna le azioni valide per l'agente corrente."""
        if self.fase == "chiamata":
            valid_actions = []
            for giocatore_idx in self.giocatori_disponibili:
                # Per ogni giocatore, trova le offerte valide
                max_offerta = min(
                    self.budgets[self.agente_di_turno],
                    self.max_offerta_iniziale
                )
                for offerta in range(1, max_offerta + 1):
                    if self._is_valid_chiamata(giocatore_idx, offerta):
                        valid_actions.append([giocatore_idx, offerta])
            return valid_actions
        else:
            # Fase rilancio
            valid_actions = [[0, 0]]  # Può sempre passare
            
            if self.giocatore_in_asta >= 0:
                min_offerta = self.offerta_corrente + self.rialzo_minimo
                max_offerta = self.budgets[self.agente_rilancio]
                
                for offerta in range(min_offerta, max_offerta + 1):
                    if self._is_valid_rilancio(offerta):
                        valid_actions.append([0, offerta])
            
            return valid_actions
    
    def render(self, mode="human"):
        """Visualizza lo stato corrente dell'asta."""
        if mode != "human":
            return
            
        print(f"\n=== TURNO {self.turno_globale} - FASE: {self.fase.upper()} ===")
        
        if self.fase == "chiamata":
            print(f"🎯 Agente {self.agente_di_turno} deve chiamare un giocatore")
        else:
            print(f"💰 Agente {self.agente_rilancio} deve decidere se rilanciare")
            if self.giocatore_in_asta >= 0:
                giocatore = self.giocatori_df.loc[self.giocatore_in_asta]
                print(f"⚽ Giocatore in asta: {giocatore['Nome']} ({giocatore['Ruolo']})")
                print(f"💎 Valore stimato: {giocatore['Valore']} | Offerta: {self.offerta_corrente} crediti")
                print(f"🏆 Offerente: Agente {self.agente_offerta}")
        
        print(f"\n📊 Giocatori disponibili: {len(self.giocatori_disponibili)}")
        
        print(f"\n{'='*60}")
        print("STATO DEGLI AGENTI:")
        for i in range(self.n_agenti):
            marker = ">>> " if (i == self.agente_di_turno and self.fase == "chiamata") or \
                              (i == self.agente_rilancio and self.fase == "rilancio") else "    "
            
            valore_rosa = self.valore_totale_rose[i]
            budget_speso = self.budget_iniziale - self.budgets[i]
            efficienza = valore_rosa / max(budget_speso, 1)
            
            print(f"{marker}Agente {i}: 💰{self.budgets[i]} crediti | Rosa: {valore_rosa:.1f} | Efficienza: {efficienza:.2f}")
            
            for ruolo in ["P", "D", "C", "A"]:
                count = len(self.roster_ruoli[i][ruolo])
                required = self.ruoli_dict[ruolo]
                status = "✅" if count >= required else "❌"
                print(f"      {status} {ruolo}: {count}/{required}", end="")
                
                if self.roster_ruoli[i][ruolo]:
                    nomi = [self.giocatori_df.loc[idx, 'Nome'][:10] for idx in self.roster_ruoli[i][ruolo][:3]]
                    if len(self.roster_ruoli[i][ruolo]) > 3:
                        nomi.append("...")
                    print(f" ({', '.join(nomi)})")
                else:
                    print()
        
        print("=" * 60)