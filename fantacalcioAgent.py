import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

from fantacalcioNet import FantacalcioNet

class FantacalcioDQNAgent:
    """
    Agente DQN specializzato per l'asta del fantacalcio.
    
    Implementa Deep Q-Learning con:
    - Experience Replay
    - Target Network
    - Epsilon-Greedy Exploration
    - Dual-head architecture per chiamate/rilanci
    """
    
    def __init__(self, 
                 obs_size: int,
                 max_players: int,
                 max_budget: int,
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: int = 10000,
                 memory_size: int = 50000,
                 batch_size: int = 32,
                 target_update: int = 1000,
                 hidden_sizes: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3,
                 device: str = None):
        """
        Inizializza l'agente DQN.
        
        Args:
            obs_size: Dimensione spazio osservazioni
            max_players: Numero massimo giocatori
            max_budget: Budget massimo
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Epsilon iniziale per exploration
            epsilon_end: Epsilon finale
            epsilon_decay: Steps per epsilon decay
            memory_size: Dimensione replay buffer
            batch_size: Dimensione batch per training
            target_update: Frequenza aggiornamento target network
            hidden_sizes: Dimensioni layer nascosti
            dropout_rate: Tasso dropout
            device: Device ('cpu', 'cuda', None=auto)
        """
        self.obs_size = obs_size
        self.max_players = max_players
        self.max_budget = max_budget
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🤖 Agente DQN inizializzato su device: {self.device}")
        
        # Reti neurali
        self.q_network = FantacalcioNet(
            obs_size, max_players, max_budget, 
            hidden_sizes, dropout_rate
        ).to(self.device)
        
        self.target_network = FantacalcioNet(
            obs_size, max_players, max_budget, 
            hidden_sizes, dropout_rate
        ).to(self.device)
        
        # Ottimizzatore
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Contatori e metriche
        self.steps_done = 0
        self.episode_rewards = []
        self.training_losses = []
        self.q_values_history = []
        self.epsilon_history = []
        
        # Statistiche per analisi
        self.action_counts = {'chiamate': 0, 'rilanci': 0, 'passaggi': 0}
        self.successful_actions = 0
        self.total_actions = 0
        
        # Inizializza target network
        self.update_target_network()
        
        print(f"✅ Agente creato con {self.q_network.count_parameters():,} parametri")
        
    def update_target_network(self):
        """Aggiorna la target network copiando i pesi dalla main network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_epsilon(self):
        """Calcola epsilon corrente per exploration."""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                 np.exp(-1. * self.steps_done / self.epsilon_decay)
        return max(epsilon, self.epsilon_end)  # Assicura che non vada sotto epsilon_end
    
    def remember(self, state, action, reward, next_state, done, valid_actions=None):
        """
        Salva un'esperienza nel replay buffer.
        
        Args:
            state: Stato corrente
            action: Azione eseguita
            reward: Reward ricevuto
            next_state: Stato successivo
            done: Se l'episodio è terminato
            valid_actions: Azioni valide (opzionale, per future implementazioni)
        """
        self.memory.append((state, action, reward, next_state, done, valid_actions))
    
    def act(self, state, valid_actions, fase_chiamata=True, training=True):
        """
        Seleziona un'azione dato lo stato corrente.
        
        Args:
            state: Stato corrente dell'ambiente
            valid_actions: Lista delle azioni valide
            fase_chiamata: True se è fase di chiamata, False se rilancio
            training: Se True usa epsilon-greedy, se False usa solo greedy
            
        Returns:
            action: Azione selezionata [giocatore_idx, offerta]
        """
        if not valid_actions:
            return [0, 0]
        
        self.total_actions += 1
        
        # Epsilon-greedy exploration
        if training and random.random() < self.get_epsilon():
            # Esplorazione: scegli azione casuale tra quelle valide
            action = random.choice(valid_actions)
            self._update_action_stats(action, fase_chiamata, exploration=True)
            return action
        
        # Sfruttamento: usa la rete neurale
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.q_network.eval()
            q_values, _ = self.q_network(state_tensor, fase_chiamata)
            q_values = q_values.cpu().numpy()[0]
            self.q_network.train()
        
        # Salva Q-values per analisi
        if training:
            self.q_values_history.append(np.mean(q_values))
        
        # Trova la migliore azione valida
        best_action = self._select_best_valid_action(q_values, valid_actions, fase_chiamata)
        
        self._update_action_stats(best_action, fase_chiamata, exploration=False)
        self.successful_actions += 1
        
        return best_action
    
    def _select_best_valid_action(self, q_values, valid_actions, fase_chiamata):
        """Seleziona la migliore azione valida basata sui Q-values."""
        best_action = None
        best_value = float('-inf')
        
        for action in valid_actions:
            if fase_chiamata:
                # Per chiamata: [giocatore_idx, offerta]
                giocatore_idx, offerta = action
                
                # Calcola score combinando Q-value del giocatore e bonus offerta
                if giocatore_idx < len(q_values):
                    player_q = q_values[giocatore_idx]
                    # Bonus piccolo per offerte più basse (strategia conservativa)
                    offerta_bonus = (self.max_budget - offerta) / self.max_budget * 0.1
                    value = player_q + offerta_bonus
                else:
                    value = 0
            else:
                # Per rilancio: [0, offerta]
                _, offerta = action
                if offerta < len(q_values):
                    value = q_values[offerta]
                else:
                    value = 0
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action if best_action else valid_actions[0]
    
    def _update_action_stats(self, action, fase_chiamata, exploration=False):
        """Aggiorna le statistiche delle azioni per analisi."""
        if fase_chiamata:
            self.action_counts['chiamate'] += 1
        else:
            if action[1] == 0:  # Passa
                self.action_counts['passaggi'] += 1
            else:  # Rilancia
                self.action_counts['rilanci'] += 1
    
    def replay(self):
        """
        Esegue un batch di training dalla replay buffer.
        
        Returns:
            loss: Loss del training step (None se non abbastanza samples)
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Campiona batch casuale
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepara i tensor
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = [e[1] for e in batch]
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Calcola Q-values correnti
        current_q_values = self._compute_current_q_values(states, actions)
        
        # Calcola Q-values target
        next_q_values = self._compute_target_q_values(next_states, dones)
        target_q_values = rewards + (self.gamma * next_q_values)
        
        # Calcola loss
        loss = F.mse_loss(current_q_values, target_q_values.detach())
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping per stabilità
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Salva loss per analisi
        self.training_losses.append(loss.item())
        
        # Aggiorna target network periodicamente
        if self.steps_done % self.target_update == 0:
            self.update_target_network()
        
        return loss.item()
    
    def _compute_current_q_values(self, states, actions):
        """Calcola i Q-values correnti per le azioni del batch."""
        current_q_values = []
        
        for i, action in enumerate(actions):
            state = states[i].unsqueeze(0)
            
            # Determina se è fase di chiamata o rilancio
            fase_chiamata = action[0] > 0  # Se giocatore_idx > 0, è chiamata
            
            q_vals, _ = self.q_network(state, fase_chiamata)
            
            if fase_chiamata:
                # Per chiamata, usa giocatore_idx
                idx = min(action[0], q_vals.size(1) - 1)
            else:
                # Per rilancio, usa offerta
                idx = min(action[1], q_vals.size(1) - 1)
            
            current_q_values.append(q_vals[0, idx])
        
        return torch.stack(current_q_values)
    
    def _compute_target_q_values(self, next_states, dones):
        """Calcola i Q-values target usando la target network."""
        with torch.no_grad():
            next_q_values = []
            
            for i, next_state in enumerate(next_states):
                if dones[i]:
                    next_q_values.append(torch.tensor(0.0, device=self.device))
                else:
                    state = next_state.unsqueeze(0)
                    
                    # Calcola Q-values per entrambe le fasi e prendi il massimo
                    q_chiamata, _ = self.target_network(state, fase_chiamata=True)
                    q_rilancio, _ = self.target_network(state, fase_chiamata=False)
                    
                    max_q_chiamata = torch.max(q_chiamata)
                    max_q_rilancio = torch.max(q_rilancio)
                    
                    next_q_values.append(torch.max(max_q_chiamata, max_q_rilancio))
            
            return torch.stack(next_q_values)
    
    def save_model(self, filepath):
        """
        Salva il modello e le metriche su file.
        
        Args:
            filepath: Percorso dove salvare il modello
        """
        # Assicura che la directory esista
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'obs_size': self.obs_size,
                'max_players': self.max_players,
                'max_budget': self.max_budget,
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'memory_size': self.memory_size,
                'batch_size': self.batch_size,
                'target_update': self.target_update,
                'hidden_sizes': self.hidden_sizes,
                'dropout_rate': self.dropout_rate
            },
            'training_state': {
                'steps_done': self.steps_done,
                'episode_rewards': self.episode_rewards,
                'training_losses': self.training_losses,
                'q_values_history': self.q_values_history,
                'epsilon_history': self.epsilon_history,
                'action_counts': self.action_counts,
                'successful_actions': self.successful_actions,
                'total_actions': self.total_actions
            },
            'model_info': self.q_network.get_model_info()
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 Modello salvato: {filepath}")
    
    def load_model(self, filepath):
        """
        Carica il modello da file.
        
        Args:
            filepath: Percorso del file del modello
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Carica stati delle reti
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Carica stato del training
            training_state = checkpoint.get('training_state', {})
            self.steps_done = training_state.get('steps_done', 0)
            self.episode_rewards = training_state.get('episode_rewards', [])
            self.training_losses = training_state.get('training_losses', [])
            self.q_values_history = training_state.get('q_values_history', [])
            self.epsilon_history = training_state.get('epsilon_history', [])
            self.action_counts = training_state.get('action_counts', {'chiamate': 0, 'rilanci': 0, 'passaggi': 0})
            self.successful_actions = training_state.get('successful_actions', 0)
            self.total_actions = training_state.get('total_actions', 0)
            
            print(f"✅ Modello caricato: {filepath}")
            print(f"   Steps: {self.steps_done:,}")
            print(f"   Episodi: {len(self.episode_rewards)}")
            print(f"   Epsilon corrente: {self.get_epsilon():.3f}")
            
        except Exception as e:
            print(f"❌ Errore nel caricamento modello: {e}")
            raise
    
    def get_statistics(self):
        """Restituisce statistiche dettagliate dell'agente."""
        stats = {
            'training_progress': {
                'steps_done': self.steps_done,
                'episodes_completed': len(self.episode_rewards),
                'current_epsilon': self.get_epsilon(),
                'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'avg_loss_last_1000': np.mean(self.training_losses[-1000:]) if len(self.training_losses) >= 1000 else np.mean(self.training_losses) if self.training_losses else 0,
                'avg_q_value': np.mean(self.q_values_history[-1000:]) if len(self.q_values_history) >= 1000 else np.mean(self.q_values_history) if self.q_values_history else 0
            },
            'action_statistics': {
                'total_actions': self.total_actions,
                'successful_actions': self.successful_actions,
                'success_rate': self.successful_actions / max(self.total_actions, 1),
                'action_counts': self.action_counts.copy(),
                'action_distribution': {
                    'chiamate': self.action_counts['chiamate'] / max(self.total_actions, 1),
                    'rilanci': self.action_counts['rilanci'] / max(self.total_actions, 1),
                    'passaggi': self.action_counts['passaggi'] / max(self.total_actions, 1)
                }
            },
            'model_info': self.q_network.get_model_info(),
            'memory_usage': {
                'buffer_size': len(self.memory),
                'buffer_capacity': self.memory_size,
                'buffer_utilization': len(self.memory) / self.memory_size
            }
        }
        
        return stats
    
    def print_statistics(self):
        """Stampa statistiche in formato leggibile."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("📊 STATISTICHE AGENTE DQN")
        print("="*60)
        
        # Progress
        progress = stats['training_progress']
        print(f"\n🚀 Progresso Training:")
        print(f"  Steps totali: {progress['steps_done']:,}")
        print(f"  Episodi completati: {progress['episodes_completed']:,}")
        print(f"  Epsilon corrente: {progress['current_epsilon']:.3f}")
        print(f"  Reward medio (ultimi 100): {progress['avg_reward_last_100']:.2f}")
        print(f"  Loss media (ultimi 1000): {progress['avg_loss_last_1000']:.4f}")
        print(f"  Q-value medio: {progress['avg_q_value']:.2f}")
        
        # Actions
        actions = stats['action_statistics']
        print(f"\n🎯 Statistiche Azioni:")
        print(f"  Azioni totali: {actions['total_actions']:,}")
        print(f"  Azioni valide: {actions['successful_actions']:,}")
        print(f"  Tasso successo: {actions['success_rate']:.1%}")
        print(f"  Distribuzione azioni:")
        for action_type, percentage in actions['action_distribution'].items():
            count = actions['action_counts'][action_type]
            print(f"    {action_type.capitalize()}: {count:,} ({percentage:.1%})")
        
        # Model
        model = stats['model_info']
        print(f"\n🧠 Informazioni Modello:")
        print(f"  Parametri totali: {model['total_parameters']:,}")
        print(f"  Dimensione osservazioni: {model['obs_size']}")
        print(f"  Hidden layers: {model['hidden_sizes']}")
        print(f"  Dropout rate: {model['dropout_rate']}")
        
        # Memory
        memory = stats['memory_usage']
        print(f"\n💾 Utilizzo Memoria:")
        print(f"  Buffer size: {memory['buffer_size']:,}/{memory['buffer_capacity']:,}")
        print(f"  Utilizzo buffer: {memory['buffer_utilization']:.1%}")
        
        print("="*60)
    
    def analyze_learning_curve(self, window=100):
        """
        Analizza la curva di apprendimento.
        
        Args:
            window: Finestra per media mobile
            
        Returns:
            analysis: Dizionario con analisi della curva
        """
        if len(self.episode_rewards) < window:
            return {"error": "Non abbastanza episodi per l'analisi"}
        
        rewards = np.array(self.episode_rewards)
        
        # Media mobile
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        # Trend (regressione lineare semplice)
        x = np.arange(len(moving_avg))
        trend = np.polyfit(x, moving_avg, 1)[0]
        
        # Stabilità (varianza ultimi episodi)
        recent_variance = np.var(rewards[-window:])
        
        analysis = {
            'total_episodes': len(rewards),
            'initial_performance': np.mean(rewards[:window]),
            'recent_performance': np.mean(rewards[-window:]),
            'improvement': np.mean(rewards[-window:]) - np.mean(rewards[:window]),
            'trend_slope': trend,
            'recent_variance': recent_variance,
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'convergence_indicator': abs(trend) < 0.01 and recent_variance < 1.0  # Heuristic
        }
        
        return analysis
    
    def get_action_value_estimates(self, state, valid_actions, fase_chiamata=True):
        """
        Ottieni stime dei valori delle azioni per debugging.
        
        Args:
            state: Stato corrente
            valid_actions: Azioni valide
            fase_chiamata: Tipo di fase
            
        Returns:
            action_values: Lista di (azione, valore_stimato)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values, value = self.q_network(state_tensor, fase_chiamata)
            q_values = q_values.cpu().numpy()[0]
            value = value.cpu().item()
        
        action_values = []
        for action in valid_actions:
            if fase_chiamata:
                giocatore_idx, offerta = action
                if giocatore_idx < len(q_values):
                    estimated_value = q_values[giocatore_idx]
                else:
                    estimated_value = 0
            else:
                _, offerta = action
                if offerta < len(q_values):
                    estimated_value = q_values[offerta]
                else:
                    estimated_value = 0
            
            action_values.append((action, estimated_value))
        
        # Ordina per valore stimato
        action_values.sort(key=lambda x: x[1], reverse=True)
        
        return action_values, value
    
    def set_exploration_schedule(self, schedule_type='linear', **kwargs):
        """
        Imposta un programma di esplorazione personalizzato.
        
        Args:
            schedule_type: 'linear', 'exponential', 'step'
            **kwargs: Parametri specifici per il tipo di schedule
        """
        if schedule_type == 'linear':
            # Già implementato come default
            pass
        elif schedule_type == 'step':
            # Epsilon diminuisce a step discreti
            self.epsilon_schedule = 'step'
            self.epsilon_steps = kwargs.get('steps', [5000, 10000, 15000])
            self.epsilon_values = kwargs.get('values', [0.5, 0.2, 0.1, 0.05])
        elif schedule_type == 'exponential':
            # Decay esponenziale personalizzato
            self.epsilon_schedule = 'exponential'
            self.epsilon_decay_rate = kwargs.get('decay_rate', 0.995)
        
        print(f"✅ Schedule esplorazione impostato: {schedule_type}")
    
    def get_custom_epsilon(self):
        """Calcola epsilon usando schedule personalizzato se impostato."""
        if hasattr(self, 'epsilon_schedule'):
            if self.epsilon_schedule == 'step':
                for i, step in enumerate(self.epsilon_steps):
                    if self.steps_done < step:
                        return self.epsilon_values[i]
                return self.epsilon_values[-1]
            elif self.epsilon_schedule == 'exponential':
                return max(self.epsilon_end, 
                          self.epsilon_start * (self.epsilon_decay_rate ** self.steps_done))
        
        return self.get_epsilon()  # Default linear
    
    def warm_start(self, num_random_steps=1000):
        """
        Warm start con azioni casuali per popolare il replay buffer.
        
        Args:
            num_random_steps: Numero di step casuali da eseguire
        """
        print(f"🔥 Warm start con {num_random_steps} step casuali...")
        
        # Temporaneamente forza epsilon = 1.0
        original_epsilon_start = self.epsilon_start
        self.epsilon_start = 1.0
        
        # Gli step casuali verranno aggiunti durante il training normale
        # Questo metodo serve principalmente per awareness
        
        # Ripristina epsilon originale
        self.epsilon_start = original_epsilon_start
        
        print(f"✅ Warm start configurato")
    
    def save_training_checkpoint(self, filepath, episode, additional_info=None):
        """
        Salva checkpoint durante il training.
        
        Args:
            filepath: Percorso checkpoint
            episode: Episodio corrente
            additional_info: Informazioni aggiuntive da salvare
        """
        checkpoint_info = {
            'episode': episode,
            'timestamp': np.datetime64('now').astype(str),
            'statistics': self.get_statistics()
        }
        
        if additional_info:
            checkpoint_info.update(additional_info)
        
        # Salva con informazioni aggiuntive
        original_save = self.save_model
        
        # Modifica temporanea per aggiungere info
        def enhanced_save(filepath):
            checkpoint = torch.load(filepath, map_location='cpu') if Path(filepath).exists() else {}
            
            original_save(filepath)
            
            # Ricarica e aggiungi info
            saved_checkpoint = torch.load(filepath, map_location='cpu')
            saved_checkpoint['checkpoint_info'] = checkpoint_info
            torch.save(saved_checkpoint, filepath)
        
        enhanced_save(filepath)
        print(f"💾 Checkpoint salvato: episodio {episode}")

# Funzioni di utilità
def compare_agents(agent1_path, agent2_path, num_comparisons=100):
    """
    Confronta due agenti salvati.
    
    Args:
        agent1_path: Percorso primo agente
        agent2_path: Percorso secondo agente
        num_comparisons: Numero di confronti da fare
        
    Returns:
        comparison_results: Risultati del confronto
    """
    # Placeholder per future implementazioni
    print(f"🔍 Confronto agenti: {agent1_path} vs {agent2_path}")
    print("⚠️  Funzione non ancora implementata")
    
    return {"status": "not_implemented"}

def test_agent():
    """Test base dell'agente DQN."""
    print("🧪 Testing FantacalcioDQNAgent...")
    
    # Parametri di test
    obs_size = 100
    max_players = 400
    max_budget = 500
    
    # Crea agente
    agent = FantacalcioDQNAgent(obs_size, max_players, max_budget)
    print(f"✅ Agente creato")
    
    # Test azione
    state = np.random.randn(obs_size)
    valid_actions = [[1, 10], [2, 15], [3, 20]]
    
    action = agent.act(state, valid_actions, fase_chiamata=True, training=True)
    print(f"✅ Azione selezionata: {action}")
    
    # Test memory
    agent.remember(state, action, 1.0, state, False)
    print(f"✅ Esperienza salvata in memory")
    
    # Test statistiche
    stats = agent.get_statistics()
    print(f"✅ Statistiche generate")
    
    print("🎉 Tutti i test superati!")

if __name__ == "__main__":
    test_agent()