import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import signal
import sys

from fantacalcioAgent import FantacalcioDQNAgent

class FantacalcioTrainer:
    """
    Classe per gestire il training dell'agente DQN per il Fantacalcio.
    
    Fornisce:
    - Training loop con logging avanzato
    - Salvataggio automatico e checkpoint
    - Validazione periodica
    - Early stopping
    - Visualizzazioni real-time
    - Gestione interruzioni
    """
    
    def __init__(self, 
                 env,
                 agent: FantacalcioDQNAgent,
                 n_episodes: int = 2000,
                 validation_episodes: int = 10,
                 validation_freq: int = 100,
                 save_freq: int = 100,
                 early_stopping_patience: int = 200,
                 target_reward: float = None,
                 log_dir: str = "logs",
                 checkpoint_dir: str = "checkpoints"):
        """
        Inizializza il trainer.
        
        Args:
            env: Environment del Fantacalcio
            agent: Agente DQN da addestrare
            n_episodes: Numero totale di episodi
            validation_episodes: Episodi per validazione
            validation_freq: Frequenza validazione (ogni N episodi)
            save_freq: Frequenza salvataggio (ogni N episodi)
            early_stopping_patience: Episodi senza miglioramento per early stopping
            target_reward: Reward target per terminazione anticipata
            log_dir: Directory per i log
            checkpoint_dir: Directory per i checkpoint
        """
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.validation_episodes = validation_episodes
        self.validation_freq = validation_freq
        self.save_freq = save_freq
        self.early_stopping_patience = early_stopping_patience
        self.target_reward = target_reward
        
        # Directory setup
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metriche di training
        self.episode_rewards = []
        self.episode_lengths = []
        self.validation_rewards = []
        self.training_losses = []
        self.episode_times = []
        
        # Early stopping
        self.best_validation_reward = float('-inf')
        self.episodes_without_improvement = 0
        self.stopped_early = False
        
        # Stato del training
        self.start_time = None
        self.current_episode = 0
        self.training_interrupted = False
        
        # Callback per eventi personalizzati
        self.callbacks = {
            'episode_end': [],
            'validation_end': [],
            'training_end': []
        }
        
        # Setup signal handler per interruzioni
        self._setup_signal_handlers()
        
        print(f"🏋️ Trainer inizializzato:")
        print(f"   Episodi: {n_episodes}")
        print(f"   Validazione: ogni {validation_freq} episodi")
        print(f"   Salvataggio: ogni {save_freq} episodi")
        print(f"   Early stopping: {early_stopping_patience} episodi")
        
    def _setup_signal_handlers(self):
        """Setup gestori per interruzioni (Ctrl+C)."""
        def signal_handler(signum, frame):
            print(f"\n⚠️ Interruzione ricevuta (signal {signum})")
            self.training_interrupted = True
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def add_callback(self, event: str, callback: Callable):
        """
        Aggiunge callback per eventi del training.
        
        Args:
            event: 'episode_end', 'validation_end', 'training_end'
            callback: Funzione da chiamare
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            raise ValueError(f"Evento '{event}' non riconosciuto")
    
    def train(self, verbose: bool = True, plot_realtime: bool = False):
        """
        Esegue il training completo dell'agente.
        
        Args:
            verbose: Se stampare output dettagliato
            plot_realtime: Se mostrare grafici in tempo reale (richiede GUI)
            
        Returns:
            training_history: Dizionario con tutta la storia del training
        """
        self.start_time = time.time()
        
        if verbose:
            print(f"\n🚀 AVVIO TRAINING")
            print(f"Episodi: {self.n_episodes}")
            print(f"Environment: {self.env.__class__.__name__}")
            print(f"Agente: {self.agent.__class__.__name__}")
            print("-" * 60)
        
        try:
            for episode in range(self.n_episodes):
                if self.training_interrupted:
                    print(f"\n⏹️ Training interrotto dall'utente all'episodio {episode}")
                    break
                
                self.current_episode = episode
                
                # Esegui episodio di training
                episode_start_time = time.time()
                episode_reward, episode_length = self._run_training_episode()
                episode_time = time.time() - episode_start_time
                
                # Salva metriche
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_times.append(episode_time)
                self.agent.episode_rewards.append(episode_reward)
                
                # Logging periodico
                if verbose and episode % 50 == 0:
                    self._print_progress(episode)
                
                # Validazione periodica
                if episode % self.validation_freq == 0 and episode > 0:
                    self._run_validation(episode, verbose)
                
                # Salvataggio periodico
                if episode % self.save_freq == 0 and episode > 0:
                    self._save_checkpoint(episode)
                
                # Check early stopping
                if self._check_early_stopping():
                    if verbose:
                        print(f"\n🛑 Early stopping attivato all'episodio {episode}")
                    self.stopped_early = True
                    break
                
                # Check target reward
                if self.target_reward and episode_reward >= self.target_reward:
                    if verbose:
                        print(f"\n🎯 Target reward raggiunto all'episodio {episode}!")
                    break
                
                # Callback fine episodio
                for callback in self.callbacks['episode_end']:
                    callback(episode, episode_reward, episode_length)
                
                # Plot real-time (opzionale)
                if plot_realtime and episode % 100 == 0:
                    self._plot_realtime_progress()
        
        except Exception as e:
            print(f"\n❌ Errore durante il training: {e}")
            raise
        
        finally:
            # Cleanup e salvataggio finale
            self._finalize_training(verbose)
        
        # Prepara history
        training_history = self._get_training_history()
        
        # Callback fine training
        for callback in self.callbacks['training_end']:
            callback(training_history)
        
        return training_history
    
    def _run_training_episode(self):
        """Esegue un singolo episodio di training."""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Determina fase
            fase_chiamata = (self.env.fase == "chiamata")
            
            # Ottieni azioni valide
            valid_actions = self.env.get_valid_actions()
            if not valid_actions:
                break
            
            # Scegli azione
            action = self.agent.act(state, valid_actions, fase_chiamata, training=True)
            
            # Esegui azione
            next_state, reward, done, info = self.env.step(action)
            
            # Salva esperienza
            next_valid_actions = self.env.get_valid_actions() if not done else []
            self.agent.remember(state, action, reward, next_state, done, next_valid_actions)
            
            # Training dell'agente
            if len(self.agent.memory) > self.agent.batch_size:
                loss = self.agent.replay()
                if loss is not None:
                    self.training_losses.append(loss)
            
            total_reward += reward
            steps += 1
            state = next_state
            self.agent.steps_done += 1
            
            if done:
                break
        
        return total_reward, steps
    
    def _run_validation(self, episode: int, verbose: bool = True):
        """Esegue validazione dell'agente."""
        validation_rewards = []
        
        for val_ep in range(self.validation_episodes):
            val_reward, _ = self._run_validation_episode()
            validation_rewards.append(val_reward)
        
        avg_validation_reward = np.mean(validation_rewards)
        self.validation_rewards.append(avg_validation_reward)
        
        # Check per early stopping
        if avg_validation_reward > self.best_validation_reward:
            self.best_validation_reward = avg_validation_reward
            self.episodes_without_improvement = 0
            
            # Salva miglior modello
            best_model_path = self.checkpoint_dir / "best_model.pth"
            self.agent.save_model(str(best_model_path))
        else:
            self.episodes_without_improvement += 1
        
        if verbose:
            print(f"\n📊 Validazione episodio {episode}:")
            print(f"   Reward medio: {avg_validation_reward:.2f} ± {np.std(validation_rewards):.2f}")
            print(f"   Miglior reward: {self.best_validation_reward:.2f}")
            print(f"   Episodi senza miglioramento: {self.episodes_without_improvement}")
        
        # Callback validazione
        for callback in self.callbacks['validation_end']:
            callback(episode, avg_validation_reward, validation_rewards)
    
    def _run_validation_episode(self):
        """Esegue un singolo episodio di validazione (senza training)."""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            fase_chiamata = (self.env.fase == "chiamata")
            valid_actions = self.env.get_valid_actions()
            
            if not valid_actions:
                break
            
            # Azione senza training (no exploration)
            action = self.agent.act(state, valid_actions, fase_chiamata, training=False)
            next_state, reward, done, info = self.env.step(action)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        return total_reward, steps
    
    def _check_early_stopping(self):
        """Controlla se attivare early stopping."""
        if self.early_stopping_patience <= 0:
            return False
        
        return self.episodes_without_improvement >= self.early_stopping_patience
    
    def _save_checkpoint(self, episode: int):
        """Salva checkpoint del training."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_episode_{episode}.pth"
        
        additional_info = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'validation_rewards': self.validation_rewards,
            'training_losses': self.training_losses,
            'best_validation_reward': self.best_validation_reward,
            'episodes_without_improvement': self.episodes_without_improvement
        }
        
        self.agent.save_training_checkpoint(str(checkpoint_path), episode, additional_info)
    
    def _print_progress(self, episode: int):
        """Stampa progresso del training."""
        if len(self.episode_rewards) == 0:
            return
        
        # Calcola metriche
        recent_rewards = self.episode_rewards[-50:] if len(self.episode_rewards) >= 50 else self.episode_rewards
        avg_reward = np.mean(recent_rewards)
        
        recent_lengths = self.episode_lengths[-50:] if len(self.episode_lengths) >= 50 else self.episode_lengths
        avg_length = np.mean(recent_lengths)
        
        recent_times = self.episode_times[-50:] if len(self.episode_times) >= 50 else self.episode_times
        avg_time = np.mean(recent_times)
        
        epsilon = self.agent.get_epsilon()
        
        # Calcola ETA
        elapsed_time = time.time() - self.start_time
        episodes_per_second = episode / elapsed_time if elapsed_time > 0 else 0
        remaining_episodes = self.n_episodes - episode
        eta_seconds = remaining_episodes / episodes_per_second if episodes_per_second > 0 else 0
        eta_minutes = eta_seconds / 60
        
        print(f"Episodio {episode:4d}/{self.n_episodes} | "
              f"Reward: {avg_reward:6.2f} | "
              f"Length: {avg_length:4.0f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Time: {avg_time:.1f}s | "
              f"ETA: {eta_minutes:.0f}m")
        
        # Mostra info aggiuntive occasionalmente
        if episode % 200 == 0 and episode > 0:
            memory_usage = len(self.agent.memory) / self.agent.memory_size
            loss_avg = np.mean(self.training_losses[-1000:]) if self.training_losses else 0
            
            print(f"         Memory: {memory_usage:.1%} | "
                  f"Loss: {loss_avg:.4f} | "
                  f"Steps: {self.agent.steps_done:,}")
    
    def _plot_realtime_progress(self):
        """Mostra grafici in tempo reale (richiede GUI)."""
        try:
            plt.figure(figsize=(15, 5))
            
            # Reward plot
            plt.subplot(1, 3, 1)
            if len(self.episode_rewards) > 1:
                plt.plot(self.episode_rewards, alpha=0.7)
                if len(self.episode_rewards) > 100:
                    # Media mobile
                    window = min(100, len(self.episode_rewards) // 5)
                    moving_avg = pd.Series(self.episode_rewards).rolling(window=window).mean()
                    plt.plot(moving_avg, color='red', linewidth=2)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            
            # Loss plot
            plt.subplot(1, 3, 2)
            if len(self.training_losses) > 1:
                plt.plot(self.training_losses, alpha=0.7)
                if len(self.training_losses) > 100:
                    window = min(100, len(self.training_losses) // 5)
                    moving_avg = pd.Series(self.training_losses).rolling(window=window).mean()
                    plt.plot(moving_avg, color='red', linewidth=2)
            plt.title('Training Loss')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            
            # Validation plot
            plt.subplot(1, 3, 3)
            if len(self.validation_rewards) > 1:
                validation_episodes = [i * self.validation_freq for i in range(len(self.validation_rewards))]
                plt.plot(validation_episodes, self.validation_rewards, 'o-')
            plt.title('Validation Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Avg Validation Reward')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.pause(0.01)
            plt.show(block=False)
            
        except Exception as e:
            print(f"⚠️ Errore plot real-time: {e}")
    
    def _finalize_training(self, verbose: bool = True):
        """Finalizza il training con salvataggi e cleanup."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # Salva modello finale
        final_model_path = self.checkpoint_dir / "final_model.pth"
        self.agent.save_model(str(final_model_path))
        
        # Salva log completo
        self._save_training_log()
        
        # Crea grafici finali
        self._create_final_plots()
        
        if verbose:
            print(f"\n✅ TRAINING COMPLETATO!")
            print(f"Tempo totale: {total_time/60:.1f} minuti")
            print(f"Episodi completati: {len(self.episode_rewards)}")
            if self.episode_rewards:
                print(f"Reward finale: {self.episode_rewards[-1]:.2f}")
                print(f"Miglior reward: {max(self.episode_rewards):.2f}")
            if self.stopped_early:
                print(f"Training terminato con early stopping")
            print(f"Modello salvato: {final_model_path}")
    
    def _save_training_log(self):
        """Salva log completo del training."""
        log_data = {
            'training_config': {
                'n_episodes': self.n_episodes,
                'validation_episodes': self.validation_episodes,
                'validation_freq': self.validation_freq,
                'save_freq': self.save_freq,
                'early_stopping_patience': self.early_stopping_patience,
                'target_reward': self.target_reward
            },
            'agent_config': self.agent.get_statistics(),
            'training_results': {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'validation_rewards': self.validation_rewards,
                'training_losses': self.training_losses,
                'episode_times': self.episode_times,
                'total_time': time.time() - self.start_time if self.start_time else 0,
                'stopped_early': self.stopped_early,
                'best_validation_reward': self.best_validation_reward,
                'episodes_without_improvement': self.episodes_without_improvement,
                'final_epsilon': self.agent.get_epsilon()
            },
            'final_statistics': self._calculate_final_statistics()
        }
        
        log_path = self.log_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        print(f"📝 Log salvato: {log_path}")
    
    def _calculate_final_statistics(self):
        """Calcola statistiche finali del training."""
        if not self.episode_rewards:
            return {}
        
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)
        
        stats = {
            'reward_statistics': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'median': float(np.median(rewards)),
                'q25': float(np.percentile(rewards, 25)),
                'q75': float(np.percentile(rewards, 75))
            },
            'length_statistics': {
                'mean': float(np.mean(lengths)),
                'std': float(np.std(lengths)),
                'min': float(np.min(lengths)),
                'max': float(np.max(lengths))
            },
            'learning_progress': {
                'initial_100_mean': float(np.mean(rewards[:100])) if len(rewards) >= 100 else float(np.mean(rewards)),
                'final_100_mean': float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards)),
                'improvement': float(np.mean(rewards[-100:]) - np.mean(rewards[:100])) if len(rewards) >= 100 else 0.0,
                'best_100_episodes': float(np.max([np.mean(rewards[i:i+100]) for i in range(len(rewards)-99)])) if len(rewards) >= 100 else float(np.mean(rewards))
            }
        }
        
        if self.validation_rewards:
            validation_rewards = np.array(self.validation_rewards)
            stats['validation_statistics'] = {
                'mean': float(np.mean(validation_rewards)),
                'std': float(np.std(validation_rewards)),
                'min': float(np.min(validation_rewards)),
                'max': float(np.max(validation_rewards)),
                'best': self.best_validation_reward
            }
        
        return stats
    
    def _create_final_plots(self):
        """Crea grafici finali del training."""
        try:
            # Setup stile
            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
            
            fig = plt.figure(figsize=(20, 12))
            
            # 1. Reward per episodio con media mobile
            plt.subplot(2, 3, 1)
            if self.episode_rewards:
                episodes = range(len(self.episode_rewards))
                plt.plot(episodes, self.episode_rewards, alpha=0.6, color='lightblue', linewidth=0.8)
                
                # Media mobile
                if len(self.episode_rewards) > 50:
                    window = min(100, len(self.episode_rewards) // 10)
                    moving_avg = pd.Series(self.episode_rewards).rolling(window=window).mean()
                    plt.plot(episodes, moving_avg, color='darkblue', linewidth=2, label=f'Media mobile ({window})')
                    plt.legend()
                
                plt.title('Reward per Episodio', fontsize=14, fontweight='bold')
                plt.xlabel('Episodio')
                plt.ylabel('Reward')
                plt.grid(True, alpha=0.3)
            
            # 2. Distribuzione reward
            plt.subplot(2, 3, 2)
            if self.episode_rewards:
                plt.hist(self.episode_rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                plt.axvline(np.mean(self.episode_rewards), color='red', linestyle='--', 
                           label=f'Media: {np.mean(self.episode_rewards):.2f}')
                plt.axvline(np.median(self.episode_rewards), color='green', linestyle='--',
                           label=f'Mediana: {np.median(self.episode_rewards):.2f}')
                plt.title('Distribuzione Reward', fontsize=14, fontweight='bold')
                plt.xlabel('Reward')
                plt.ylabel('Frequenza')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # 3. Lunghezza episodi
            plt.subplot(2, 3, 3)
            if self.episode_lengths:
                plt.plot(self.episode_lengths, alpha=0.7, color='green')
                if len(self.episode_lengths) > 50:
                    window = min(100, len(self.episode_lengths) // 10)
                    moving_avg = pd.Series(self.episode_lengths).rolling(window=window).mean()
                    plt.plot(moving_avg, color='darkgreen', linewidth=2, label=f'Media mobile ({window})')
                    plt.legend()
                plt.title('Lunghezza Episodi', fontsize=14, fontweight='bold')
                plt.xlabel('Episodio')
                plt.ylabel('Passi')
                plt.grid(True, alpha=0.3)
            
            # 4. Training loss
            plt.subplot(2, 3, 4)
            if self.training_losses:
                plt.plot(self.training_losses, alpha=0.7, color='orange')
                if len(self.training_losses) > 100:
                    window = min(1000, len(self.training_losses) // 10)
                    moving_avg = pd.Series(self.training_losses).rolling(window=window).mean()
                    plt.plot(moving_avg, color='darkorange', linewidth=2, label=f'Media mobile ({window})')
                    plt.legend()
                plt.title('Training Loss', fontsize=14, fontweight='bold')
                plt.xlabel('Training Step')
                plt.ylabel('Loss')
                plt.yscale('log')  # Log scale per loss
                plt.grid(True, alpha=0.3)
            
            # 5. Validazione (se disponibile)
            plt.subplot(2, 3, 5)
            if self.validation_rewards:
                validation_episodes = [i * self.validation_freq for i in range(len(self.validation_rewards))]
                plt.plot(validation_episodes, self.validation_rewards, 'o-', color='purple', markersize=4)
                plt.axhline(self.best_validation_reward, color='red', linestyle='--',
                           label=f'Best: {self.best_validation_reward:.2f}')
                plt.title('Reward di Validazione', fontsize=14, fontweight='bold')
                plt.xlabel('Episodio')
                plt.ylabel('Avg Reward Validazione')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'Nessuna validazione\neseguita', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes, fontsize=12)
                plt.title('Validazione', fontsize=14, fontweight='bold')
            
            # 6. Epsilon decay
            plt.subplot(2, 3, 6)
            if len(self.episode_rewards) > 0:
                episodes = range(len(self.episode_rewards))
                # Ricostruisci epsilon history
                epsilon_history = []
                for ep in episodes:
                    # Simula steps_done basato su lunghezza media episodi
                    avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 200
                    simulated_steps = ep * avg_length
                    epsilon = self.agent.epsilon_end + (self.agent.epsilon_start - self.agent.epsilon_end) * \
                             np.exp(-1. * simulated_steps / self.agent.epsilon_decay)
                    epsilon_history.append(max(epsilon, self.agent.epsilon_end))
                
                plt.plot(episodes, epsilon_history, color='red', linewidth=2)
                plt.title('Epsilon Decay', fontsize=14, fontweight='bold')
                plt.xlabel('Episodio')
                plt.ylabel('Epsilon')
                plt.grid(True, alpha=0.3)
            
            plt.suptitle(f'Risultati Training Fantacalcio RL Agent\n'
                        f'({len(self.episode_rewards)} episodi completati)', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Salva grafico
            plot_path = self.log_dir / "training_results.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Grafici salvati: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ Errore nella creazione dei grafici: {e}")
    
    def _get_training_history(self):
        """Restituisce la storia completa del training."""
        return {
            'episode_rewards': self.episode_rewards.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'validation_rewards': self.validation_rewards.copy(),
            'training_losses': self.training_losses.copy(),
            'episode_times': self.episode_times.copy(),
            'config': {
                'n_episodes': self.n_episodes,
                'validation_freq': self.validation_freq,
                'save_freq': self.save_freq,
                'early_stopping_patience': self.early_stopping_patience,
                'target_reward': self.target_reward
            },
            'final_stats': self._calculate_final_statistics(),
            'agent_stats': self.agent.get_statistics(),
            'stopped_early': self.stopped_early,
            'training_interrupted': self.training_interrupted
        }
    
    def evaluate(self, n_episodes=10, render=False, verbose=True):
        """
        Valuta l'agente addestrato senza ulteriore training.
        
        Args:
            n_episodes: Numero di episodi di valutazione
            render: Se mostrare il rendering
            verbose: Se stampare output dettagliato
            
        Returns:
            evaluation_results: Risultati della valutazione
        """
        if verbose:
            print(f"\n🎯 VALUTAZIONE AGENTE ({n_episodes} episodi)")
            print("-" * 50)
        
        eval_rewards = []
        eval_lengths = []
        eval_details = []
        
        for episode in range(n_episodes):
            if verbose and render:
                print(f"\n📺 Episodio {episode + 1}/{n_episodes}")
            
            state = self.env.reset()
            total_reward = 0
            steps = 0
            episode_log = []
            
            if render:
                self.env.render()
            
            while True:
                fase_chiamata = (self.env.fase == "chiamata")
                valid_actions = self.env.get_valid_actions()
                
                if not valid_actions:
                    break
                
                # Azione senza training e senza exploration
                action = self.agent.act(state, valid_actions, fase_chiamata, training=False)
                
                # Log dell'azione per analisi
                step_info = {
                    'step': steps,
                    'fase': self.env.fase,
                    'action': action,
                    'valid_actions_count': len(valid_actions)
                }
                
                next_state, reward, done, info = self.env.step(action)
                
                step_info['reward'] = reward
                step_info['info'] = info
                episode_log.append(step_info)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if render:
                    self.env.render()
                    if hasattr(self.env, 'storia_acquisti') and self.env.storia_acquisti:
                        ultimo = self.env.storia_acquisti[-1]
                        if len(episode_log) == 1 or ultimo not in [log.get('acquisto') for log in episode_log[:-1]]:
                            print(f"💰 {ultimo['nome']} -> Agente {ultimo['agente']} per {ultimo['prezzo_pagato']} crediti")
                
                if done:
                    if render and 'final_scores' in info:
                        print(f"\n🏁 RISULTATI FINALI:")
                        for agente, score in info['final_scores'].items():
                            print(f"  {agente}: Valore {score['valore_rosa']:.1f} | "
                                  f"Efficienza {score['efficienza']:.2f} | "
                                  f"Rosa completa: {'✅' if score['rosa_completa'] else '❌'}")
                    break
            
            eval_rewards.append(total_reward)
            eval_lengths.append(steps)
            eval_details.append({
                'episode': episode,
                'reward': total_reward,
                'length': steps,
                'log': episode_log
            })
            
            if verbose:
                print(f"Episodio {episode + 1:2d}: Reward {total_reward:6.2f}, Passi {steps:3d}")
        
        # Calcola statistiche
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        avg_length = np.mean(eval_lengths)
        
        evaluation_results = {
            'episodes': n_episodes,
            'rewards': eval_rewards,
            'lengths': eval_lengths,
            'details': eval_details,
            'statistics': {
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'min_reward': min(eval_rewards),
                'max_reward': max(eval_rewards),
                'avg_length': avg_length,
                'success_rate': sum(1 for r in eval_rewards if r > 0) / len(eval_rewards)
            }
        }
        
        if verbose:
            print(f"\n📊 RISULTATI VALUTAZIONE:")
            print(f"  Reward medio: {avg_reward:.2f} ± {std_reward:.2f}")
            print(f"  Range reward: [{min(eval_rewards):.2f}, {max(eval_rewards):.2f}]")
            print(f"  Lunghezza media: {avg_length:.0f} passi")
            print(f"  Episodi positivi: {sum(1 for r in eval_rewards if r > 0)}/{len(eval_rewards)}")
        
        return evaluation_results
    
    def resume_training(self, checkpoint_path: str, additional_episodes: int = 1000):
        """
        Riprende il training da un checkpoint.
        
        Args:
            checkpoint_path: Percorso del checkpoint
            additional_episodes: Episodi aggiuntivi da eseguire
            
        Returns:
            training_history: Storia del training ripreso
        """
        print(f"🔄 Ripresa training da: {checkpoint_path}")
        
        # Carica checkpoint
        self.agent.load_model(checkpoint_path)
        
        # Aggiorna numero episodi
        old_n_episodes = self.n_episodes
        self.n_episodes = len(self.agent.episode_rewards) + additional_episodes
        
        print(f"📈 Training ripreso:")
        print(f"  Episodi precedenti: {len(self.agent.episode_rewards)}")
        print(f"  Episodi aggiuntivi: {additional_episodes}")
        print(f"  Totale episodi: {self.n_episodes}")
        
        # Continua training
        return self.train(verbose=True)

# Funzioni di utilità per analisi avanzata
def analyze_training_stability(trainer: FantacalcioTrainer, window_size: int = 100):
    """Analizza la stabilità del training."""
    if len(trainer.episode_rewards) < window_size * 2:
        return {"error": "Non abbastanza episodi per l'analisi"}
    
    rewards = np.array(trainer.episode_rewards)
    
    # Calcola varianza in finestre scorrevoli
    variances = []
    means = []
    
    for i in range(len(rewards) - window_size + 1):
        window = rewards[i:i + window_size]
        variances.append(np.var(window))
        means.append(np.mean(window))
    
    # Trend della varianza
    variance_trend = np.polyfit(range(len(variances)), variances, 1)[0]
    mean_trend = np.polyfit(range(len(means)), means, 1)[0]
    
    stability_analysis = {
        'variance_trend': variance_trend,  # Negativo = più stabile nel tempo
        'mean_trend': mean_trend,          # Positivo = miglioramento nel tempo
        'final_variance': variances[-1],
        'initial_variance': variances[0],
        'variance_reduction': variances[0] - variances[-1],
        'is_converging': variance_trend < -0.01 and abs(mean_trend) < 0.1
    }
    
    return stability_analysis

def compare_training_runs(trainer1: FantacalcioTrainer, trainer2: FantacalcioTrainer):
    """Confronta due run di training."""
    print("🔍 Confronto Training Runs")
    print("-" * 40)
    
    # Statistiche di base
    stats1 = trainer1._calculate_final_statistics()
    stats2 = trainer2._calculate_final_statistics()
    
    comparisons = {
        'episodes': [len(trainer1.episode_rewards), len(trainer2.episode_rewards)],
        'final_performance': [
            stats1['reward_statistics']['mean'],
            stats2['reward_statistics']['mean']
        ],
        'stability': [
            stats1['reward_statistics']['std'],
            stats2['reward_statistics']['std']
        ],
        'best_performance': [
            stats1['reward_statistics']['max'],
            stats2['reward_statistics']['max']
        ]
    }
    
    # Chi è migliore?
    winner = 1 if comparisons['final_performance'][0] > comparisons['final_performance'][1] else 2
    
    print(f"🏆 Training {winner} ha performance migliori")
    print(f"Run 1: {comparisons['final_performance'][0]:.2f} ± {comparisons['stability'][0]:.2f}")
    print(f"Run 2: {comparisons['final_performance'][1]:.2f} ± {comparisons['stability'][1]:.2f}")
    
    return comparisons

def test_trainer():
    """Test basic del trainer."""
    print("🧪 Testing FantacalcioTrainer...")
    
    # Mock environment e agent per test
    class MockEnv:
        def __init__(self):
            self.fase = "chiamata"
            self.step_count = 0
            
        def reset(self):
            self.step_count = 0
            return np.random.randn(10)
            
        def step(self, action):
            self.step_count += 1
            reward = np.random.randn()
            done = self.step_count > 50
            return np.random.randn(10), reward, done, {}
            
        def get_valid_actions(self):
            return [[1, 10], [2, 15]] if self.step_count < 50 else []
            
        def render(self):
            pass
    
    class MockAgent:
        def __init__(self):
            self.memory = []
            self.batch_size = 32
            self.steps_done = 0
            self.episode_rewards = []
            
        def act(self, state, valid_actions, fase_chiamata, training):
            return valid_actions[0] if valid_actions else [0, 0]
            
        def remember(self, *args):
            self.memory.append(args)
            
        def replay(self):
            return 0.1 if len(self.memory) > self.batch_size else None
            
        def get_epsilon(self):
            return 0.1
            
        def get_statistics(self):
            return {'test': True}
            
        def save_model(self, path):
            pass
            
        def save_training_checkpoint(self, path, episode, info):
            pass
    
    # Test trainer
    env = MockEnv()
    agent = MockAgent()
    trainer = FantacalcioTrainer(env, agent, n_episodes=10, validation_freq=5)
    
    # Test training breve
    history = trainer.train(verbose=False)
    
    print(f"✅ Training test completato: {len(history['episode_rewards'])} episodi")
    print("🎉 Test trainer superato!")

if __name__ == "__main__":
    test_trainer()