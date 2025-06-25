#!/usr/bin/env python3
"""
Main script per il progetto Fantacalcio RL.
Integra tutti i componenti e fornisce un'interfaccia semplice per l'utilizzo.
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

from playerstats import get_player_stats
from auctionEnv import FantacalcioAstaChiamataEnv
from fantacalcioNet import FantacalcioNet
from fantacalcioAgent import FantacalcioDQNAgent
from fantacalcioTrainer import FantacalcioTrainer

def setup_directories():
    """Crea le directory necessarie."""
    directories = ['data', 'models', 'logs', 'plots', 'checkpoints', 'results']
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    print(f"✅ Directory create: {', '.join(directories)}")

def validate_data_files(players_file, quotazioni_file):
    """Valida i file dati."""
    if not Path(players_file).exists():
        print(f"❌ File giocatori non trovato: {players_file}")
        return False
    
    if not Path(quotazioni_file).exists():
        print(f"❌ File quotazioni non trovato: {quotazioni_file}")
        return False
    
    try:
        # Test caricamento
        players_df = pd.read_excel(players_file)
        quotazioni_df = pd.read_excel(quotazioni_file)
        
        print(f"✅ File dati validati:")
        print(f"   Giocatori: {len(players_df)} righe")
        print(f"   Quotazioni: {len(quotazioni_df)} righe")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore nella lettura dei file: {e}")
        return False

def train_model(players_file, quotazioni_file, config):
    """Funzione principale per il training."""
    print("\n🚀 AVVIO TRAINING")
    print("=" * 50)
    
    # Carica e prepara dati
    print("📊 Caricamento dati...")
    try:
        giocatori_df = get_player_stats(players_file, quotazioni_file)
        print(f"✅ Dati caricati: {len(giocatori_df)} giocatori")
    except Exception as e:
        print(f"❌ Errore caricamento dati: {e}")
        return None
    
    # Crea environment
    print("🏟️ Creazione environment...")
    env = FantacalcioAstaChiamataEnv(
        giocatori_df=giocatori_df,
        n_agenti=config['n_agents'],
        budget_iniziale=config['budget'],
        max_offerta_iniziale=config.get('max_offerta_iniziale', 50)
    )
    print(f"✅ Environment creato con {config['n_agents']} agenti")
    
    # Crea agente
    print("🤖 Creazione agente...")
    agent = FantacalcioDQNAgent(
        obs_size=env.observation_space.shape[0],
        max_players=len(giocatori_df),
        max_budget=config['budget'],
        lr=config['learning_rate'],
        gamma=config['gamma'],
        epsilon_decay=config['epsilon_decay'],
        memory_size=config['memory_size'],
        batch_size=config['batch_size']
    )
    print(f"✅ Agente creato con {agent.q_network.count_parameters():,} parametri")
    
    # Crea trainer
    print("🏋️ Configurazione trainer...")
    trainer = FantacalcioTrainer(
        env=env,
        agent=agent,
        n_episodes=config['episodes'],
        validation_freq=config.get('validation_freq', 100),
        save_freq=config.get('save_freq', 100),
        early_stopping_patience=config.get('early_stopping_patience', 200)
    )
    
    # Avvia training
    print(f"🎯 Inizio training per {config['episodes']} episodi...")
    history = trainer.train(verbose=True)
    
    # Salva modello finale
    final_model_path = f"models/fantacalcio_agent_{config['episodes']}ep.pth"
    agent.save_model(final_model_path)
    
    print(f"\n✅ TRAINING COMPLETATO!")
    print(f"📁 Modello salvato: {final_model_path}")
    
    return history, agent, trainer

def test_model(model_path, players_file, quotazioni_file, config):
    """Funzione per testare un modello addestrato."""
    print(f"\n🧪 TEST MODELLO: {model_path}")
    print("=" * 50)
    
    # Carica dati
    giocatori_df = get_player_stats(players_file, quotazioni_file)
    
    # Crea environment
    env = FantacalcioAstaChiamataEnv(
        giocatori_df=giocatori_df,
        n_agenti=config['n_agents'],
        budget_iniziale=config['budget']
    )
    
    # Crea e carica agente
    agent = FantacalcioDQNAgent(
        obs_size=env.observation_space.shape[0],
        max_players=len(giocatori_df),
        max_budget=config['budget']
    )
    
    try:
        agent.load_model(model_path)
        print(f"✅ Modello caricato: {model_path}")
    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return None
    
    # Crea trainer per valutazione
    trainer = FantacalcioTrainer(env, agent)
    
    # Esegui test
    results = trainer.evaluate(
        n_episodes=config.get('test_episodes', 10),
        render=config.get('render', False),
        verbose=True
    )
    
    return results

def interactive_mode(model_path, players_file, quotazioni_file, config):
    """Modalità interattiva per giocare contro l'agente."""
    print(f"\n🎮 MODALITÀ INTERATTIVA")
    print("=" * 50)
    
    # Setup come test_model
    giocatori_df = get_player_stats(players_file, quotazioni_file)
    env = FantacalcioAstaChiamataEnv(giocatori_df=giocatori_df, n_agenti=config['n_agents'], budget_iniziale=config['budget'])
    
    agent = FantacalcioDQNAgent(
        obs_size=env.observation_space.shape[0],
        max_players=len(giocatori_df),
        max_budget=config['budget']
    )
    
    agent.load_model(model_path)
    
    # Implementazione modalità interattiva semplificata
    print("🎯 Avvio partita interattiva...")
    print("Tu sarai l'Agente 0, l'IA controlla gli altri")
    
    state = env.reset()
    env.render()
    
    while True:
        fase_chiamata = (env.fase == "chiamata")
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            break
        
        # Determina chi deve agire
        agente_attivo = env.agente_di_turno if fase_chiamata else env.agente_rilancio
        
        if agente_attivo == 0:  # Turno umano
            print(f"\n🎯 È il tuo turno!")
            print(f"Azioni disponibili: {len(valid_actions)}")
            print("Inserisci l'indice dell'azione (0 per la prima): ", end="")
            
            try:
                choice = int(input())
                if 0 <= choice < len(valid_actions):
                    action = valid_actions[choice]
                else:
                    print("Scelta non valida, uso prima azione disponibile")
                    action = valid_actions[0]
            except (ValueError, KeyboardInterrupt):
                print("\nUscita...")
                break
        else:  # Turno IA
            action = agent.act(state, valid_actions, fase_chiamata, training=False)
            print(f"\n🤖 Agente {agente_attivo} sceglie: {action}")
        
        # Esegui azione
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        env.render()
        
        if done:
            print(f"\n🏁 PARTITA TERMINATA!")
            if 'final_scores' in info:
                for agente, score in info['final_scores'].items():
                    marker = "👤 TU" if agente == 'agente_0' else "🤖"
                    print(f"  {marker} {agente}: Valore {score['valore_rosa']:.1f}")
            break

def main():
    """Funzione principale con argomenti da command line."""
    parser = argparse.ArgumentParser(description='Fantacalcio RL - Sistema completo')
    
    # Comando principale
    parser.add_argument('command', choices=['train', 'test', 'interactive', 'setup'], 
                       help='Comando da eseguire')
    
    # File dati
    parser.add_argument('--players', type=str, default='data/',
                       help='File Excel giocatori')
    parser.add_argument('--quotazioni', type=str, default='data/quotazioni.xlsx', 
                       help='File Excel quotazioni')
    
    # Parametri training
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Numero episodi training')
    parser.add_argument('--agents', type=int, default=8,
                       help='Numero agenti asta')
    parser.add_argument('--budget', type=int, default=500,
                       help='Budget iniziale')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    # Parametri test
    parser.add_argument('--model', type=str, default='models/fantacalcio_agent_final.pth',
                       help='Path modello per test/interactive')
    parser.add_argument('--test-episodes', type=int, default=10,
                       help='Episodi di test')
    parser.add_argument('--render', action='store_true',
                       help='Mostra rendering durante test')
    
    # Altre opzioni
    parser.add_argument('--create-data', action='store_true',
                       help='Crea dati di esempio se mancanti')
    
    args = parser.parse_args()
    
    # Configurazione
    config = {
        'episodes': args.episodes,
        'n_agents': args.agents,
        'budget': args.budget,
        'learning_rate': args.lr,
        'gamma': 0.99,
        'epsilon_decay': args.episodes // 2,
        'memory_size': 100000,
        'batch_size': 64,
        'test_episodes': args.test_episodes,
        'render': args.render
    }
    
    print("🏆 FANTACALCIO RL - Sistema Completo")
    print("=" * 50)

    
    # Setup directory
    if args.command == 'setup':
        setup_directories()
        create_example_data()
        print("✅ Setup completato!")
        return 0
    
    setup_directories()
    
    # Gestione dati mancanti
    if not validate_data_files(args.players, args.quotazioni):
        if args.create_data:
            print("📝 Creazione dati di esempio...")
            args.players, args.quotazioni = create_example_data()
        else:
            print("❌ File dati mancanti. Usa --create-data per crearli")
            return 1
    
    # Esegui comando
    if args.command == 'train':
        history, agent, trainer = train_model(args.players, args.quotazioni, config)
        if history:
            print(f"🎉 Training completato con successo!")
    
    elif args.command == 'test':
        if not Path(args.model).exists():
            print(f"❌ Modello non trovato: {args.model}")
            return 1
        
        results = test_model(args.model, args.players, args.quotazioni, config)
        if results:
            print(f"🎉 Test completato!")
    
    elif args.command == 'interactive':
        if not Path(args.model).exists():
            print(f"❌ Modello non trovato: {args.model}")
            return 1
        
        interactive_mode(args.model, args.players, args.quotazioni, config)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)