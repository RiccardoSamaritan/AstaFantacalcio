import pandas as pd

def get_player_stats(players_path, quotazioni_path):
    """
    Funzione per ottenere un datagram statistiche dei giocatori dai file Excel con statistiche e quotazioni.
    :param players_path: Percorso del file Excel contenente i dati dei giocatori.
    :param quotazioni_path: Percorso del file Excel contenente le quotazioni dei giocatori.
    :return: DataFrame contenente le statistiche dei giocatori e le loro quotazioni.
    """
    # Carica i dati dei giocatori
    players_data = pd.read_excel(players_path)

    # Carico i dati delle quotazioni 
    quotazioni = pd.read_excel(quotazioni_path, usecols=['Id', 'Qt.I'])

    # Faccio merge tra i due dataframe
    players_data = players_data.merge(quotazioni, on='Id', how='left')
    
    # Rimuovo i giocatori senza quotazione
    players_data = players_data.dropna(subset=['Qt.I'])
    
    # Rinomino la colonna 'Qt.I' in 'Quota_base'
    players_data = players_data.rename(columns={'Qt.I': 'Valore'})

    print(f"Dataset creato con {len(players_data)} giocatori")
    
    return players_data