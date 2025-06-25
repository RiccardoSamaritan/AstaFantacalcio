import torch
import torch.nn as nn
import torch.nn.functional as F

class FantacalcioNet(nn.Module):
    """
    Rete neurale per l'agente del fantacalcio.
    Utilizza un'architettura che separa la valutazione delle chiamate dai rilanci.
    
    Architecture:
    - Encoder comune per processare le osservazioni
    - Dual heads: uno per chiamate, uno per rilanci
    - Value head per critic (opzionale per future implementazioni A3C/PPO)
    """
    
    def __init__(self, 
                 obs_size: int, 
                 max_players: int, 
                 max_budget: int,
                 hidden_sizes: list = [512, 256, 128],
                 dropout_rate: float = 0.3,
                 activation: str = "relu"):
        """
        Inizializza la rete neurale.
        
        Args:
            obs_size: Dimensione dello spazio delle osservazioni
            max_players: Numero massimo di giocatori
            max_budget: Budget massimo (per dimensionare output)
            hidden_sizes: Lista delle dimensioni dei layer nascosti
            dropout_rate: Tasso di dropout per regolarizzazione
            activation: Funzione di attivazione ('relu', 'tanh', 'leaky_relu')
        """
        super(FantacalcioNet, self).__init__()
        
        self.obs_size = obs_size
        self.max_players = max_players
        self.max_budget = max_budget
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # Selezione funzione di attivazione
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Activation {activation} non supportata")
        
        # Encoder comune per processare le osservazioni
        self.encoder = self._build_encoder()
        
        # Head per la fase di chiamata (valuta giocatore + offerta iniziale)
        self.chiamata_head = self._build_chiamata_head()
        
        # Head per la fase di rilancio (valuta solo offerte)
        self.rilancio_head = self._build_rilancio_head()
        
        # Value head per critic (utile per algoritmi actor-critic)
        self.value_head = self._build_value_head()
        
        # Inizializzazione pesi
        self._initialize_weights()
        
    def _build_encoder(self):
        """Costruisce l'encoder comune."""
        layers = []
        input_size = self.obs_size
        
        for i, hidden_size in enumerate(self.hidden_sizes):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(self.activation)
            
            # Dropout dopo ogni layer nascosto (eccetto l'ultimo)
            if i < len(self.hidden_sizes) - 1:
                layers.append(nn.Dropout(self.dropout_rate))
            
            input_size = hidden_size
        
        return nn.Sequential(*layers)
    
    def _build_chiamata_head(self):
        """Costruisce l'head per le chiamate."""
        encoder_output_size = self.hidden_sizes[-1]
        intermediate_size = 256
        output_size = self.max_players + self.max_budget
        
        return nn.Sequential(
            nn.Linear(encoder_output_size, intermediate_size),
            self.activation,
            nn.Dropout(self.dropout_rate * 0.5),  # Dropout ridotto negli heads
            nn.Linear(intermediate_size, output_size)
        )
    
    def _build_rilancio_head(self):
        """Costruisce l'head per i rilanci."""
        encoder_output_size = self.hidden_sizes[-1]
        intermediate_size = 256
        output_size = self.max_budget + 1  # +1 per l'azione "passa" (0)
        
        return nn.Sequential(
            nn.Linear(encoder_output_size, intermediate_size),
            self.activation,
            nn.Dropout(self.dropout_rate * 0.5),
            nn.Linear(intermediate_size, output_size)
        )
    
    def _build_value_head(self):
        """Costruisce il value head per critic."""
        encoder_output_size = self.hidden_sizes[-1]
        
        return nn.Sequential(
            nn.Linear(encoder_output_size, 64),
            self.activation,
            nn.Linear(64, 1)
        )
    
    def _initialize_weights(self):
        """Inizializza i pesi della rete."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization per layer lineari
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, obs, fase_chiamata=True):
        """
        Forward pass della rete.
        
        Args:
            obs: Tensor delle osservazioni [batch_size, obs_size]
            fase_chiamata: Se True usa chiamata_head, se False usa rilancio_head
            
        Returns:
            action_logits: Logits per le azioni [batch_size, action_space_size]
            value: Valore stimato dello stato [batch_size, 1]
        """
        # Encoding comune
        encoded = self.encoder(obs)
        
        # Selezione head appropriato
        if fase_chiamata:
            action_logits = self.chiamata_head(encoded)
        else:
            action_logits = self.rilancio_head(encoded)
        
        # Value estimation
        value = self.value_head(encoded)
        
        return action_logits, value
    
    def get_action_probabilities(self, obs, fase_chiamata=True, temperature=1.0):
        """
        Ottieni probabilità delle azioni con controllo della temperatura.
        
        Args:
            obs: Osservazioni
            fase_chiamata: Tipo di fase
            temperature: Temperatura per softmax (1.0 = normale, <1 = più deterministico)
            
        Returns:
            action_probs: Probabilità delle azioni
            value: Valore stimato
        """
        action_logits, value = self.forward(obs, fase_chiamata)
        
        # Applica temperatura
        if temperature != 1.0:
            action_logits = action_logits / temperature
        
        # Softmax per ottenere probabilità
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs, value
    
    def get_feature_importance(self, obs, fase_chiamata=True):
        """
        Calcola l'importanza delle features usando gradients.
        Utile per interpretabilità del modello.
        
        Args:
            obs: Osservazioni [1, obs_size]
            fase_chiamata: Tipo di fase
            
        Returns:
            importance: Importanza di ogni feature
        """
        obs.requires_grad_(True)
        action_logits, _ = self.forward(obs, fase_chiamata)
        
        # Calcola gradiente rispetto all'output massimo
        max_output = torch.max(action_logits)
        max_output.backward()
        
        # L'importanza è il gradiente assoluto
        importance = torch.abs(obs.grad).squeeze().detach()
        
        return importance
    
    def count_parameters(self):
        """Conta il numero di parametri trainabili."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """Restituisce informazioni sul modello."""
        total_params = self.count_parameters()
        
        # Conta parametri per componente
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        chiamata_params = sum(p.numel() for p in self.chiamata_head.parameters() if p.requires_grad)
        rilancio_params = sum(p.numel() for p in self.rilancio_head.parameters() if p.requires_grad)
        value_params = sum(p.numel() for p in self.value_head.parameters() if p.requires_grad)
        
        info = {
            'total_parameters': total_params,
            'encoder_parameters': encoder_params,
            'chiamata_head_parameters': chiamata_params,
            'rilancio_head_parameters': rilancio_params,
            'value_head_parameters': value_params,
            'obs_size': self.obs_size,
            'max_players': self.max_players,
            'max_budget': self.max_budget,
            'hidden_sizes': self.hidden_sizes,
            'dropout_rate': self.dropout_rate
        }
        
        return info
    
    def freeze_encoder(self):
        """Congela i parametri dell'encoder (per transfer learning)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Scongela i parametri dell'encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def save_checkpoint(self, filepath, optimizer=None, epoch=None, loss=None):
        """
        Salva un checkpoint del modello.
        
        Args:
            filepath: Percorso dove salvare
            optimizer: Ottimizzatore (opzionale)
            epoch: Epoca corrente (opzionale)
            loss: Loss corrente (opzionale)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'epoch': epoch,
            'loss': loss
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_from_checkpoint(cls, filepath, device='cpu'):
        """
        Carica un modello da checkpoint.
        
        Args:
            filepath: Percorso del checkpoint
            device: Device su cui caricare il modello
            
        Returns:
            model: Modello caricato
            info: Informazioni aggiuntive dal checkpoint
        """
        checkpoint = torch.load(filepath, map_location=device)
        model_info = checkpoint['model_info']
        
        # Ricrea il modello con le stesse configurazioni
        model = cls(
            obs_size=model_info['obs_size'],
            max_players=model_info['max_players'],
            max_budget=model_info['max_budget'],
            hidden_sizes=model_info['hidden_sizes'],
            dropout_rate=model_info['dropout_rate']
        )
        
        # Carica i pesi
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Informazioni aggiuntive
        info = {
            'epoch': checkpoint.get('epoch'),
            'loss': checkpoint.get('loss'),
            'model_info': model_info
        }
        
        return model, info

# Funzioni di utilità per testing e debugging
def test_network():
    """Test base della rete neurale."""
    print("🧪 Testing FantacalcioNet...")
    
    # Parametri di test
    obs_size = 100
    max_players = 400
    max_budget = 500
    batch_size = 32
    
    # Crea rete
    net = FantacalcioNet(obs_size, max_players, max_budget)
    print(f"✅ Rete creata: {net.count_parameters():,} parametri")
    
    # Test forward pass
    obs = torch.randn(batch_size, obs_size)
    
    # Test fase chiamata
    action_logits_call, value_call = net(obs, fase_chiamata=True)
    print(f"✅ Chiamata - Output shape: {action_logits_call.shape}, Value: {value_call.shape}")
    
    # Test fase rilancio
    action_logits_bid, value_bid = net(obs, fase_chiamata=False)
    print(f"✅ Rilancio - Output shape: {action_logits_bid.shape}, Value: {value_bid.shape}")
    
    # Test probabilità
    probs, _ = net.get_action_probabilities(obs[:1], fase_chiamata=True)
    print(f"✅ Probabilità - Shape: {probs.shape}, Sum: {probs.sum():.3f}")
    
    # Test informazioni modello
    info = net.get_model_info()
    print(f"✅ Info modello: {info}")
    
    print("🎉 Tutti i test superati!")

if __name__ == "__main__":
    test_network()