import numpy as np
import pygame
import sys
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque


# Dimensioni ridotte per una visualizzazione più chiara
DIMENSIONI = 5
DIMENSIONI_CELLA = 100  # Dimensione maggiore per cella per una migliore visualizzazione




schermo = pygame.display.set_mode((DIMENSIONI * DIMENSIONI_CELLA, DIMENSIONI * DIMENSIONI_CELLA))
pygame.display.set_caption("Labirinto Q-Learning")
pygame.init()

class Agent:
    """Definisce l'agente nel labirinto."""
    def __init__(self, i=0, j=0):
        self.i = i  # Posizione verticale dell'agente
        self.j = j  # Posizione orizzontale dell'agente

    @property
    def loc(self):
        """Restituisce la posizione corrente dell'agente."""
        return (self.i, self.j)

    def vmove(self, direction):
        """Muove l'agente verticalmente."""
        return Agent(self.i + direction, self.j)

    def hmove(self, direction):
        """Muove l'agente orizzontalmente."""
        return Agent(self.i, self.j + direction)

class Maze:
    """Rappresenta il labirinto con l'agente e l'obiettivo."""
    def __init__(self, rows, columns, goal_position):
        self.rows = rows  # Numero di righe nel labirinto
        self.columns = columns  # Numero di colonne nel labirinto
        self.agent = Agent()  # Crea un nuovo agente
        self.goal_position = goal_position  # Posizione dell'obiettivo
        self.env = np.zeros((rows, columns))  # Rappresentazione del labirinto come array
        self.env[goal_position] = 2  # Marca la posizione dell'obiettivo nel labirinto
        self.add_walls()

    
    def add_walls(self):
        # Aggiungi muri in maniera casuale
        for _ in range(int(self.rows * self.columns * 0.2)):  # circa il 20% delle celle
            x, y = random.randint(0, self.rows - 1), random.randint(0, self.columns - 1)
            if (x, y) != self.goal_position and (x, y) != (self.agent.i, self.agent.j):
                self.env[x][y] = 1

    def is_valid_move(self, agent):
        """Controlla se la mossa dell'agente è valida (non attraversa muri)."""
        if 0 <= agent.i < self.rows and 0 <= agent.j < self.columns:
            return self.env[agent.i, agent.j] != 1
        return False

    def move_agent(self, action):
        """Muove l'agente nel labirinto in base all'azione scelta."""
        # Definisce le azioni di movimento: 0=Su, 1=Giu, 2=Destra, 3=Sinistra
        if action == 0:
            new_agent = self.agent.vmove(-1)
        elif action == 1:
            new_agent = self.agent.vmove(1)
        elif action == 2:
            new_agent = self.agent.hmove(1)
        elif action == 3:
            new_agent = self.agent.hmove(-1)
        
        # Se la mossa è valida, aggiorna la posizione dell'agente
        if self.is_valid_move(new_agent):
            self.agent = new_agent
            return self.get_reward(new_agent), False
        # Ritorna una penalità per mosse contro i muri o mosse non valide
        return -0.1, False

    def get_reward(self, agent):
        """Calcola il reward basato sulla posizione dell'agente."""
        if (agent.i, agent.j) == self.goal_position:
            return 1  # Reward per aver raggiunto l'obiettivo
        return -0.01  # Piccola penalità per ogni mossa

    def get_state(self):
        """Restituisce lo stato corrente del labirinto e la posizione dell'agente."""
        state = np.copy(self.env).flatten()
        # Codifica la posizione dell'agente come vettore di stato one-hot
        agent_state = np.zeros(self.rows * self.columns)
        agent_pos = self.agent.i * self.columns + self.agent.j
        agent_state[agent_pos] = 1
        return np.concatenate([state, agent_state])

class Experience:
    """Gestisce la memoria e l'apprendimento dell'agente."""
    def __init__(self, model, max_memory=1000, discount=0.95):
        self.model = model  # Modello di rete neurale per l'apprendimento Q
        self.memory = deque(maxlen=max_memory)  # Memoria per esperienze passate
        self.discount = discount  # Fattore di sconto per il reward futuro

    def remember(self, state, action, reward, next_state, done):
        """Memorizza un'esperienza nell'agente."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """Riproduce le esperienze per allenare il modello."""
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.discount * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

class QLearningAgent:
    """Agente che impara a navigare nel labirinto usando Q-Learning."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Dimensione dello stato
        self.action_size = action_size  # Numero di azioni possibili
        self.learning_rate = 0.001  # Tasso di apprendimento
        self.epsilon = 1.0  # Probabilità di scegliere un'azione casuale
        self.epsilon_min = 0.01  # Minima probabilità di esplorazione
        self.epsilon_decay = 0.995  # Tasso di decadimento dell'esplorazione
        self.model = self._build_model() 
        self.memory = Experience(self._build_model(), max_memory=2000, discount=0.95)  # Esperienza dell'agente

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size=32):
        """Riproduce le esperienze passate per allenare il modello."""
        self.memory.replay(batch_size)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Loop principale del gioco
def run_game():
    maze = Maze(DIMENSIONI, DIMENSIONI, (DIMENSIONI-1, DIMENSIONI-1))
    state_size = (maze.rows * maze.columns) * 2  # stato attuale + posizione dell'agente
    action_size = 4  # su, giù, destra, sinistra
    agent = QLearningAgent(state_size, action_size)
    batch_size = 32

    for e in range(1000):  # numero di episodi
        state = np.reshape(maze.get_state(), [1, state_size])
        for time in range(500):  # limite di mosse per episodio
            action = agent.act(state)
            reward, done = maze.move_agent(action)
            next_state = np.reshape(maze.get_state(), [1, state_size])
            agent.memory.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episodio: {e}/{1000}, punteggio: {time}, epsilon: {agent.epsilon:.2}")
                break
            if len(agent.memory.memory) > batch_size:
                agent.replay(batch_size)
            draw_maze(maze)  # Aggiorna la visualizzazione ad ogni mossa
            pygame.time.delay(100)  # Introduce un ritardo di 100 millisecondi

            # Gestione degli eventi Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

def draw_maze(maze):
    for i in range(maze.rows):
        for j in range(maze.columns):
            # Calcola la posizione e le dimensioni del rettangolo
            rect = pygame.Rect(j * DIMENSIONI_CELLA, i * DIMENSIONI_CELLA, DIMENSIONI_CELLA, DIMENSIONI_CELLA)
            
            # Disegna un muro se l'elemento nella griglia è un muro
            if maze.env[i][j] == 1:
                pygame.draw.rect(schermo, (0, 0, 0), rect)
            
            # Disegna l'obiettivo se l'elemento nella griglia è l'obiettivo
            elif (i, j) == maze.goal_position:
                pygame.draw.rect(schermo, (0, 255, 0), rect)
            
            # Disegna lo spazio vuoto
            else:
                pygame.draw.rect(schermo, (255, 255, 255), rect)
    
    # Disegna l'agente come un cerchio
    agent_pos = maze.agent.loc
    agent_center = (int(agent_pos[1] * DIMENSIONI_CELLA + DIMENSIONI_CELLA / 2), int(agent_pos[0] * DIMENSIONI_CELLA + DIMENSIONI_CELLA / 2))
    pygame.draw.circle(schermo, (255, 0, 0), agent_center, int(DIMENSIONI_CELLA / 4))
    
    # Aggiorna la visualizzazione
    pygame.display.flip()

    
    # Aggiorna la visualizzazione
    pygame.display.flip()



    # Introduce un ritardo di 100 millisecondi
    pygame.time.delay(1000)


    

if __name__ == "__main__":
    run_game()  # Avvia il gioco
    pygame.quit()  # Chiude Pygame quando il gioco è finito
    sys.exit()  # Esce dal programma
