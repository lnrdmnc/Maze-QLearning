from datetime import datetime
import numpy as np
from collections import namedtuple
import random
from pygame.locals import *  # Importa costanti come QUIT, KEYDOWN, ecc.
from OpenGL.GL import *  # Importa funzioni dal modulo OpenGL
from OpenGL.GLU import *  # Importa funzioni dal modulo OpenGL Utility
import sys
""" 

States 
0 means free
-1 mean not traversable
1 means goal 

"""

class Agent:
    def __init__(self,i=0,j=0):
        self.i=i
        self.j=j

    @property
    def loc(self):
        return (self.i,self.j)
    
    # moviemnto verticale
    def vmove(self, direction):
        direction= 1 if direction >0 else -1
        return Agent(self.i + direction,self.j)
    
    # movimento orizzontale
    def hmove(self,direction):
        direction= 1 if direction >0 else -1
        return Agent(self.i,self.j+direction)
    
    def __repr__(self):
        return str(self.loc)
    
    def draw(self, TILE_SIZE):
        x = self.j * TILE_SIZE
        y = self.i * TILE_SIZE
        z = TILE_SIZE / 2  # Altezza del giocatore sopra il pavimento

        # Disegna il giocatore come un cubo
        glBegin(GL_QUADS)
        glColor3fv((1, 0, 0))  # Colore del giocatore (rosso)
        glVertex3fv((x, y, z))
        glVertex3fv((x + TILE_SIZE, y, z))
        glVertex3fv((x + TILE_SIZE, y + TILE_SIZE, z))
        glVertex3fv((x, y + TILE_SIZE, z))
        glEnd()
    

class Maze:
    def __init__(self, rows, columns, goal_position):
        self.rows = rows
        self.columns = columns
        self.agent = Agent(0, 0)  # Inizializza l'agente nella posizione (0, 0)
        self.env = np.zeros((rows, columns))  # Crea una griglia del labirinto

        self.goal_position = goal_position  # Memorizza la posizione dell'obiettivo
        self.env[goal_position] = 1  # Imposta la cella obiettivo nella griglia del labirinto

    def get_reward(self, new_position):
        """Calcola il reward in base alla nuova posizione dell'agente."""
        if new_position == self.goal_position:
            return 1  # Reward positivo per raggiungere l'obiettivo
        else:
            return -0.1  # Piccola penalità per ogni mossa


    def nei_limiti(self,i,j):
        nr,nc=self.env.shape
        return i>=0 and i<nr and j>=0 and j<nc
    
    def agente_nei_limiti(self,a):
        return self.nei_limiti(a.i,a.j)
    
    def agent_dient(self,a):
        return not self.env[a.i,a.j] == -1
    
    def is_valid_new_agent(self,a):
        return self.agente_nei_limiti(a) and self.agent_dient(a)
      
    def computa_possibili_movimenti(self):
        a=self.agent
        moves= [
            a.vmove(1),
            a.vmove(-1),
            a.hmove(1),
            a.hmove(-1),
        ]
        return [m for m in moves if self.is_valid_new_agent(m)]
    
    def has_won(self):
        a=self.agent
        return self.env[a.i, a.j] == 1
    
    def do_a_move(self,a):
        assert self.is_valid_new_agent(a),"Non puoi andare qui"
        self.agent=a
        return 10 if self.has_won() else -0.1
    
    def visualizza(self):
        assert self.agente_nei_limiti(self.mousy), "Fuori dai limiti"
        e=self.env.copy()
        m=self.agent
        e[m.i,m.j]=6
        print(e)

    def get_state(self):
        """Ritorna lo stato attuale del labirinto in un formato adatto per il modello di Q-Learning."""
        state = np.copy(self.env)
        state[self.agent.i, self.agent.j] = 2  # Marca la posizione dell'agente
        return state.reshape((1, -1))


        """ 
        Aggiorna la posizione dell'agente in base all'azione.
        Azioni: 0 = Su, 1 = Giù, 2 = Destra, 3 = Sinistra
        """

    def update_agent(self, action):
    # Ottiene l'agente corrente
        current_agent = self.agent

        # Calcola la nuova posizione in base all'azione
        if action == 0:  # Su
            new_agent = current_agent.vmove(-1)
        elif action == 1:  # Giù
            new_agent = current_agent.vmove(1)
        elif action == 2:  # Destra
            new_agent = current_agent.hmove(1)
        elif action == 3:  # Sinistra
            new_agent = current_agent.hmove(-1)
        else:  
            raise ValueError("Azione non valida")

        # Controlla se la nuova posizione è valida
        if self.is_valid_new_agent(new_agent):
            # Aggiorna la posizione dell'agente se la nuova posizione è valida
            self.agent = new_agent
        else:
            # Qui puoi gestire cosa succede se la mossa non è valida
            # Ad esempio, puoi decidere di non muovere l'agente o di penalizzare l'agente per una mossa non valida
            # Per ora, lascia l'agente nella posizione corrente
            pass

        # Ritorna la nuova posizione dell'agente, utile per il calcolo del reward
        return self.agent.loc


class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        # Inizializzazione dell'oggetto Experience con un modello, memoria massima, e sconto per i reward futuri
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()  # Lista per memorizzare le esperienze passate
        self.num_actions = model.output_shape[-1]  # Numero di azioni possibili nel modello

    def remember(self, episode):
        # Memorizza un episodio nella memoria dell'esperienza
        # episode = [envstate, action, reward, envstate_next, game_over]
        # envstate è l'informazione sullo stato ambientale (labirinto) appiattito in 1D
        self.memory.append(episode)
        # Se la memoria supera la dimensione massima, rimuovi la più vecchia
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        # Predice il valore Q per uno stato ambientale dato
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        # Ottiene un batch casuale di dati dalla memoria
        env_size = self.memory[0][0].shape[1]  # Dimensione dell'envstate 1D
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))

        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate

            # Non ci devono essere valori target per azioni non prese
            targets[i] = self.predict(envstate)

            # Calcola il Q_sa, la policy derivata
            Q_sa = np.max(self.predict(envstate_next))

            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa

        return inputs, targets

def qtrain(model, maze, **opt):
    global epsilon
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    # Se si desidera continuare l'addestramento da un modello precedente,
    # basta fornire il nome del file h5 all'opzione weights_file
    if weights_file:
        print("Caricamento pesi da file: %s" % (weights_file,))
        model.load_weights(weights_file)

    # Costruisci l'ambiente/gioco da un array numpy: maze
    qmaze = Maze(maze)

    # Inizializza l'oggetto di replay dell'esperienza
    experience = Experience(model, max_memory=max_memory)

    win_history = []   # Storia delle partite vinte/persa
    n_free_cells = len(qmaze.free_cells)
    hsize = qmaze.maze.size//2   # Dimensione della finestra di storia
    win_rate = 0.0
    imctr = 1

    for epoch in range(n_epoch):
        loss = 0.0
        rat_cell = random.choice(qmaze.free_cells)
        qmaze.reset(rat_cell)
        game_over = False

        # Ottieni lo stato ambientale iniziale (canvas appiattito in 1D)
        envstate = qmaze.observe()

        n_episodes = 0
        while not game_over:
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            # Ottieni l'azione successiva
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(prev_envstate))

            # Applica l'azione, ottieni il reward e il nuovo stato ambientale
            envstate, reward, game_status = qmaze.act(action)
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False

            # Memorizza l'episodio (esperienza)
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1

            # Addestra il modello della rete neurale
            inputs, targets = experience.get_data(data_size=data_size)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0,
            )
            loss = model.evaluate(inputs, targets, verbose=0)

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize
    
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))
        # Controlliamo semplicemente se l'addestramento ha esaurito tutte le celle libere e se in tutti
        # i casi l'agente ha vinto
        if win_rate > 0.9: epsilon = 0.05
        if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):
            print("Raggiunto il 100%% di vittorie all'epoca: %d" % (epoch,))
            break

    # Salva i pesi addestrati del modello e l'architettura, verranno utilizzati dal codice di visualizzazione
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds

# Questa è una piccola utilità per stampare stringhe di tempo leggibili:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f secondi" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minuti" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f ore" % (h,)
    

def build_model(maze, lr=0.001):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model


model = build_model(maze)
qtrain(model, maze, epochs=1000, max_memory= 8* maze.size, data_size=32)