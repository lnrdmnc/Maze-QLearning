import numpy as np
import pygame
import sys



visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
pos_mark = 0.5      # The current rat cell will be painteg by gray 0.5
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

# Exploration factor
epsilon = 0.1






class Maze:
    def __init__(self,labirinto, pos=(0,0)):
        self._labirinto=np.array(labirinto)
        righe,colonne=self._labirinto.shape
        self.destinazione=(righe-1,colonne-1) #cella obiettivo dove si trova l'uscita
        self.celle_libere=[(r,c) for r in range(righe) for c in range(colonne) if self._labirinto[r, c] == 1.0]
        self.celle_libere.remove(self.destinazione)

        if self._labirinto[self.destinazione] == 0.0:
            raise Exception("Labirinto non valido: la cella obiettivo non può essere bloccata!")
        
        if not pos in self.celle_libere:
            raise Exception("Posizione iniziale non valida: deve trovarsi su una cella libera")
        
        self.reset(pos)

    def reset(self,pos):
        self.pos=pos
        self._labirinto=np.copy(self._labirinto)
        nrighe,ncolonne=self._labirinto.shape
        righe,colonne=pos
        self._labirinto[righe,colonne]=pos_mark
        self.state=(righe,colonne,'start')
        self.min_reward=-0.5 * self._labirinto.size
        self.total_reward=0
        self.visited=set()

    def update_state(self, action):
        nrighe, ncolonne = self._labirinto.shape
        nriga, ncolonna, nmode = pos_row, pos_col, mode = self.state

        # Se la cella è libera, la segnala come visitata
        if self.maze[pos_row, pos_col] > 0.0:
            self.visited.add((pos_row, pos_col))  # segna la cella visitata

        # Ottiene le azioni valide per la posizione corrente 
        valid_actions = self.valid_actions()
        
        # Verifica se non ci sono azioni valide (l'agente è bloccato)
        if not valid_actions:
            nmode = 'blocked'
        # Altrimenti, verifica se l'azione è tra le azioni valide
        elif action in valid_actions:
            nmode = 'valid'
            # Aggiorna la posizione del topo in base all'azione scelta
            if action == LEFT:
                ncolonna -= 1
            elif action == UP:
                nriga -= 1
            if action == RIGHT:
                ncolonna += 1
            elif action == DOWN:
                nriga += 1
        else:
            # Azione non valida, nessun cambiamento nella posizione 
            mode = 'invalid'

        # Imposta il nuovo stato
        self.state = (nriga, ncolonna, nmode)

    def aggiornamento_stato(self):
        pos_riga, pos_colonna, mode = self.state
        nrighe, ncolonne = self._labirinto.shape
        if pos_riga == nrighe-1 and pos_colonna == ncolonne-1:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (pos_riga, pos_colonna) in self.visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04
        
    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

        
    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate
    
    def draw_env(self):
        canvas = np.copy(self.maze)
        nrighe, ncolonne = self._labirinto.shape
        # clear all visual marks
        for r in range(nrighe):
            for c in range(ncolonne):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the pos
        righe, colonne, valid = self.state
        canvas[righe, colonne] = pos_mark
        return canvas
    
    def get_reward(self):
        pos_riga, pos_colonna, mode = self.state
        nrows, ncols = self.maze.shape
        if pos_riga == nrows-1 and pos_colonna == ncols-1:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (pos_riga,pos_colonna) in self.visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        pos_riga, pos_colonna, mode = self.state
        nrighe, ncolonne = self._labirinto.shape
        if pos_riga == nrighe-1 and pos_colonna == ncolonne-1:
            return 'win'
        return 'not_over'

    def azioni_valide(self,cell=None):
        if cell is None:
            riga, colonna, mode = self.state
        else:
            riga, colonna = cell
        actions = [0, 1, 2, 3]
        nrighe, ncolonne = self._labirinto.shape
        if riga == 0:
            actions.remove(1)
        elif riga == nrighe-1:
            actions.remove(3)

        if colonna == 0:
            actions.remove(0)
        elif colonna == ncolonne-1:
            actions.remove(2)

        if riga>0 and self.maze[riga-1,colonna] == 0.0:
            actions.remove(1)

        if riga<nrighe -1 and self._labirinto[riga+1,colonna] == 0.0:
            actions.remove(3)

        if colonna>0 and self._labirinto[riga,colonna-1] == 0.0:
            actions.remove(0)

        if colonna<ncolonne-1 and self._labirinto[riga,colonna+1] == 0.0:
            actions.remove(2)
        return actions


# Inizializzazione Pygame
pygame.init()

# Definizione di costanti
SCREEN_WIDTH = 750
SCREEN_HEIGHT = 600
TILE_SIZE = 150
LABIRINTO = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1],
]

# Colori
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Creazione della finestra
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Labirinto")

# Caricamento delle immagini o definizione delle superfici delle tessere
# Assicurati di avere le immagini o le superfici delle tessere necessarie
# tiles = {0: pygame.image.load("tile_libera.png"), 1: pygame.image.load("tile_muro.png")}

# Definizione del giocatore
player_color = (255, 0, 0)
player_pos = [1, 1]

# Funzione per disegnare il labirinto
def draw_labirinto():
    for row in range(len(LABIRINTO)):
        for column in range(len(LABIRINTO[row])):
            x = column * TILE_SIZE
            y = row * TILE_SIZE
            pygame.draw.rect(screen, WHITE, (x, y, TILE_SIZE, TILE_SIZE))  # Rettangolo bianco come sfondo
            if LABIRINTO[row][column] == 1:
                pygame.draw.rect(screen, BLACK, (x, y, TILE_SIZE, TILE_SIZE))  # Rettangolo nero per i muri
            # Aggiungi qui il disegno delle immagini o superfici delle tessere

# Funzione per disegnare il giocatore
def draw_player():
    x = player_pos[1] * TILE_SIZE
    y = player_pos[0] * TILE_SIZE
    pygame.draw.rect(screen, player_color, (x, y, TILE_SIZE, TILE_SIZE))  # Rettangolo rosso per il giocatore

# Ciclo principale del gioco
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Pulizia dello schermo
    screen.fill(WHITE)

    # Disegna il labirinto e il giocatore
    draw_labirinto()
    draw_player()

    # Aggiorna la finestra
    pygame.display.flip()

    # Imposta il framerate
    pygame.time.Clock().tick(30)


