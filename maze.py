import numpy as np
from collections import namedtuple
import random
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
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
    def __init__(self,rows,columns):
        ##cambiare il nome
        self.mousy=Agent(0,0)
        self.env = np.zeros((rows, columns))
    
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
        a=self.mousy
        moves= [
            a.vmove(1),
            a.vmove(-1),
            a.hmove(1),
            a.hmove(-1),
        ]
        return [m for m in moves if self.is_valid_new_agent(m)]
    
    def do_a_move(self,a):
        assert self.is_valid_new_agent(a),"Non puoi andare qui"
        self.mousy=a
        return 10 if self.has_won() else -0.1
    
    def has_won(self):
        a=self.mousy
        return self.env[a.i, a.j] == 1
    
    def visualizza(self):
        assert self.agente_nei_limiti(self.mousy), "Fuori dai limiti"
        e=self.env.copy()
        m=self.mousy
        e[m.i,m.j]=6
        print(e)


def make_test_maze(rows,columns):
    m=Maze(rows,columns)
    e=m.env
    e[3,3]=1
    e[0, 1:3]= -1
    e[1, 2:]= -1
    e[3, 0:2]= -1
    return m

def draw(TILE_SIZE, player, tiles, maze):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(-TILE_SIZE * len(maze.env[0]) / 2, -TILE_SIZE * len(maze.env) / 2, -TILE_SIZE * len(maze.env))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(1, 3, 1, 1)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_maze(TILE_SIZE, tiles, maze)
        player.draw(TILE_SIZE)  # Chiamiamo ora il metodo draw del giocatore

        pygame.display.flip()
        pygame.time.wait(10)

def draw_maze(TILE_SIZE, tiles, maze):
    default_texture = "C:\\Users\\Leonardo\\Desktop\\Maze-QLearning\\texture\\coral_stone_wall_diff_1k.jpg"

      # Sostituisci con il percorso della tua texture predefinita

    for row in range(len(maze.env)):
        for column in range(len(maze.env[row])):
            x = column * TILE_SIZE
            y = row * TILE_SIZE
            z = 0
            texture_key = maze.env[row][column]

            if texture_key in tiles:
                tile_texture = tiles[texture_key]
            else:
                tile_texture = default_texture

            glBegin(GL_QUADS)
            glColor3fv((1, 1, 1))
            glVertex3fv((x, y, z))
            glVertex3fv((x + TILE_SIZE, y, z))
            glVertex3fv((x + TILE_SIZE, y + TILE_SIZE, z))
            glVertex3fv((x, y + TILE_SIZE, z))
            glEnd()

            if maze.env[row][column] == 1:
                glBegin(GL_QUADS)
                glColor3fv((0, 0, 1))
                glVertex3fv((x, y, z + TILE_SIZE))
                glVertex3fv((x + TILE_SIZE, y, z + TILE_SIZE))
                glVertex3fv((x + TILE_SIZE, y + TILE_SIZE, z + TILE_SIZE))
                glVertex3fv((x, y + TILE_SIZE, z + TILE_SIZE))
                glEnd()

    pygame.display.flip()


         
def main():
    m=make_test_maze(4,4)
    final_score=0
    while not m.has_won():
        moves =m.computa_possibili_movimenti()
        random.shuffle(moves)
        final_score+=m.do_a_move(moves[0])
        m.visualizza()   
    print(f'punteggio finale {final_score}')
    

if __name__ == '__main__':
    m = make_test_maze(4, 4)
    final_score = 0

    # Configurazione del giocatore
    player = Agent()

    # Configurazione delle texture per il pavimento e le pareti
    tiles = {
    0: r"C:\Users\Leonardo\Desktop\Maze-QLearning\texture\coral_stone_wall_diff_1k.jpg",
    1: r"C:\Users\Leonardo\Desktop\Maze-QLearning\texture\coral_stone_wall_diff_1k.jpg"
    }

    # Dimensione di ogni cella nel labirinto
    TILE_SIZE = 40

    # Chiamata alla funzione draw
    draw(TILE_SIZE, player, tiles, m)

   
