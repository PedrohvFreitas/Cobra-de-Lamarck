import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
from evo_syst import EvolutiveAlgorithm

torch.manual_seed(123)
MAX_MEMORY = 100_000
BATCH_SIZE = 1000       #Tamanho do batch que será usado na memoria de longo prazo
LR = 0.001              #Learning Rate    
num_agents = 20         #Tamanho da população 

K = 10

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.color = self.define_color()
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.score = 0

    def get_state(self, game):
        """Função que retorna as informações de estado para a cobra

            O formato da saida é um vetor com 11 parametros sendo eles:

            state = {Paredes_proximas(tam 3),Direção atual da cobra(tam 4), Localização da maça(tam 4)}
        
        """
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #Paredes proxias
            #Mostra se tem paredes a uma distancia de 1 bloco ao redor da cabeça 
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            #Direção atual da cobra
            #Mostra a direção em que a cobra esta indo
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            #Localização da maça
            #Orientação da maça em relação a cabeça 
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, device="cpu")


    def train_short_memory(self, state, action, reward, next_state, done):
        """Realiza o treinamento instântaneo, ajustando os pesos da rede neural
           imediatamente após cada passo dado pela cobra."""
        self.trainer.train_step(state, action, reward, next_state, done, device="cpu")

    def get_action(self, state):
        """A função processa o estado na rede neural para gerar 3 ações de movimento relativas à cabeça,
           utilizando uma estratégia ϵ-greedy (epsilon decrescente) para equilibrar a exploração aleatória
           inicial com a tomada de decisão aprendida."""
        
        self.epsilon = 80 - self.n_games #episilon decai com o numero de jogos 
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def define_color(self):
        """Define a cor da cobra de forma aleatoria"""
        return  torch.randint(low=0, high=255, size=(3, 1)).squeeze(1).numpy()

agents = []
for _ in range(num_agents):
        agents.append(Agent())

def train():
    """Função que organiza o treino. Alterna entre o algoritmo evolutivo, Aprendizado por Reforço e Cria a nova geração"""
    record = 0
    mutation_percentage = 0.2 #Porcentagem dos pesos das redes que será mutado
    epoch = 0

    while True:
        epoch += 1
        global agents
        for i, agent in enumerate(agents):
            
            agent.score = 0
            game = SnakeGameAI(color=agent.color)
            while True:
                state_old = agent.get_state(game)

                final_move = agent.get_action(state_old)

                reward, done, score = game.play_step(final_move)
                state_new = agent.get_state(game)

                agent.train_short_memory(state_old, final_move, reward, state_new, done)

                agent.remember(state_old, final_move, reward, state_new, done)

                if done:
                    agent.score += score
                    game.reset()
                    agent.n_games += 1
                    agent.train_long_memory()


                    if score > record:
                        record = score
                        agent.model.save()

                    print(f'Agent: {i}, Game: {agent.n_games}, Score: {score}, Epoch: {epoch}')

                    #Cada cobra treina 3 vezes DQN(Rl) antes de entrar no algoritmo evolutivo
                    if agent.n_games % 3 == 0:
                        break

        agents_dict = {}
        for i, agent in enumerate(agents):
            agents_dict[i] = agent.score
        evol_algorithm = EvolutiveAlgorithm(k=K)
        index_best, pairs, index_for_mutation = evol_algorithm.selection_mechanism(agents_dict=agents_dict)

        new_generation = []

        all_index = np.concatenate((index_best, index_for_mutation), axis=0)
        all_index = [int(item) for item in all_index]

        for i in all_index:
            gaussian = 'small' if i < K else 'big'
            mutated_agent = evol_algorithm.mutation(gaussian=gaussian, snake=agents[i], mutation_percentage=mutation_percentage)
            new_generation.append(mutated_agent)
        

        for pair in pairs:
            snakes = [agents[pair[0]], agents[pair[1]]]
            baby_dict = evol_algorithm.reproduction(snakes)
            baby = Agent()
            baby.model.load_state_dict(baby_dict)
            new_generation.append(baby)
        
        for _ in range(5):
            new_generation.append(Agent())
        agents = new_generation
        mutation_percentage -= 0.01 #Diminui a mutação a medida que o tempo passa

if __name__ == '__main__':
    train()