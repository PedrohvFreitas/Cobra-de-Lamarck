import torch
import random
import numpy as np
from collections import deque
import torch.multiprocessing as mp
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from evo_syst import EvolutiveAlgorithm

torch.manual_seed(123)
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.005
K = 10
SIMULTANEOUS_GAMES = 5
NUM_DQN_CYCLES = 3 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_game_state(game):
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
        
        dir_l, dir_r, dir_u, dir_d,
        
        game.food.x < game.head.x,
        game.food.x > game.head.x,
        game.food.y < game.head.y,
        game.food.y > game.head.y 
    ]

    return np.array(state, dtype=int)

def game_worker(conn, agent_color):
    game = SnakeGameAI(color=agent_color)
    
    state = get_game_state(game)
    conn.send(state)
    
    while True:
        command = conn.recv()
        
        if command == "RESET":
            game.reset()
            state = get_game_state(game)
            conn.send(state)
            continue
        elif command == "CLOSE":
            break
            
        final_move = command 
        
        reward, done, score = game.play_step(final_move)
        next_state = get_game_state(game)
        
        conn.send((next_state, reward, done, score))

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.color = self.define_color()
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 64, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.score = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, device=DEVICE)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done, device=DEVICE)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(DEVICE)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
    
    def define_color(self):
        return torch.randint(low=0, high=255, size=(3, 1)).squeeze(1).numpy()
    
    def to_gpu(self):
        self.model.to(DEVICE)
    
    def to_cpu(self):
        self.model.to("cpu")

def run_batch_of_agents(batch_agents, epoch):
    for ag in batch_agents:
        ag.to_gpu()
        ag.score = 0

    for game_cycle in range(NUM_DQN_CYCLES):
        print(f"  Ciclo DQN {game_cycle + 1}/{NUM_DQN_CYCLES}...")

        processes = []
        parent_conns = []
        
        ctx = mp.get_context('spawn')
        
        for ag in batch_agents:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=game_worker, args=(child, ag.color))
            p.start()
            processes.append(p)
            parent_conns.append(parent)
        current_states = [conn.recv() for conn in parent_conns]
        active_mask = [True] * len(batch_agents)
        
        while any(active_mask):
            for i, agent in enumerate(batch_agents):
                if not active_mask[i]:
                    continue
                
                conn = parent_conns[i]
                state_old = current_states[i]

                final_move = agent.get_action(state_old)
                
                conn.send(final_move)
                
                next_state, reward, done, score = conn.recv()

                agent.train_short_memory(state_old, final_move, reward, next_state, done)
                agent.remember(state_old, final_move, reward, next_state, done)
                
                if done:
                    agent.score = score
                    agent.n_games += 1
                    
                    agent.train_long_memory()
                    
                    conn.send("CLOSE")
                    active_mask[i] = False
                else:
                    current_states[i] = next_state

        for p in processes:
            p.join()

    for ag in batch_agents:
        ag.to_cpu()


def train():
    num_agents = 10
    agents = [Agent() for _ in range(num_agents)]
    #for i in range(len(agents)):
    #    agents[i].model.load_state_dict(torch.load(f"/home/pedro/Documents/Sistemas Evolutivos/EVOL_project/hope_models/best_model_{i}_450.pt"))
    #    print(i, type(agents[i]))
    
    epoch = 0
    record = 0
    mutation_percentage = 0.2
    
    while True:
        
        print(f"--- Epoch {epoch} (AE Cycle) ---")
        
        for i in range(0, num_agents, SIMULTANEOUS_GAMES):
            batch = agents[i : i + SIMULTANEOUS_GAMES]
            print(f"Treinando Batch {i} a {min(i+SIMULTANEOUS_GAMES, num_agents)}...")
            run_batch_of_agents(batch, epoch) 

        scores = [a.score for a in agents]
        max_score = max(scores)
        if max_score > record:
            record = max_score
        
        print(f"Melhor Score: {max_score}, Média: {np.mean(scores)}")

        if epoch % 50 == 0 and epoch != 0:
            for i, agent in enumerate(agents):
                state_dict = agent.model.state_dict()
                #torch.save(state_dict, f"/home/pedro/Documents/Sistemas Evolutivos/EVOL_project/hope_models/best_model_{i}_{epoch + 400}.pt")

        epoch += 1

        
        agents_dict = {i: agent.score for i, agent in enumerate(agents)}
        evol_algorithm = EvolutiveAlgorithm(k=K)
        index_best, pairs, index_for_mutation = evol_algorithm.selection_mechanism(agents_dict=agents_dict)

        new_generation = []
        all_index = np.concatenate((index_best, index_for_mutation), axis=0)
        all_index = [int(item) for item in all_index]

        # Reprodução
        for pair in pairs:
            snakes = [agents[pair[0]], agents[pair[1]]]
            baby_dict = evol_algorithm.reproduction(snakes)
            baby = Agent()
            baby.model.load_state_dict(baby_dict)
            new_generation.append(baby)

        
            # Mutação
        for i in all_index:
            if epoch < 0:
                gaussian = 'small' if i < K else 'big'
                mutated_agent = evol_algorithm.mutation(gaussian=gaussian, snake=agents[i], mutation_percentage=mutation_percentage)
                new_generation.append(mutated_agent)
            else:
                new_generation.append(agents[i])
        
        # Novos Aleatórios
        while len(new_generation) < num_agents:
            new_generation.append(Agent())
            
        agents = new_generation
        mutation_percentage =  mutation_percentage - 0.01


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    train()