import random
from shutil import move
import sys
from game import Game
import numpy as np
import pygame
from agent import Agent, LegacyAlphaBeta, RandomAgent, AlphaBetaAgent, SARSA_FeatureAgent
from tqdm import tqdm
NUM_ROWS, NUM_COLS, NUM_EPISODES = 6, 7, 1000
import matplotlib.pyplot as plt

def SarsaLambda(
    game: Game, # connect-4 game
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:SARSA_FeatureAgent,
    Opponent:Agent,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(game,done,w,epsilon=0.2):
        nA = game.col_count
        Q = [np.dot(w, X(game,done,a)) for a in range(nA)]
        if np.random.rand() < epsilon:
            valid_moves = game.get_valid_moves()
            if len(valid_moves) == 0:
                return None
            move = random.choice(valid_moves)
            return move
        else:
            act = np.argmax(Q)
            while not game.is_valid_location(act):
                Q[act] = np.NINF
                act = np.argmax(Q)
                if Q[act] == np.NINF:
                    return None
            return act
    
    def update(w, x, z, Q_old, lam, gamma, alpha, observation, done, reward, action_prime, X:SARSA_FeatureAgent):
        x_prime = X(observation, done, action_prime)
        Q = np.dot(w,x)
        Q_prime = np.dot(w, x_prime)
        delta = reward + gamma * Q_prime - Q
        z = lam * gamma * z + (1-(alpha * gamma * lam * np.dot(z,x))) * x
        w = w + alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
        return w, z, Q_prime, x_prime

    diagnostic_file = open("Learning_Stats.txt", "w")
    policy_performance = []

    w = np.zeros((X.feature_vector_len() * game.col_count), dtype=np.float32)
    explor = .1
    e_decay = .0035
    lr_decay = .01

    win_count = 0
    interval_wins = 0.0
    interval_len = 10.0
    
    for ep in tqdm(range(num_episode)):
        observation = Game(NUM_ROWS, NUM_COLS) #change game reset'
        #add opponent action if SARSA agent is p2 / alternate initiative each episode
        # if np.random.rand() < .5:
        #     opponent_action = Opponent.get_action(observation)
        #     observation.drop_piece(opponent_action, Opponent.color)
        flipped_board = np.fliplr(np.copy(observation.board))
        flipped_observation = Game(NUM_ROWS, NUM_COLS, flipped_board)

        done = len(game.get_valid_moves()) == 0
        
        action = epsilon_greedy_policy(observation, done, w, explor)
        flipped_action =  game.col_count - 1 - action
        
        x = X(observation, done, action)
        flipped_x = X(flipped_observation, done, flipped_action)
        
        z = np.zeros(X.feature_vector_len() * observation.col_count)
        flipped_z = np.copy(z)
        
        Q_old = 0
        flipped_Q_old = 0

        lr = alpha / (1.0 + ep * lr_decay)

        win = 0
        if ep % interval_len == 0:
            interval_wins = 0.0

        move_count = 0
        while not done:
            observation, reward, done, win = step(observation, action, Opponent)
            win_count += win
            # interval_wins += win
            action_prime = epsilon_greedy_policy(observation, done, w, epsilon=explor)
            if action_prime == None:
                break
            flipped_action_prime = game.col_count - 1 - action_prime
            flipped_board = np.fliplr(np.copy(observation.board))
            flipped_observation = Game(NUM_ROWS, NUM_COLS, flipped_board)
            
            w, z, Q_old, x = update(w, x, z, Q_old, lam, gamma, lr, observation, done, reward, action_prime, X)
            w, flipped_z, flipped_Q_old, flipped_x = update(w, flipped_x, flipped_z, flipped_Q_old, lam, gamma, lr, flipped_observation, done, reward, flipped_action_prime, X)

            action = action_prime
            move_count += 1

        # explor = explor * (1-e_decay)
        X.set_weights(w)
        res, eval_moves = play_game(X, Opponent)
        if res == 1:
            interval_wins += 1
        performance = res * (21 - eval_moves) / 17
        policy_performance.append(performance)
        if ep % interval_len == interval_len - 1:
            msgs = ["interval end at ep:" + str(ep) + ",  win %: " + str(interval_wins / interval_len) + "\n"]
            msgs.append("result: " + str(res) + " in " + str(eval_moves) + " moves\n")
            diagnostic_file.writelines(msgs)

    # X.set_weights(w)
    print("win count: ", win_count)
    np.save("out.npy", w)
    diagnostic_file.close()

    smooth_performance = []
    for i in range(len(policy_performance) // 10):
        smooth_performance.append(np.mean(policy_performance[i*10: i*10+10]))
    
    plt.figure()
    plt.plot(range(1, num_episode // 10 + 1), smooth_performance)
    plt.title("Performance of policy during training")
    plt.xlabel("Trial number")
    plt.ylabel("Policy performance")
    plt.savefig('performance_graph.png')

    # print(boards)
    return w

def step(game : Game, action, opponent : Agent):
    win_count = 0
    if action == None:
        return game, 0.0, True, win_count
    player_color = opponent.opponent_color
    game.drop_piece(action, player_color)
    reward = 0.0
    if game.winning_move_faster(player_color):
        reward = 1.0
        win_count += 1
        return game, reward, True, win_count
    opponent_action = opponent.get_action(game)
    #tie
    if opponent_action == None:
        return game, 0.0, True, win_count
    game.drop_piece(opponent_action, opponent.color)

    if game.winning_move_faster(opponent.color):
        reward = -1.0
        return game, reward, True, win_count
    return game, 0.0, len(game.get_valid_moves()) == 0, win_count

#return 1 for p1 win, -1 for p2 win, 0 for draw
def play_game(p1 : Agent, p2 : Agent, draw = False):
    game = Game(NUM_ROWS, NUM_COLS)
    move_count = 0
    while True:
        move_count += 1
        #p1 turn
        p1_move = p1.get_action(game)
        game.drop_piece(p1_move, 1)
        if draw: game.print_board()
        if game.winning_move_faster(1):
            return 1, move_count
        if len(game.get_valid_moves()) == 0:
            return 0, move_count
        #p2 turn
        p2_move = p2.get_action(game)
        game.drop_piece(p2_move, 2)
        if draw: game.print_board()
        if game.winning_move_faster(2):
           return -1, move_count
        if len(game.get_valid_moves()) == 0:
            return 0, move_count
