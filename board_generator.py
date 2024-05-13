from gym_sokoban.envs.room_utils import SokobanRoomGenerator
from gym_sokoban.envs.sokoban_env import SokobanEnv
from tqdm import tqdm
import numpy as np
import joblib
import argparse

def main(num_boxes, n_boards: int):
    env = SokobanEnv(dim_room=(12, 12), num_boxes=num_boxes, mode='one_hot')
    boards = []
    for _ in tqdm(range(n_boards)):
        board = env.reset()
        boards.append(board)
        
    boards = np.array(boards)
    assert boards.shape == (n_boards, 12, 12, 7), boards.shape

    joblib.dump(boards, f'boards_1k_b{num_boxes}.joblib')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--boxes', type=int, default=3)
    parser.add_argument('--n-boards', type=int, default=10)
    args = parser.parse_args()
    main(args.boxes, args.n_boards)
