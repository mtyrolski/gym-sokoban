from gym_sokoban.envs.room_utils import SokobanRoomGenerator
from gym_sokoban.envs.sokoban_env import SokobanEnv
from typing import Any
import joblib
from tqdm import tqdm
import numpy as np
import joblib
import argparse

def main(num_boxes, n_boards: int, num_gen_steps: int, curriculum: int, p_change_directions: float):
    env = SokobanEnv(dim_room=(12, 12),
                     num_boxes=num_boxes,
                     mode='one_hot',
                     num_gen_steps=num_gen_steps,
                        curriculum=curriculum,
                        p_change_directions=p_change_directions)
    boards = []
    for _ in tqdm(range(n_boards)):
        board = env.reset()
        boards.append(board)
        
    boards = np.array(boards)
    assert boards.shape == (n_boards, 12, 12, 7), boards.shape

    name = f'boards_{n_boards}_b{num_boxes}_gs{num_gen_steps}_c{curriculum}_p{p_change_directions}.joblib'
    joblib.dump(boards, name)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--boxes', type=int, default=3)
    parser.add_argument('--n-boards', type=int, default=10)
    parser.add_argument('--num-gen-steps', type=int, default=25)
    parser.add_argument('--curriculum', type=int, default=300)
    parser.add_argument('--p-change-directions', type=float, default=0.35)
    parser.add_argument('--n-batches', type=int, default=1)
    args = parser.parse_args()
    
    if args.n_boards % args.n_batches != 0:
        raise ValueError(f'Number of boards {args.n_boards} must be divisible by number of batches {args.n_batches}')
    
    n_board_per_batch = args.n_boards // args.n_batches
    print(f'Generating {args.n_boards} boards in {args.n_batches} batches of {n_board_per_batch} boards each')
    
    joblib.Parallel(n_jobs=-1, verbose=10)(joblib.delayed(main)(num_boxes=args.boxes,
                                                    n_boards=n_board_per_batch,
                                                    num_gen_steps=args.num_gen_steps,
                                                    curriculum=args.curriculum,
                                                    p_change_directions=args.p_change_directions)
                                for _ in range(args.n_batches))

