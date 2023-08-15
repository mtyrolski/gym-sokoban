import copy
from ctypes import c_double, c_int, c_uint, c_bool, cdll, byref, POINTER
import hashlib
import struct
from typing import List, Optional, Union
import pkg_resources
import numpy as np
import numpy.ctypeslib as npct
import os


from sys import platform

if platform == "linux" or platform == "linux2":
    try:
      ext = 'so'
      lib_filename = pkg_resources.resource_filename(__name__, './room_utils_fast.' + ext)
      lib = cdll.LoadLibrary(lib_filename)
    except:
      #May help with verions problems

      print("loading failed. Try to recompile")
      lib_path = os.path.dirname(os.path.abspath(__file__))
      command = "cd {};g++ -std=c++11 -c -fPIC -O3 room_utils_fast.cpp -o room_utils_fast.o".format(lib_path)
      os.system(command)
      print("compilation done")
      command = "cd {};g++ -shared -Wl,-soname,room_utils_fast.so -o room_utils_fast.so room_utils_fast.o".format(lib_path)
      os.system(command)
      print("shared library created")
      # Try once more
      lib_filename = pkg_resources.resource_filename(__name__, './room_utils_fast.' + ext)
      lib = cdll.LoadLibrary(lib_filename)
      print("loading second time")

elif platform == "darwin":
    ext = 'dylib'
    lib_filename = pkg_resources.resource_filename(__name__, './room_utils_fast.' + ext)
    lib = cdll.LoadLibrary(lib_filename)

lib.generate_room.argtypes = [npct.ndpointer(dtype=np.uint8, ndim=1),  # room
                              npct.ndpointer(dtype=np.int32, ndim=1),  # dims
                              c_double,  # p_change_directions
                              c_int,  # num_steps
                              c_int,  # num_boxes
                              c_int,  # tries
                              POINTER(c_uint),  # seed
                              c_bool,  # do_reverse_playing
                              c_bool,
                              c_int]  # second_player



def hash_seed(seed: Optional[int] = None, max_bytes: int = 8) -> int:
    """Any given evaluation is likely to have many PRNG's active at once.
    (Most commonly, because the environment is running in multiple processes.)
    There's literature indicating that having linear correlations between seeds of multiple PRNG's can correlate the outputs:
        http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
        http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
        http://dl.acm.org/citation.cfm?id=1276928
    Thus, for sanity we hash the seeds before using them. (This scheme is likely not crypto-strength, but it should be good enough to get rid of simple correlations.)
    Args:
        seed: None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the hashed seed.
    Returns:
        The hashed seed
    """

    if seed is None:
        seed = create_seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode("utf8")).digest()
    return _bigint_from_bytes(hash[:max_bytes])


def create_seed(a: Optional[Union[int, str]] = None, max_bytes: int = 8) -> int:
    """Create a strong random seed.
    Otherwise, Python 2 would seed using the system time, which might be non-robust especially in the presence of concurrency.
    Args:
        a: None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the seed.
    Returns:
        A seed
    Raises:
        Error: Invalid type for seed, expects None or str or int
    """

    # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, str):
        bt = a.encode("utf8")
        bt += hashlib.sha512(bt).digest()
        a = _bigint_from_bytes(bt[:max_bytes])
    elif isinstance(a, int):
        a = int(a % 2 ** (8 * max_bytes))
    else:
        raise Exception(f"Invalid type for seed: {type(a)} ({a})")

    return a


# TODO: don't hardcode sizeof_int here
def _bigint_from_bytes(bt: bytes) -> int:
    sizeof_int = 4
    padding = sizeof_int - len(bt) % sizeof_int
    bt += b"\0" * padding
    int_count = int(len(bt) / sizeof_int)
    unpacked = struct.unpack(f"{int_count}I", bt)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum


def _int_list_from_bigint(bigint: int) -> List[int]:
    # Special case 0
    if bigint < 0:
        raise Exception(f"Seed must be non-negative, not {bigint}")
    elif bigint == 0:
        return [0]

    ints: List[int] = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2**32)
        ints.append(mod)
    return ints

class SokobanRoomGenerator(object):
    def __init__(self, seed, game_mode=None, verbose=True):
        assert game_mode in ["NoAlice", "Alice", "Magnetic"], "Incorrect game format!"
        self.do_reverse_playing = c_bool(game_mode == "NoAlice" or game_mode == "Magnetic")
        self.seed = c_uint(seed)
        self.verbose = verbose

    def generate_room(self, dim=(13, 13),
                      p_change_directions=0.35,
                      num_steps=25,
                      num_boxes=3,
                      tries=4,
                      second_player=False,
                      curriculum=300):
        """
        Generates a Sokoban room, represented by an integer matrix. The elements are encoded as follows:
        wall = 0
        empty space = 1
        box target = 2
        box not on target = 3
        box on target = 4
        player = 5

        :param dim:
        :param p_change_directions:
        :param num_steps:
        :return: Numpy 2d Array
        """

        room_state = np.zeros(dim[0]*dim[1], dtype=np.uint8)
        dim = np.array(dim, dtype=np.int32)

        seed_ = copy.copy(self.seed)
        score = lib.generate_room(room_state,
                                  dim,
                                  p_change_directions,
                                  num_steps,
                                  num_boxes,
                                  tries,
                                  byref(seed_),
                                  self.do_reverse_playing,
                                  second_player,
                                  c_int(curriculum))

        # rehash the seed returned from generate_room
        self.seed = c_uint(_int_list_from_bigint(hash_seed(self.seed))[1])

        # error codes from generate_room
        # -3: BAD, "Not enough free spots (#%d) to place %d player and %d boxes." (place_box_and_players)
        # -2: BAD, "More boxes (%d) than allowed (%d)!"
        # -1: BAD "Sokoban size (%dx%d) bigger than maximum size (%dx%d)!"
        # score: OK
        if score == -3:
            raise RuntimeError('Not enough free spots to place player and boxes.')
        elif score == -2:
            raise RuntimeError('More boxes ({}) than allowed!'.format(num_boxes))
        elif score == -1:
            raise RuntimeError('Sokoban size ({}x{}) bigger than maximum size!'.format(dim[0], dim[1]))
        elif score == 0:
            if self.do_reverse_playing:
                msg = 'Generated Model with score == 0' if self.verbose else ""
                raise RuntimeWarning(msg)

        room_state = room_state.reshape((dim[0], dim[1]))
        player_on_target = room_state == 6

        room_structure = room_state.copy()
        if player_on_target.any():
            room_state[player_on_target] = 5
            room_structure[player_on_target] = 2
        room_structure[room_structure > 3] = 1
        room_structure[room_structure == 3] = 2

        return room_structure, room_state, {}, np.uint32(self.seed)


