# Student agent: Add your own agent here
import time
from copy import deepcopy
from random import choice

import numpy as np
from math import sqrt, log
from agents.agent import Agent
from store import register_agent
import sys


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.count = 0
        self.name = "StudentAgent"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.tree = None

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer
        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.
        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # # dummy return
        # dim = len(chess_board[0])
        # M = Monte(chess_board, my_pos, adv_pos, max_step)
        # steps = M.get_all_steps(chess_board, my_pos, adv_pos, max_step, dim)
        # root = Node(None, None, 0, 0, None)
        # for i in range(0, len(steps)):
        #     node = Node(None, root, 0, 0, steps[i])
        #     root.children[node.move] = node
        #     node.parent = root
        # my_pos, direction = M.UCT_search(root, chess_board, adv_pos, max_step, self.count)
        # self.count = 1
        # # print(steps)
        # # my_pos, direction = M.MTCL(chess_board, my_pos, adv_pos, max_step, steps)
        #
        # # return my_pos, self.dir_map["u"]
        # return my_pos, direction
        #
        if self.tree is None:
            state = State(chess_board, my_pos, adv_pos, max_step, 0)
            self.tree = UctMctsAgent(state)
            self.tree.search(20)
            pos, dire = self.tree.best_move()
            self.tree.root_state.play((pos, dire))
            self.tree.root = self.tree.root.children[(pos, dire)]
            return pos, dire
        else:
            d = -1
            for i in range(4):
                if self.tree.root_state.chess_board[adv_pos[0]][adv_pos[1]][i] != chess_board[adv_pos[0]][adv_pos[1]][i]:
                    d = i
                    break
            if d == -1:
                print("impossible")
            else:
                if self.tree.root.children.get((adv_pos, d)) is None:
                    print("not inherent from parent, creating new tree")
                    state = State(chess_board, my_pos, adv_pos, max_step, 0)
                    self.tree.set_gamestate(state)
                    self.tree.search(2)
                    pos, dire = self.tree.best_move()
                    self.tree.root_state.play((pos, dire))
                    self.tree.root = self.tree.root.children[(pos, dire)]
                    return pos, dire
                else:
                    print("inherent from parent")
                    self.tree.root_state.play((adv_pos, d))
                    self.tree.root = self.tree.root.children[(adv_pos, d)]
                    self.tree.search(2)
                    pos, dire = self.tree.best_move()
                    self.tree.root_state.play((pos, dire))
                    self.tree.root = self.tree.root.children[(pos, dire)]
                    return pos, dire


class Node:
    """
    Node for the MCTS. Stores the move applied to reach this node from its parent,
    stats for the associated game position, children, parent and outcome
    (outcome==none unless the position ends the game).
    Args:
        move:
        parent:
        N (int): times this position was visited.
        Q (int): average reward (wins-losses) from this position.
        Q_RAVE (int): will be explained later.
        N_RAVE (int): will be explained later.
        children (dict): dictionary of successive nodes.
        outcome (int): If node is a leaf, then outcome indicates
                       the winner, else None.
    """

    def __init__(self, move: tuple = None, parent: object = None):
        """
        Initialize a new node with optional move and parent and initially empty
        children list and rollout statistics and unspecified outcome.
        """
        self.move = move
        self.parent = parent
        self.N = 0  # times this position was visited
        self.Q = 0  # average reward (wins-losses) from this position
        self.N_RAVE = 0
        self.Q_RAVE = 0
        self.children = {}
        self.outcome = None

    def add_children(self, children: dict) -> None:
        """
        Add a list of nodes to the children of this node.
        """
        for child in children:
            self.children[child.move] = child

    @property
    def value(self):
        """
        Calculate the UCT value of this node relative to its parent, the parameter
        "explore" specifies how much the value should favor nodes that have
        yet to be thoroughly explored versus nodes that seem to have a high win
        rate.
        """
        # if the node is not visited, set the value as infinity. Nodes with no visits are on priority
        # (lambda: print("a"), lambda: print("b"))[test==true]()
        if self.N == 0:
            return float('inf')
        else:
            return self.Q / self.N + sqrt(2 * log(self.parent.N) / self.N)  # exploitation + exploration


class UctMctsAgent:
    """
    Basic no frills implementation of an agent that preforms MCTS for hex.
    Attributes:
        root_state (GameState): Game simulator that helps us to understand the game situation
        root (Node): Root of the tree search
        run_time (int): time per each run
        node_count (int): the whole nodes in tree
        num_rollouts (int): The number of rollouts for each search
    """

    def __init__(self, state):
        self.root_state = deepcopy(state)
        self.root = Node()
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0

    def search(self, time_budget: int) -> None:
        """
        Search and update the search tree for a
        specified amount of time in seconds.
        """
        start_time = time.time()
        num_rollouts = 0

        # do until we exceed our time budget
        while time.time() - start_time < time_budget:
            node, state = self.select_node()
            turn = state.turn
            outcome = self.roll_out(state)
            self.backup(node, turn, outcome)
            num_rollouts += 1
        run_time = time.time() - start_time
        node_count = self.tree_size()
        self.run_time = run_time
        self.node_count = node_count
        self.num_rollouts = num_rollouts
        print(run_time, node_count, num_rollouts)

    def select_node(self) -> tuple:
        """
        Select a node in the tree to preform a single simulation from.
        """
        node = self.root
        state = deepcopy(self.root_state)

        # stop if we find reach a leaf node
        while len(node.children) != 0:
            # descend to the maximum value node, break ties at random
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value).value
            max_nodes = [n for n in node.children.values()
                         if n.value == max_value]
            node = choice(max_nodes)
            state.play(node.move)
            # if some child node has not been explored select it before expanding
            # other children
            if node.N == 0:
                return node, state
        # if we reach a leaf node generate its children and return one of them
        # if the node is terminal, just return the terminal node
        if self.expand(node, state):
            node = choice(list(node.children.values()))
            state.play(node.move)
        return node, state

    def tree_size(self) -> int:
        """
        Count nodes in tree by BFS.
        """
        Q = []
        count = 0
        Q.append(self.root)
        while len(Q) != 0:
            node = Q.pop(0)
            count += 1
            for child in node.children.values():
                Q.append(child)
        return count

    @staticmethod
    def expand(parent: Node, state) -> bool:
        """
        Generate the children of the passed "parent" node based on the available
        moves in the passed gamestate and add them to the tree.
        Returns:
            bool: returns false If node is leaf (the game has ended).
        """
        children = []
        if state.check_endgame()[0]:
            # game is over at this node so nothing to expand
            return False
        for move in state.get_all_steps():
            children.append(Node(move, parent))
        parent.add_children(children)
        return True

    @staticmethod
    def roll_out(state) -> int:
        """
        Simulate an entirely random game from the passed state and return the winning
        player.
        Args:
            state: game state
        Returns:
            int: winner of the game
        """
        res = state.check_endgame()
        while not res[0]:
            move = random_step(state.chess_board, state.my_pos, state.adv_pos, state.max_step)
            state.play(move)
            res = state.check_endgame()
        if res[1] != -1 and res[1] != 0:
            return state.turn
        else:
            return 1 - state.turn

    @staticmethod
    def backup(node: Node, turn: int, outcome: int) -> None:
        """
        Update the node statistics on the path from the passed node to root to reflect
        the outcome of a randomly simulated playout.
        Args:
            node:
            turn: winner turn
            outcome: outcome of the rollout
        Returns:
            object:
        """
        # Careful: The reward is calculated for player who just played
        # at the node and not the next player to play
        reward = 0 if outcome == turn else 1

        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent
            reward = 0 if reward == 1 else 1

    def best_move(self) -> tuple:
        """
        Return the best move according to the current tree.
        Returns:
            best move in terms of the most simulations number unless the game is over
        """
        if self.root_state.check_endgame()[0]:
            return None
        # choose the move of the most simulated node breaking ties randomly
        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        print(max_value)
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        bestchild = choice(max_nodes)
        return bestchild.move

    def set_gamestate(self, state) -> None:
        """
        Set the root_state of the tree to the passed gamestate, this clears all
        the information stored in the tree since none of it applies to the new
        state.
        """
        self.root_state = deepcopy(state)
        self.root = Node()

    def statistics(self) -> tuple:
        return self.num_rollouts, self.node_count, self.run_time


# class Node:
#     def __init__(self, child, parent, score, times, move):
#         self.children = {}
#         self.parent = parent
#         self.score = score
#         self.times = times
#         self.move = move
#
#     def cal_UCT(self, explore: float = 0.5):
#         if self.times == 0:
#             return 0 if explore == 0 else float('inf')
#         else:
#             # print(self.score)
#             # print(self.times)
#             # print(self.parent.times)
#
#             return self.score / self.times + explore * sqrt(2) * sqrt(log(self.parent.times) / self.times)


class State:
    def __init__(self, chess_board, my_pos, adv_pos, max_step, turn):
        # self.board_size = board_dim = len(chess_board[0])
        self.count = 0
        self.max_step = max_step
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.setup = True
        self.wins = {}
        self.plays = {}
        self.num = 0
        self.turn = turn
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def play(self, move):
        pos, dir = move
        r, c = pos
        self.set_barrier(self.chess_board, r, c, dir)
        self.my_pos = r, c
        tmp = self.adv_pos
        self.adv_pos = self.my_pos
        self.my_pos = tmp
        self.turn = 1 - self.turn

    # def simulation(self):
    #     chess_board = self.chess_board
    #     my_pos = self.my_pos
    #     adv_pos = self.adv_pos
    #     turn = self.turn
    #     max_step = self.max_step
    #     board = deepcopy(chess_board)
    #     board_size = len(board[0])
    #     res = self.check_endgame()
    #     while not res[0]:
    #         if turn == 0:
    #             random = self.random_moves(board, my_pos, adv_pos, max_step)
    #             my_pos = random[0]
    #             my_dir = random[1]
    #             self.set_barrier(board, my_pos[0], my_pos[1], my_dir)
    #             # print("here")
    #             turn = 1
    #         elif turn == 1:
    #             random = self.random_moves(board, adv_pos, my_pos, max_step)
    #             adv_pos = random[0]
    #             adv_dir = random[1]
    #             self.set_barrier(board, adv_pos[0], adv_pos[1], adv_dir)
    #             # print("there")
    #             turn = 0
    #         res = self.check_endgame1(board, my_pos, adv_pos, max_step, board_size)
    #         # print(res[0])
    #         # print(my_pos)
    #         # print(adv_pos)
    #     return res[1]

    def get_all_steps(self):
        chess_board = self.chess_board
        my_pos = self.my_pos
        adv_pos = self.adv_pos
        max_step = self.max_step
        dim = len(self.chess_board[0])
        all_valids = []
        for r in range(max(0, my_pos[0] - max_step), min(dim, my_pos[0] + max_step)):
            for c in range(max(0, my_pos[1] - max_step), min(dim, my_pos[1] + max_step)):
                if (abs(my_pos[0] - r) + abs(my_pos[1] - c)) in range(0, max_step + 1):
                    for key in list(self.dir_map):
                        dire = self.dir_map[key]
                        if self.check_valid_step(chess_board, my_pos, (r, c), dire, adv_pos, max_step):
                            all_valids.append(((r, c), dire))
        return all_valids

    def set_barrier(self, chess_board, r, c, dir):
        # chess_board = self.chess_board
        chess_board[int(r), int(c), int(dir)] = True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    def check_valid_step(self, chess_board, start_pos, end_pos, barrier_dir, adv_pos, max_step):
        """
        Check if the step the agent takes is valid (reachable and within max steps).
        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # Get position of the adversary
        # adv_pos = self.p0_pos if self.turn else self.p1_pos

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            # print(cur_pos)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue

                # next_pos = cur_pos + move
                next_pos = cur_pos[0] + move[0], cur_pos[1] + move[1]
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    def random_moves(self, chess_board, my_pos, adv_pos, max_step):
        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)
        # print(chess_board)

        # Random Walkpython simulator.py --player_1 student_agent --player_2 random_agent
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

    def check_endgame(self):
        """
        Check if the game ends and compute the current score of the agents.
        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        chess_board = self.chess_board
        my_pos = self.my_pos
        adv_pos = self.adv_pos
        board_size = len(self.chess_board[0])
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            # print(board_size)
            # print("fa pos", father[pos])
            # print("pos", pos)
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                        self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            # print(1)
            return False, 0
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
            return True, 0

        if (p0_score - p1_score) < 0:
            return True, -1
        else:
            return True, 1

    def check_endgame1(self, chess_board, my_pos, adv_pos, max_step, board_size):
        """
        Check if the game ends and compute the current score of the agents.
        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """

        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            # print(board_size)
            # print("fa pos", father[pos])
            # print("pos", pos)
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                        self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            # print(1)
            return False, 0
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
            return True, 0

        if (p0_score - p1_score) < 0:
            return True, -1
        else:
            return True, 1


def random_step(chess_board, my_pos, adv_pos, max_step):
    # Moves (Up, Right, Down, Left)
    ori_pos = deepcopy(my_pos)
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    steps = np.random.randint(0, max_step + 1)

    # Random Walk
    for _ in range(steps):
        r, c = my_pos
        dir = np.random.randint(0, 4)
        m_r, m_c = moves[dir]
        my_pos = (r + m_r, c + m_c)

        # Special Case enclosed by Adversary
        k = 0
        while chess_board[r, c, dir] or my_pos == adv_pos:
            k += 1
            if k > 300:
                break
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

        if k > 300:
            my_pos = ori_pos
            break

    # Put Barrier
    dir = np.random.randint(0, 4)
    r, c = my_pos
    while chess_board[r, c, dir]:
        dir = np.random.randint(0, 4)

    return my_pos, dir
