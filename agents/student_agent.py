"""Student agent: edit ONLY this file."""

import constants

from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view, LINES)
from constants import *
import numpy as np
import time


_LINE_WEIGHTS = {
    3: (0.0, 2.5, 65.0, 0.0),
    4: (0.0, 1.0, 9.0, 95.0, 0.0),
}
_WIN_SCORE = 1_000_000.0


def _make_move(board, reserves, player, move):
    """Apply a generated legal move."""
    if move.src is None:
        reserves[player, move.size] -= 1
    else:
        sr, sc = move.src
        board[sr, sc, move.size] = 0
    dr, dc = move.dest
    board[dr, dc, move.size] = player


def _undo_move(board, reserves, player, move):
    """Undo a move made by _make_move."""
    dr, dc = move.dest
    board[dr, dc, move.size] = 0
    if move.src is None:
        reserves[player, move.size] += 1
    else:
        sr, sc = move.src
        board[sr, sc, move.size] = player


def _evaluate(board, reserves, perspective, lines):
    """Score promising lines and central control for one player."""
    owners, sizes = top_view(board)
    other = 3 - perspective
    weights = _LINE_WEIGHTS[constants.WIN_LENGTH]
    score = 0.0

    for line in lines:
        mine = theirs = 0
        blocked = False
        for r, c in line:
            owner = int(owners[r, c])
            if owner == 3:
                blocked = True
                break
            if owner == perspective:
                mine += 1
            elif owner == other:
                theirs += 1

        if blocked:
            continue
        if mine and not theirs:
            score += weights[mine]
        elif theirs and not mine:
            score -= weights[theirs]
        elif mine and theirs:
            # Covered pieces can reopen a mixed line, so keep a small residual score.
            score += 0.12 * (weights[mine] - weights[theirs])

    for r in range(4):
        for c in range(4):
            owner = int(owners[r, c])
            if owner not in (1, 2):
                continue
            sign = 1.0 if owner == perspective else -1.0
            if 1 <= r <= 2 and 1 <= c <= 2:
                score += sign * 0.30
            score += sign * (0.04 * int(sizes[r, c]))

    for size, value in ((1, 0.06), (2, 0.14)):
        score += value * (
            int(reserves[perspective, size]) - int(reserves[other, size])
        )

    return score


@register_agent("student_agent")          # do not change this string
class StudentAgent(Agent):
    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"

    def step(self, board, reserves, player, opponent):
        """Return the Move you want to play.

        Takes: board, reserves - your own copies of the current position.
               player, opponent - your player number (1 or 2) and the other one.
        Agent.step in agents/agent.py explains each argument and the time limit.
        """
        start = time.perf_counter()
        # ALLOW_MOVES and WIN_LENGTH (from constants) are the rule flags of this game.
        # helpers.get_valid_moves / check_winner already respect them; your own
        # move generator and evaluation function must too.
        # TODO: replace with your search. Keep (time.perf_counter() - start) < STEP_TIME_LIMIT - margin.

        moves = get_valid_moves(board, reserves, player)
        lines = LINES[constants.WIN_LENGTH]
        best_move = None
        best_score = float("-inf")

        for index, move in enumerate(moves):
            _make_move(board, reserves, player, move)
            try:
                winner = check_winner(board, player)
                if winner == player:
                    return move
                if winner == opponent:
                    score = -_WIN_SCORE
                else:
                    score = _evaluate(board, reserves, player, lines)
            finally:
                _undo_move(board, reserves, player, move)

            if score > best_score:
                best_move, best_score = move, score

            if (index & 7) == 7 and time.perf_counter() - start >= STEP_TIME_LIMIT - 0.20:
                break

        if best_move is None:
            return random_move(board, reserves, player)
        return best_move
