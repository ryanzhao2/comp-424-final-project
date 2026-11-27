"""Student agent: iterative-deepening alpha-beta search for Goblet Gobblers."""

import time

import constants

from agents.agent import Agent
from constants import ALLOW_MOVES, BOARD_SIZE, MAX_PLIES, OBSTACLE, STEP_TIME_LIMIT, WIN_LENGTH
from helpers import LINES, Move, check_winner, get_valid_moves, top_view
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view, LINES)
from constants import *
import numpy as np


_MATE_SCORE = 1_000_000
_INFINITY = 2_000_000
_EXACT, _LOWER, _UPPER = 0, 1, 2
_WIN_SCORE = _MATE_SCORE
_LINE_WEIGHTS = {
    3: (0.0, 2.5, 65.0, 0.0),
    4: (0.0, 1.0, 9.0, 95.0, 0.0),
}


def _make_move(board, reserves, player, move):
    """Apply a generated legal move, returning enough information to undo it."""
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
    """Score open and partly contested lines from `perspective`'s point of view."""
    owners, sizes = top_view(board)
    other = 3 - perspective
    weights = _LINE_WEIGHTS[constants.WIN_LENGTH]
    score = 0.0

    for line in lines:
        mine = theirs = 0
        blocked = False
        for r, c in line:
            owner = int(owners[r, c])
            if owner == constants.OBSTACLE:
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
            # A piece can be covered, so a mixed line is not completely dead. It is
            # much less promising than an uncontested line, though.
            score += 0.12 * (weights[mine] - weights[theirs])

    # A small positional term breaks ties between otherwise equal line patterns.
    # The four central cells participate in more possible winning lines.
    for r in range(4):
        for c in range(4):
            owner = int(owners[r, c])
            if owner not in (1, 2):
                continue
            sign = 1.0 if owner == perspective else -1.0
            if 1 <= r <= 2 and 1 <= c <= 2:
                score += sign * 0.30
            size = int(sizes[r, c])
            score += sign * (0.04 * size)

    # Retaining larger pieces leaves more ways to cover opposing tops later.
    for size, value in ((1, 0.06), (2, 0.14)):
        score += value * (int(reserves[perspective, size]) - int(reserves[other, size]))

    return score


def _time_check(context, force=False):
    """Raise when the current iterative-deepening pass reaches its time budget."""
    if force or (context["nodes"] & 31) == 0:
        if time.perf_counter() >= context["deadline"]:
            raise TimeoutError


def _ordered_moves(board, reserves, side, moves, maximizing, root_player,
                   ply, lines, context, preferred=None):
    """Order moves by their resulting evaluation, putting a table move first."""
    scored = []
    for index, move in enumerate(moves):
        _make_move(board, reserves, side, move)
        try:
            winner = check_winner(board, side)
            if winner == root_player:
                score = _MATE_SCORE - (ply + 1)
            elif winner != 0:
                score = -_MATE_SCORE + (ply + 1)
            elif ply + 1 >= constants.MAX_PLIES:
                score = 0.0
            else:
                score = _evaluate(board, reserves, root_player, lines)
        finally:
            _undo_move(board, reserves, side, move)

        is_preferred = move == preferred
        scored.append((is_preferred, score, index, move))
        if index & 7 == 7:
            _time_check(context, force=True)

    # Keep the helper's fixed ordering for exact ties.
    scored.sort(key=lambda item: (item[0], item[1] if maximizing else -item[1]), reverse=True)
    return [item[3] for item in scored]


def _alpha_beta(board, reserves, side, depth, alpha, beta, root_player, ply,
                lines, context, table):
    """Return a minimax score with alpha-beta pruning and a depth-aware TT."""
    context["nodes"] += 1
    _time_check(context)

    if ply >= constants.MAX_PLIES:
        return 0.0
    if depth <= 0:
        return _evaluate(board, reserves, root_player, lines)

    key = board.tobytes() + reserves.tobytes() + bytes((side, ply))
    entry = table.get(key)
    preferred = None
    if entry is not None:
        entry_depth, entry_score, entry_bound, preferred = entry
        if entry_depth >= depth:
            if entry_bound == _EXACT:
                return entry_score
            if entry_bound == _LOWER:
                alpha = max(alpha, entry_score)
            else:
                beta = min(beta, entry_score)
            if alpha >= beta:
                return entry_score

    alpha_original, beta_original = alpha, beta

    moves = get_valid_moves(board, reserves, side)
    if not moves:
        if constants.ALLOW_MOVES:
            return (-_MATE_SCORE + ply) if side == root_player else (_MATE_SCORE - ply)

        # With board moves disabled, a player with no placement passes if the
        # opponent still has a move. If neither can place, the game is drawn.
        if not get_valid_moves(board, reserves, 3 - side):
            return 0.0
        if ply + 1 >= constants.MAX_PLIES:
            return 0.0
        return _alpha_beta(board, reserves, 3 - side, depth - 1, alpha, beta,
                           root_player, ply + 1, lines, context, table)

    maximizing = side == root_player
    ordered = _ordered_moves(board, reserves, side, moves, maximizing,
                             root_player, ply, lines, context, preferred)
    best_move = None
    if maximizing:
        best_score = -_INFINITY
        for move in ordered:
            _make_move(board, reserves, side, move)
            try:
                winner = check_winner(board, side)
                if winner == root_player:
                    score = _MATE_SCORE - (ply + 1)
                elif winner != 0:
                    score = -_MATE_SCORE + (ply + 1)
                elif ply + 1 >= constants.MAX_PLIES:
                    score = 0.0
                else:
                    score = _alpha_beta(board, reserves, 3 - side, depth - 1,
                                        alpha, beta, root_player, ply + 1,
                                        lines, context, table)
            finally:
                _undo_move(board, reserves, side, move)
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
    else:
        best_score = _INFINITY
        for move in ordered:
            _make_move(board, reserves, side, move)
            try:
                winner = check_winner(board, side)
                if winner == root_player:
                    score = _MATE_SCORE - (ply + 1)
                elif winner != 0:
                    score = -_MATE_SCORE + (ply + 1)
                elif ply + 1 >= constants.MAX_PLIES:
                    score = 0.0
                else:
                    score = _alpha_beta(board, reserves, 3 - side,
                                        depth - 1, alpha, beta, root_player,
                                        ply + 1, lines, context, table)
            finally:
                _undo_move(board, reserves, side, move)
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, best_score)
            if alpha >= beta:
                break

    if best_score <= alpha_original:
        bound = _UPPER
    elif best_score >= beta_original:
        bound = _LOWER
    else:
        bound = _EXACT
    table[key] = (depth, best_score, bound, best_move)
    return best_score


@register_agent("student_agent")
class StudentAgent(Agent):
    """Choose moves using iterative-deepening minimax with alpha-beta pruning."""

    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "StudentAgent"
        self._turns_taken = 0
        self._rock_announced = False

    def _return_move(self, move, player, ply):
        """Print the selected move before returning it."""
        print(f"{self.name} P{player} ply {ply + 1}: {move}", flush=True)
        return move

    def step(self, board, reserves, player, opponent):
        """Return the Move you want to play.

        Takes: board, reserves - your own copies of the current position.
               player, opponent - your player number (1 or 2) and the other one.
        Agent.step in agents/agent.py explains each argument and the time limit.
        This implementation uses iterative deepening and returns the best move found
        before the per-turn search budget expires.
        """
        start = time.perf_counter()
        # This is the documented ply of this agent's current turn. It also lets
        # the search respect the referee's 100-ply draw limit.
        ply = 2 * self._turns_taken + player - 1
        self._turns_taken += 1

        moves = get_valid_moves(board, reserves, player)

        if not self._rock_announced:
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if board[r, c, 0] == constants.OBSTACLE:
                        print(f"Rock at ({r}, {c})", flush=True)
                        self._rock_announced = True
                        break
                if self._rock_announced:
                    break

        lines = LINES[constants.WIN_LENGTH]
        budget = max(0.05, min(STEP_TIME_LIMIT - 0.20, STEP_TIME_LIMIT * 0.84))
        context = {"deadline": start + budget, "nodes": 0}
        table = {}

        # Score every root action once. Besides improving alpha-beta move order,
        # this immediately returns a win and remembers a safe fallback.
        scored = []
        fallback = moves[0]
        fallback_score = -_INFINITY
        try:
            for index, move in enumerate(moves):
                _make_move(board, reserves, player, move)
                try:
                    winner = check_winner(board, player)
                    if winner == player:
                        return self._return_move(move, player, ply)
                    if winner == opponent:
                        score = -_MATE_SCORE + (ply + 1)
                    elif ply + 1 >= constants.MAX_PLIES:
                        score = 0.0
                    else:
                        score = _evaluate(board, reserves, player, lines)
                finally:
                    _undo_move(board, reserves, player, move)

                if score > fallback_score:
                    fallback, fallback_score = move, score
                scored.append((score, index, move))
                if index & 7 == 7:
                    _time_check(context, force=True)
        except TimeoutError:
            return self._return_move(fallback, player, ply)

        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        ordered_root = [item[2] for item in scored]
        best_move = fallback

        # Keep only fully completed iterations. A timeout in a deeper iteration
        # cannot replace a legal move with a partially searched result.
        depth = 1
        while depth <= constants.MAX_PLIES - ply:
            iteration_best = best_move
            iteration_score = -_INFINITY
            alpha, beta = -_INFINITY, _INFINITY
            try:
                for move in ordered_root:
                    _time_check(context, force=True)
                    _make_move(board, reserves, player, move)
                    try:
                        winner = check_winner(board, player)
                        if winner == player:
                            score = _MATE_SCORE - (ply + 1)
                        elif winner != 0:
                            score = -_MATE_SCORE + (ply + 1)
                        elif ply + 1 >= constants.MAX_PLIES:
                            score = 0.0
                        else:
                            score = _alpha_beta(board, reserves, opponent,
                                                depth - 1, alpha, beta,
                                                player, ply + 1, lines,
                                                context, table)
                    finally:
                        _undo_move(board, reserves, player, move)
                    if score > iteration_score:
                        iteration_score, iteration_best = score, move
                    alpha = max(alpha, iteration_score)
                    if alpha >= beta:
                        break
            except TimeoutError:
                break

            best_move = iteration_best
            # Search the previous iteration's best move first next time.
            ordered_root.remove(best_move)
            ordered_root.insert(0, best_move)
            depth += 1

        return self._return_move(best_move, player, ply)
