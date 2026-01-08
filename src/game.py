import numpy as np
from typing import List, Tuple, Optional


class Game81:
    def __init__(self):
        self.board = np.zeros((9, 9), dtype=np.int8)
        self.big_board = np.zeros(9, dtype=np.int8)
        self.current_player = 1
        self.last_move = None
        self.move_count = 0
    
    def copy(self) -> 'Game81':
        game = Game81()
        game.board = self.board.copy()
        game.big_board = self.big_board.copy()
        game.current_player = self.current_player
        game.last_move = self.last_move
        game.move_count = self.move_count
        return game
    
    def _get_big_cell_index(self, row: int, col: int) -> int:
        return (row // 3) * 3 + (col // 3)
    
    def _get_local_position(self, row: int, col: int) -> Tuple[int, int]:
        return row % 3, col % 3
    
    def _check_line_win(self, cells: np.ndarray, player: int) -> bool:
        for i in range(3):
            if np.all(cells[i, :] == player) or np.all(cells[:, i] == player):
                return True
        if np.all(np.diag(cells) == player) or np.all(np.diag(np.fliplr(cells)) == player):
            return True
        return False
    
    def _check_big_cell_status(self, big_idx: int) -> int:
        start_row = (big_idx // 3) * 3
        start_col = (big_idx % 3) * 3
        cells = self.board[start_row:start_row+3, start_col:start_col+3]
        
        for player in [1, 2]:
            if self._check_line_win(cells, player):
                return player
        
        if np.all(cells != 0):
            return 3
        return 0
    
    def _get_target_big_cell(self) -> Optional[int]:
        if self.last_move is None:
            return None
        local_row, local_col = self._get_local_position(*self.last_move)
        target = local_row * 3 + local_col
        if self.big_board[target] != 0:
            return None
        return target
    
    def get_valid_moves(self) -> List[int]:
        valid = []
        target_big = self._get_target_big_cell()
        
        if target_big is not None:
            start_row = (target_big // 3) * 3
            start_col = (target_big % 3) * 3
            for r in range(3):
                for c in range(3):
                    row, col = start_row + r, start_col + c
                    if self.board[row, col] == 0:
                        valid.append(row * 9 + col)
        else:
            for big_idx in range(9):
                if self.big_board[big_idx] != 0:
                    continue
                start_row = (big_idx // 3) * 3
                start_col = (big_idx % 3) * 3
                for r in range(3):
                    for c in range(3):
                        row, col = start_row + r, start_col + c
                        if self.board[row, col] == 0:
                            valid.append(row * 9 + col)
        return valid
    
    def make_move(self, action: int) -> bool:
        row, col = action // 9, action % 9
        if self.board[row, col] != 0:
            return False
        
        big_idx = self._get_big_cell_index(row, col)
        if self.big_board[big_idx] != 0:
            return False
        
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.move_count += 1
        
        status = self._check_big_cell_status(big_idx)
        if status != 0:
            self.big_board[big_idx] = status
        
        self.current_player = 3 - self.current_player
        return True
    
    def check_winner(self) -> int:
        big_grid = self.big_board.reshape(3, 3)
        for player in [1, 2]:
            if self._check_line_win(big_grid, player):
                return player
        
        if np.all(self.big_board != 0):
            p1_count = np.sum(self.big_board == 1)
            p2_count = np.sum(self.big_board == 2)
            if p1_count > p2_count:
                return 1
            elif p2_count > p1_count:
                return 2
            else:
                return 2
        
        if len(self.get_valid_moves()) == 0:
            p1_count = np.sum(self.big_board == 1)
            p2_count = np.sum(self.big_board == 2)
            if p1_count > p2_count:
                return 1
            elif p2_count > p1_count:
                return 2
            else:
                return 2
        
        return 0
    
    def is_terminal(self) -> bool:
        return self.check_winner() != 0 or len(self.get_valid_moves()) == 0
    
    def get_state(self) -> np.ndarray:
        state = np.zeros((6, 9, 9), dtype=np.float32)
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == 3 - self.current_player).astype(np.float32)
        
        for i in range(9):
            start_row = (i // 3) * 3
            start_col = (i % 3) * 3
            if self.big_board[i] == self.current_player:
                state[2, start_row:start_row+3, start_col:start_col+3] = 1
            elif self.big_board[i] == 3 - self.current_player:
                state[3, start_row:start_row+3, start_col:start_col+3] = 1
        
        valid_moves = self.get_valid_moves()
        for move in valid_moves:
            row, col = move // 9, move % 9
            state[4, row, col] = 1
        
        if self.current_player == 1:
            state[5] = 1
        
        return state
    
    def get_canonical_state(self) -> np.ndarray:
        return self.get_state()
    
    def action_to_string(self, action: int) -> str:
        row, col = action // 9, action % 9
        big_names = ['LU', 'U', 'RU', 'L', 'M', 'R', 'LD', 'D', 'RD']
        big_idx = self._get_big_cell_index(row, col)
        local_row, local_col = self._get_local_position(row, col)
        return f"{big_names[big_idx]}({local_row},{local_col})"
    
    def display(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        big_symbols = {0: ' ', 1: 'X', 2: 'O', 3: '#'}
        
        print("\n  ", end="")
        for c in range(9):
            if c % 3 == 0 and c > 0:
                print("|", end="")
            print(f"{c}", end="")
        print()
        print("  " + "-" * 11)
        
        for r in range(9):
            if r % 3 == 0 and r > 0:
                print("  " + "-" * 11)
            print(f"{r}|", end="")
            for c in range(9):
                if c % 3 == 0 and c > 0:
                    print("|", end="")
                print(symbols[self.board[r, c]], end="")
            print("|")
        
        print("\n大格状态:", end=" ")
        for i in range(9):
            print(f"{big_symbols[self.big_board[i]]}", end=" ")
        print(f"\n当前玩家: {symbols[self.current_player]}")
