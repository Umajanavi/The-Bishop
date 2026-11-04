import chess


def is_game_over(board: chess.Board):
    return board.is_game_over() or board.can_claim_draw()


def find_legal_move(board_before: chess.Board, board_after: chess.Board):
    for move in board_before.legal_moves:
        board_before.push(move)
        matches = board_before.board_fen() == board_after.board_fen()
        board_before.pop()

        if matches:
            return move

    return None


def diff_board(board: chess.Board, target_board: chess.Board):
    surplus, wanted = [], {}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        target_piece = target_board.piece_at(square)

        if piece == target_piece:
            continue
        if piece is not None:
            surplus.append(square)
        if target_piece is not None:
            wanted[square] = target_piece

    return surplus, wanted


def set_board_pieces(board: chess.Board, pieces):
    board = board.copy()
    board.clear_board()

    for piece in pieces:
        if piece.square is not None:
            board.set_piece_at(piece.square, piece.type)

    return board
