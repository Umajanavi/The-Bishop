import chess
import numpy as np
from src.franka_emika import *
from src.chess_bot import *
from src.canvas import *
from src.motion_planner import *
from src.vision import *
from src.types import *
from src.board import *

chess_bot = ChessBot()
canvas = Canvas()

result_cmd = Command("r", "reset")
continue_cmd = Command(" ", "continue")
go_to_saved_cmd = Command("g", "go to saved position")
end_turn_cmd = Command(" ", "end turn")
look_again = Command(" ", "look again")
retry_cmd = Command(" ", "retry")

def wait_for_retry(error):
    canvas.set_text(f"Error moving to pose: {error}")
    canvas.set_commands([retry_cmd])
    canvas.draw()
    canvas.wait_for_command()

robot = FrankaEmika(wait_for_retry)
planner = MotionPlanner(robot)
vision = Vision()


def set_viewing_pose():
    vision.set_autofocus_locked(False)

    while True:
        state = vision.detect_state(chess.Board())
        board_visible = state.board is not None
        has_saved_pose = planner.has_saved_viewing_pose()

        commands = []
        if board_visible:
            commands.append(continue_cmd)
        if has_saved_pose:
            commands.append(go_to_saved_cmd)

        canvas.set_text("Line up the board")
        canvas.set_frame(state.frame)
        canvas.set_commands(commands)
        canvas.draw()
        command = canvas.read_command()

        if command == go_to_saved_cmd:
            canvas.set_text("Moving to saved position...")
            canvas.draw()
            planner.move_to_saved_viewing_pose()
        elif command == continue_cmd:
            planner.set_viewing_pose()
            planner.save_viewing_pose()
            vision.set_autofocus_locked(True)
            return state

def get_state(old_board, is_setup=False):
    while True:
        planner.look_at_board()
        state = vision.detect_state(old_board)

        if state.board is None:
            set_viewing_pose()
        elif len(state.pieces) != TOTAL_PIECES:
            canvas.set_text(f"Only {len(state.pieces)}/{TOTAL_PIECES} pieces visible")
            canvas.set_frame(state.frame)
            canvas.set_commands([])
            canvas.draw()
        elif is_setup:
            return state
        else:
            move = find_legal_move(old_board, state.board)
            if move is not None:
                game_board = old_board.copy()
                game_board.push(move)
                return state._replace(board=game_board)

            canvas.set_text("Illegal move")
            canvas.set_detected_board(state.board)
            canvas.set_frame(state.frame)
            canvas.set_commands([look_again, result_cmd])
            canvas.draw()
            if canvas.wait_for_command() == result_cmd:
                return None

def set_board(state, target_board, title):
    canvas.set_text(f"{title} - planning")
    canvas.set_board(target_board)
    canvas.set_detected_board(state.board)
    canvas.set_frame(state.frame)
    canvas.draw()

    while True:
        transfer = planner.plan_transfer(state, target_board)
        if transfer is None:
            return state._replace(board=target_board)

        canvas.set_text(f"{title} - moving")
        for configuration in planner.make_transfer(state, transfer):
            canvas.set_robot_configuration(configuration, planner.camera_to_base)
            canvas.draw()

        state = get_state(state.board, is_setup=True)
        canvas.set_text(f"{title} - planning")
        canvas.set_detected_board(state.board)
        canvas.set_frame(state.frame)
        canvas.draw()

def get_outcome_message(board: chess.Board):
    outcome = board.outcome(claim_draw=True)

    if outcome.winner == chess.WHITE:
        return "The Bishop won"
    if outcome.winner == chess.BLACK:
        return "You won"

    return "It's a draw"


def take_turns(state):
    while not is_game_over(state.board):
        state.board.turn = chess.BLACK
        canvas.set_text("Your turn")
        canvas.set_board(state.board)
        canvas.set_detected_board(state.board)
        canvas.set_frame(state.frame)
        canvas.set_commands([end_turn_cmd, result_cmd])
        canvas.draw()
        command = canvas.wait_for_command()
        if command == result_cmd:
            return

        state = get_state(state.board)
        if state is None:
            return

        if is_game_over(state.board):
            break

        canvas.set_text("The Bishop's turn - thinking")
        canvas.set_board(state.board)
        canvas.set_detected_board(state.board)
        canvas.set_frame(state.frame)
        canvas.draw()

        state = set_board(state, chess_bot.play(state.board), "The Bishop's turn")

    canvas.set_text(get_outcome_message(state.board))
    canvas.set_board(state.board)
    canvas.set_frame(state.frame)
    canvas.set_commands([result_cmd])
    canvas.draw()
    canvas.wait_for_command()


if __name__ == "__main__":
    try:
        state = set_viewing_pose()
        state = get_state(chess.Board(), is_setup=True)

        while True:
            take_turns(state)
            state = get_state(chess.Board(), is_setup=True)
            state = set_board(state, chess.Board(), "Resetting the board")

    except Quit:
        pass

    finally:
        canvas.quit()
        vision.quit()
        chess_bot.quit()
        robot.quit()
