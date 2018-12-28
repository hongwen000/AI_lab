import MCST
import reversi_MCST
board, color = reversi_MCST.init_board()
root = MCST.Node(board, color, None)
MCST.MCTS(root, 100)
MCST.draw_tree(root)
pass