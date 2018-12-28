import MCST
import reversi
board, color = reversi.init_board()
root = MCST.Node(board, color, None)
MCST.MCTS(root, 100)
MCST.draw_tree(root)
pass