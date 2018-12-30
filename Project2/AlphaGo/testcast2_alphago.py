import AlphaGo.MCST as MCST
import reversi_AlphaGo
board, color = reversi_AlphaGo.init_board()
root = MCST.Node(board, color, None, None)
MCST.MCTS(root, 100)
MCST.draw_tree(root)
pass