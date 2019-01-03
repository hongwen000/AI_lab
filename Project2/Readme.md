# 项目说明
本次项目中我实现了
- 基于AlphaGo、遗传算法、蒙特卡洛树搜索的三种黑白棋AI
- QLearning和DQN的框架代码
- 基于Cython的快速黑白棋规则程序
- 改进实验8中实现的Qt的图形界面黑白棋对战程序，使之支持AI跨语言（C++，Python）对战
# 文件说明
.
├── AlphaGo                 AlphaGo的实现

│  ├── Arg.py               训练参数

│  ├── MCTS.py              蒙特卡洛树搜索

│  ├── Network.py           神经网络实现

│  └── Tree.svg             AlphaGo生成的搜索树样例

├── config.py               全局设置

├── dqn                     DQN算法框架代码

│  └── dqn.py

├── fast_place.c            Cython编译出的库

├── fast_place.cpython-37m-x86_64-linux-gnu.so

├── fast_place.pyx          Cython加速的规则程序

├── genetic                 遗传算法

│  ├── CppVersion           C++实现

│  │  ├── CMakeLists.txt    CMake项目文件

│  │  ├── main.cpp          遗传算法实现的主要代码

│  │  ├── reversi.cpp       黑白棋框架程序

│  │  └── reversi.h

│  ├── genetic.py           遗传算法框架代码

│  └── Reversi              使用Qt实现AI对战程序

│     ├── agent.py          Python接口

│     ├── black_chess.png

│     ├── Even.png

│     ├── first.png

│     ├── grid.cpp

│     ├── grid.h

│     ├── main.cpp

│     ├── mainwindow.cpp

│     ├── mainwindow.h

│     ├── mainwindow.ui

│     ├── Odd.png

│     ├── rc.qrc

│     ├── ReadMe.md

│     ├── reversi.cpp

│     ├── reversi.h

│     ├── Reversi.pro       项目文件

│     ├── Reversi.pro.user

│     ├── room.cpp

│     ├── room.h

│     ├── white.png

│     └── white_chess.png

├── MCTS                    蒙特卡洛搜索的实现

│  └── MCTS.py

├── play_AlphaGo_vs_AlphaGo.py  AlphaGo自我对弈程序

├── play_AlphaGo_vs_Random.py   AlphaGo与随机落子对弈程序

├── play_MCST_vs_Random.py      MCST与随机落子对弈程序

├── ql                          QLearning 算法框架代码

│  └── ql.py

├── Readme.md

├── setup.py                Cython项目文件

└── train_AlphaGo.py        AlphaGo训练程序

# 使用说明
- 首先执行setup.py编译Cython规则程序
- 执行play_MCST_vs_Random.py可以进行蒙特卡洛树搜索和随机程序的对战
- 执行play_AlphaGo_vs_Random.py可以进行AlphaGo和随机程序的对战，默认使用GPU
- 执行train_AlphaGo.py可以训练AlphaGo，训练结果保存为param文件。也可以使用pre_train.param提供的预训练参数文件
- 编译执行genetic/CppVersion项目可以训练遗传算法
- 编译执行genetic/Reversi项目可以使用遗传算法程序与AlphaBeta剪枝程序对战
- 将遗传算法训练好的基因展开为价值矩阵放入reversi.cpp中以使用
