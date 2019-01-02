#include <cstring>
#include <cfloat>
#include <iostream>
#include <random>
#include "algorithm"
#include "reversi.h"

struct action
{
    int x = -1;
    int y = -1;
};


int dir[8][2] = {
        {-1, 0},
        {-1, 1},
        { 0, 1},
        { 1, 1},
        { 1, 0},
        {1, -1},
        {0, -1},
        {-1,-1},};

static bool check(int board[8][8], int x, int y, int color)
{
    int d, i, j, cnt;
    if (board[x][y] != 0)
        return false;
    for (d = 0; d < 8; d++)
    {
        i = x;
        j = y;
        cnt = 0;
        while (true) {
            i += dir[d][0];
            j += dir[d][1];
            if (i < 0 || i > 7 || j < 0 || j > 7) {
                cnt = 0;
                break;
            }
            if (board[i][j] == -color)
                cnt++;
            else if (board[i][j] == 0)
            {
                cnt = 0;
                break;
            }
                // ????????
            else
                break;
        }
        if (cnt != 0)
            return true;
    }
    return false;
}

static vector<action> get_actions(int board[8][8], int color) {
    vector<action> ret;
    for(int i = 0; i < 8; ++i)
    {
        for(int j = 0; j < 8; ++j)
        {
            if(check(board, i,j,color))
            {
                ret.push_back({i,j});
            }
        }
    }
    return ret;
}

bool place(int board[8][8], int x, int y, int color)
{
//    for(int i = 0; i < 8; ++i)
//    {
//        for(int j = 0; j < 8; ++j)
//            cout << board[i][j] << " ";
//        cout << endl;
//    }
//    cout << endl;
    if(x < 0) return false;
    board[x][y] = color;
    bool valid = false;
    for (auto &d : dir) {
        int i = x + d[0];
        int j = y + d[1];
        while(0 <= i && i < 8 && 0 <= j && j < 8 && board[i][j] == -color)
        {
            i+= d[0];
            j+= d[1];
        }
        if(0 <= i && i < 8 && 0 <= j && j < 8 && board[i][j] == color)
        {
            while(true)
            {
                i-= d[0];
                j-= d[1];
                // ??????????????
                if(i == x && j == y)
                    break;
                valid = true;
//                cout << board[i][j] << endl;
                board[i][j] = color;
            }
        }
    }
    return valid;
}
//static const int values[8][8] =
//{{ 30, -25, 10, 5, 5, 10, -25,  30,},
//{-25, -25,  1, 1, 1,  1, -25, -25,},
//{ 10,   1,  5, 2, 2,  5,   1,  10,},
//{  5,   1,  2, 1, 1,  2,   1,   5,},
//{  5,   1,  2, 1, 1,  2,   1,   5,},
//{ 10,   1,  5, 2, 2,  5,   1,  10,},
//{-25, -25,  1, 1, 1,  1, -25, -25,},
//{ 30, -25, 10, 5, 5, 10, -25,  30,},};



static const int gene_map[8][8] =
        {{ 0, 1, 3, 5, 5, 3, 1, 0},
         { 1, 2, 4, 7, 7, 4, 2, 1},
         { 3, 4, 6, 8, 8, 6, 4, 3},
         { 5, 7, 8, 9, 9, 8, 7, 5},
         { 5, 7, 8, 9, 9, 8, 7, 5},
         { 3, 4, 6, 8, 8, 6, 4, 3},
         { 1, 2, 4, 7, 7, 4, 2, 1},
         { 0, 1, 3, 5, 5, 3, 1, 0}};

void explain(double values[8][8], const chrom_t& gene)
{
    for(int i = 0; i < 8; ++i)
    {
        for(int j = 0; j < 8; ++j)
        {
            values[i][j] = gene[gene_map[i][j]];
        }
    }
}

double heru(int board[8][8], int color, const chrom_t& gene, double value1[8][8], double value2[8][8])
{
    int bcnt = 0;
    int wcnt = 0;
    auto avi = get_actions(board, color);
    auto coner = board[0][0] + board[7][7] + board[0][7] + board[7][0];
    auto bad_coner = board[0][1] + board[0][6] + board[7][1] + board[7][6] + \
                     board[1][1] + board[1][6] + board[6][1] + board[6][7] + \
                     board[1][0] + board[6][0] + board[1][7] + board[6][7];
    double mat = 0;
    double(* pv)[8];
    if(color == 1) pv = value1;
    else pv = value2;
    for(int i = 0; i < 8; ++i)
    {
        for(int j = 0; j < 8; ++j)
        {
            mat += board[i][j] * pv[i][j];
        }
    }
    for(int i = 0; i < 8; ++i)
    {
        for(int j = 0; j < 8; ++j)
        {
            if(board[i][j] == 1) bcnt++;
            else if(board[i][j] == 0) wcnt++;
        }
    }
    auto g10 = color == BLACK ? (double)bcnt / (bcnt + wcnt) : -(double)wcnt / (bcnt + wcnt);
    return mat + gene[10] * g10 + gene[11] * coner  + gene[12] * bad_coner;
}

double AlphaBeta(int board[8][8], int color, int limit,
                 double alpha, double beta, action & bestMove, const chrom_t gene[2], double values1[8][8], double values2[8][8])
{
    if(limit == 0) return heru(board, color, gene[color == WHITE], values1, values2);
    auto avil = get_actions(board, color);
    action lbestMove;
    //TODO: WARNING HERE
    int ns[8][8];
    if(avil.empty())
    {
        bestMove.x = -1;
        bestMove.y = -1;
        auto avil2 = get_actions(board, -color);
        if(avil2.empty())
            return heru(board, color, gene[color == WHITE], values1, values2);
        else
        {
            memcpy(ns, board, 64 * sizeof(int));
            return AlphaBeta(ns, -color, limit - 1, alpha, beta, lbestMove, gene, values1, values2);
        }
    }
    if(color == BLACK)
    {
        for(const auto& a: avil)
        {
            auto x = a.x;
            auto y = a.y;
            memcpy(ns, board, 64 * sizeof(int));
            ns[x][y] = (uint8_t)color;
            auto new_alpha = AlphaBeta(ns, -color, limit - 1, alpha, beta, lbestMove, gene, values1, values2);
            if(new_alpha > alpha)
            {
                alpha = new_alpha;
                bestMove.x = x;
                bestMove.y = y;
            }
            if(beta <= alpha)
                break;
        }
        return alpha;
    }
    else
    {
        for(const auto& a: avil) {
            auto x = a.x;
            auto y = a.y;
            memcpy(ns, board, 64 * sizeof(int));
            ns[x][y] = (uint8_t) color;
            auto new_beta = AlphaBeta(ns, -color, limit - 1, alpha, beta, lbestMove, gene, values1, values2);
            if (new_beta < beta) {
                beta = new_beta;
                bestMove.x = x;
                bestMove.y = y;
            }
            if (beta <= alpha)
                break;
        }
        return beta;
    }
}
void print_chess(int chess)
{
    if(chess == 1)
        cout << "?";
    else if (chess == -1)
        cout << "?";
    else
        cout << " ";

}

void print_board(int board[8][8], int color) {

    auto moves = get_actions(board, -color);
    printf("------------------------------------\n");
    printf("|   |");
    for (int j = 0; j < 8; ++j)
        printf(" %d |", j);
    printf("\n------------------------------------\n");
    for(int i = 0; i < 8; ++i)
    {
        printf("| %d |", i);
        for(int j = 0; j < 8; ++j)
        {
            if(std::find_if(moves.begin(), moves.end(), [&](const auto& a){return a.x == i && a.y == j;}) != moves.end())
            {
                printf(" %s |", "+");
            }
            else
            {
                cout << " "; print_chess(board[i][j]); cout << " |";
            }
        }
        printf("\n------------------------------------\n");
    }

}
void print_game(int board[8][8], int x, int y, int color, bool start = false)
{
    int bcnt = 0;
    int wcnt = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if(board[i][j] == 1) bcnt++;
            else if(board[i][j] == -1) wcnt++;
        }
    }
    string str_color = color == 1 ? "black" : "white";
    if(start)
        cout << "Game start" << endl;
    else{
        if(x < 0)
        {
            cout << str_color << " can't play!" << endl;
        }
        else
        {
            cout << str_color << " put the chess piece in " << x << " " << y << endl;
        }
    }
    print_board(board, color);
    cout << "[Black: %d]" << bcnt << endl;
    cout << "[White: %d]" << wcnt << endl;
    cout << endl;

}

int pk(const chrom_t& g1, const chrom_t & g2)
{
//int(* board)[8] = new int [8][8]
    int board[8][8] =
            {{ 0, 0, 0, 0, 0, 0, 0, 0},
             { 0, 0, 0, 0, 0, 0, 0, 0},
             { 0, 0, 0, 0, 0, 0, 0, 0},
             { 0, 0, 0,-1, 1, 0, 0, 0},
             { 0, 0, 0, 1,-1, 0, 0, 0},
             { 0, 0, 0, 0, 0, 0, 0, 0},
             { 0, 0, 0, 0, 0, 0, 0, 0},
             { 0, 0, 0, 0, 0, 0, 0, 0},};

    double value1[8][8];
    double value2[8][8];

    explain(value1, g1);
    explain(value2, g2);
    chrom_t gene[2];
    gene[0] = g1;
    gene[1] = g2;
    int color = BLACK;
#ifdef USE_DEBUG
    print_game(board, -1, -1, -1, true);
#endif
    while (true)
    {
        action a;
        AlphaBeta(board, color, THINKINGLEVEL, -DBL_MAX, DBL_MAX, a, gene, value1, value2);
        if(a.x != -1)
        {
            place(board, a.x, a.y, color);
            color = -color;
#ifdef USE_DEBUG
            print_game(board, a.x, a.y, color);
#endif
        }
        else
        {
            color = -color;
            AlphaBeta(board, color, THINKINGLEVEL, -DBL_MAX, DBL_MAX, a, gene, value1, value2);
            if(a.x != -1)
            {
                place(board, a.x, a.y, color);
#ifdef USE_DEBUG
                print_game(board, a.x, a.y, color);
#endif
            }
            else
            {
                int bcnt = 0;
                int wcnt = 0;
                for (int i = 0; i < 8; ++i) {
                    for (int j = 0; j < 8; ++j) {
                        if(board[i][j] == 1) bcnt++;
                        else if(board[i][j] == -1) wcnt++;
                    }
                }
#ifdef USE_DEBUG
                if(bcnt > wcnt) cout << "Black wins" << endl;
                else if (bcnt < wcnt) cout << "White wins" << endl;
                else cout << "Draw" << endl;
#endif
                if(bcnt > wcnt) return 1;
                else if (bcnt < wcnt) return -1;
                else return 0;
            }
        }
    }
    int bcnt = 0;
    int wcnt = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if(board[i][j] == 1) bcnt++;
            else if(board[i][j] == -1) wcnt++;
        }
    }
    if(bcnt > wcnt) return 1;
    else if (bcnt == wcnt) return  0;
    else return -1;
}