#ifndef REVERSI_H
#define REVERSI_H
#include <cstdint>
#include <array>
using namespace std;
#define GAMESCALE 8
#include <utility>
#include <functional>
enum {EMPTY = 0, BLACK, WHITE};
#define THINKINGLEVEL 10

using State = std::array<char, GAMESCALE * GAMESCALE>;

extern const int checkDirection[8][2];
bool inBoundary(int x, int y);
pair<int,int> getXY(int id);
int getID(int x, int y);
State getNextState(const State& s, int id, int color);
std::array<int, GAMESCALE * GAMESCALE> getAvail(const State& s, int color);
int positionalValue(const State& s, int color);
using valueFunc = std::function<int(const State& s, int color)>;
int MinMax(const State& s, int color, int limit, const valueFunc& V, int & bestMove);
int AlphaBeta(const State& s, int color, int limit, const valueFunc& V, int alpha, int beta, int & bestMove);

#endif // REVERSI_H
