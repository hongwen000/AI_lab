#include "reversi.h"

const int checkDirection[8][2] = {
    {-1, 0},
    {-1, 1},
    { 0, 1},
    { 1, 1},
    { 1, 0},
    {1, -1},
    {0, -1},
    {-1,-1},
};
bool inBoundary(int x, int y)
{
    return (x >= 0) && (x < GAMESCALE) && (y >= 0) && (y < GAMESCALE);
}

pair<int, int> getXY(int id)
{
    return {id / GAMESCALE, id % GAMESCALE};
}

int getID(int x, int y)
{
    return x * GAMESCALE + y;
}

State getNextState(const State &s, int id, int color)
{
    auto ret = s;

    auto iid = getXY(id);
    int x = iid.first;
    int y = iid.second;
//    auto [x, y] = getXY(id);
    for(int d = 0; d < 8; ++d)
    {
        int i = x;
        int j = y;
        int fx = -1;
        int fy = -1;
//        auto dd = checkDirection[d];
        auto dx = checkDirection[d][0];
        auto dy = checkDirection[d][1];
        while(inBoundary(i += dx, j += dy))
        {
            if(color == s[getID(i,j)])
            {
                fx = i;
                fy = j;
            }
            if(s[getID(i,j)] == EMPTY)
            {
                break;
            }
        }
        if(fx != -1 && fy != -1)
        {
            i = x;
            j = y;
            while(inBoundary(i += dx, j += dy))
            {
                if(i == fx && j == fy)
                    break;
                if(s[getID(i,j)] != EMPTY)
                {
                    ret[getID(i,j)] = (uint8_t)color;
                }
            }
        }
    }
    return ret;
}

bool sameColorChess(int c1, int c2)
{
    if(c1 == EMPTY || c2 == EMPTY)
        return false;
    if(c1 == c2)
        return true;
    return false;
}

bool diffColorChess(int c1, int c2)
{
    if(c1 == EMPTY || c2 == EMPTY)
        return false;
    if(c1 != c2)
        return true;
    return false;
}
std::array<int, GAMESCALE * GAMESCALE> getAvail(const State& s, int color)
{
    std::array<int, GAMESCALE * GAMESCALE> ret;
    for(auto & i: ret) i = 0;
    for(int id = 0; id < GAMESCALE * GAMESCALE; ++id)
    {
        if(s[id] != EMPTY)
        {
            ret[id] = 0;
            continue;
        }
        auto iid = getXY(id);
        int x = iid.first;
        int y = iid.second;
//        auto [x, y] = getXY(id);
        for(int d = 0; d < 8; ++d)
        {
            int this_d = 0;
//            auto [dx, dy] = checkDirection[d];
            auto dx = checkDirection[d][0];
            auto dy = checkDirection[d][1];
            int x2 = x + dx;
            int y2 = y + dy;
            bool flag = false;
            auto id2 = getID(x2, y2);
            if(inBoundary(x2, y2) && diffColorChess(color, s[id2]))
            {
                this_d += 1;
                while(inBoundary(x2 += dx, y2 += dy))
                {
                    id2 = getID(x2, y2);
                    if(sameColorChess(color, s[id2]))
                    {
                        flag = true;
                        break;
                    }
                    else if(diffColorChess(color, s[id2]))
                    {
                        this_d++;
                    }
                    else if(s[id2] == EMPTY)
                    {
                        break;
                    }
                }
                if(!flag) this_d = 0;
            }
            ret[id] += this_d;
        }
    }
    return ret;
}

static const int values[8][8] =
{{ 30, -25, 10, 5, 5, 10, -25,  30,},
{-25, -25,  1, 1, 1,  1, -25, -25,},
{ 10,   1,  5, 2, 2,  5,   1,  10,},
{  5,   1,  2, 1, 1,  2,   1,   5,},
{  5,   1,  2, 1, 1,  2,   1,   5,},
{ 10,   1,  5, 2, 2,  5,   1,  10,},
{-25, -25,  1, 1, 1,  1, -25, -25,},
{ 30, -25, 10, 5, 5, 10, -25,  30,},};

static const int * pvalues = ((const int *)values);


// 评估函数，s是要评估的状态，color是当前要落子的棋子颜色
int positionalValue(const State& s, int color)
{
    // 黑子数量
    int bcnt = 0;
    // 白子数量
    int wcnt = 0;
    // 空白位置数量
    int ecnt = 0;
    // 仅用于8X8的黑白棋
    if(GAMESCALE != 8) throw;
    // 效用值
    int ret = 0;
    // 获取当前可行动的位置
    auto avi = getAvail(s, color);
    // 移动性
    int mobility = 0;
    // 如果某个位置可落子数大于0，移动性+1
    for(const auto& i : avi) if(i)mobility++;
    // 遍历所有位置, 依据价值表计算局面的价值
    for(int i = 0; i < 64; ++i)
    {
        // 若为空,价值为0
        if(s[i] == EMPTY)
        {
            ecnt++;
            continue;
        }
        // 若为黑子，加上价值
        else if(s[i] == BLACK)
        {
            ret += *(pvalues + i);
            bcnt++;
        }
        // 若为白子，减去价值
        else
        {
            ret -= *(pvalues + i);
            wcnt++;
        }
    }
    // 判断当前局面是否已经失败
    if(mobility == 0 && color == BLACK && bcnt < wcnt)
        ret -= 1000 * (ecnt == 0);
    else if(mobility == 0 && color == WHITE && bcnt > wcnt)
        ret += 1000 * (ecnt == 0);
    // 返回加权后的价值
    return (ret + mobility * 5 *(color == BLACK? 1 : -1));
}

/* 	MinMax搜索
/	s是当前状态
/	color是当前棋手持的棋子颜色（BLACK/WHITE）
/	limit是最大搜索层数
/ 	V是评估函数
/	bestMove是最佳落子位置
*/
int MinMax(const State& s, int color, int limit, const valueFunc& V, int & bestMove)
{
    // 若达到了最大探索深度，则对当前状态使用启发式函数求估算的价值
    if(limit == 0) return V(s, color);
    // 获得当前颜色的棋子可移动的位置
    auto avil = getAvail(s, color);
    // 记录是否为叶子节点
    bool isTerminal = true;
    // 记录最佳效益值
    int bestValue = 0;
    int lbestMove;
    // 记录是否在探索第一个子节点
    bool isFirstTime = true;
    // 遍历所有位置
    for(size_t i = 0; i < avil.size(); ++i)
    {
        // 如果该位置当前棋手不可落子，检查下一状态
        if(avil[i] == 0) continue;
        // 发现了可落子位置，说明不是叶子节点
        isTerminal = false;
        // 生成在该点落子的子状态
        auto ns = s;
        ns[i] = (uint8_t)color;
        // 下一轮的棋子颜色
        int nextColor = (color == BLACK) ? WHITE : BLACK;
        // 获取子状态的效益值
        auto ret = MinMax(ns, nextColor, limit - 1, V, lbestMove);
        // 以下三种情况更新最佳效益值
        // 1. 第一次探索子状态
        // 2. 当前棋手持黑子，且子状态效益值大于最佳效益值
        // 2. 当前棋手持白子，且子状态效益值小于最佳效益值
        if(isFirstTime || (color == BLACK && ret > bestValue) || (color == WHITE && ret < bestValue))
        {
            bestValue = ret;
            bestMove = i;
        }
        isFirstTime = false;
    }
    // 未发现可落子位置，则对当前状态使用启发式函数求估算的价值
    if(isTerminal) return V(s, color);
    // 返回最佳效益值
    return bestValue;
}

/* 	MinMax搜索
/	s是当前状态
/	color是当前棋手持的棋子颜色（BLACK/WHITE）
/	limit是最大搜索层数
/ 	V是评估函数
/	alpha是alpha值
/	beta是beta值
/	bestMove是最佳落子位置
*/
int AlphaBeta_worker(const State& s, int color, int limit, const valueFunc& V,
              int alpha, int beta);
int AlphaBeta(const State& s, int color, int limit, const valueFunc& V,
              int alpha, int beta, int & bestMove)
{
    // 若达到了最大探索深度，则对当前状态使用启发式函数求估算的价值
    if(limit == 0) return V(s, color);
    // 获得当前颜色的棋子可移动的位置
    auto avil = getAvail(s, color);
    // 记录是否为叶子节点
    bool isTerminal = true;
    int lbestMove;
    // 若当前棋手持黑子
    if(color == BLACK)
    {
        // 遍历所有位置
        for(size_t i = 0; i < avil.size(); ++i)
        {
            // 如果该位置当前棋手不可落子，检查下一状态
            if(avil[i] == 0) continue;
            // 发现了可落子位置，说明不是叶子节点
            isTerminal = false;
            // 生成在该点落子的子状态
            auto ns = s;
            ns[i] = (uint8_t)color;
            // 下一轮的棋子颜色
            int nextColor = (color == BLACK) ? WHITE : BLACK;
            // 得到某个子节点的效用值
            auto new_alpha = AlphaBeta(ns, nextColor, limit - 1, V, alpha, beta, lbestMove);
            // 若子节点的效用值大于alpha值，则更新该值
            if(new_alpha > alpha)
            {
                alpha = new_alpha;
                bestMove = i;
            }
            // 当前节点的alpha值已经大于父节点的beta值，当前节点已经不会被选
            if(beta <= alpha)
                //发送剪枝
                break;
        }
        // 未发现可落子位置，则对当前状态使用启发式函数求估算的价值
        if(isTerminal) return V(s, color);
        // 返回alpha值
        else return alpha;
    }
    // 若当前棋手持白子，同理
    else
    {
        for(size_t i = 0; i < avil.size(); ++i)
        {
            if(avil[i] == 0) continue;
            isTerminal = false;
            auto ns = s;
            ns[i] = (uint8_t)color;
            int nextColor = (color == BLACK) ? WHITE : BLACK;
            auto new_beta = AlphaBeta(ns, nextColor, limit - 1, V, alpha, beta, lbestMove);
            if(new_beta < beta)
            {
                beta = new_beta;
                bestMove = i;
            }
            if(beta <= alpha)
                break;
        }
        if(isTerminal) return V(s, color);
        else return beta;
    }
}

