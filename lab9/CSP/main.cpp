#include <iostream>
#include <functional>
#include <vector>
#include <sstream>
#include <chrono>
#define GAMESCALE 8

using namespace std;

//stringstream ss;
//#define cout ss

using State = int*;

int cnt = 0;
void inline print(State X)
{
    cnt++;
    return;
    for(int i = 0; i < GAMESCALE; ++i)
    {
        printf("%d ,", X[i]);
    }
    printf("\n");
}

using domain_t = int[GAMESCALE][GAMESCALE];
domain_t domains;
int X[GAMESCALE];

void init()
{
    cnt = 0;
    for(int i = 0; i < GAMESCALE; ++i)
    {
        X[i] = -1;
        for(int j = 0; j < GAMESCALE; ++j)
        {
            domains[i][j] = j;
        }
    }
}


auto col_constraint =  [](const auto& p, auto v, auto i){
    for(int v2 = 0; v2 < GAMESCALE; ++v2)
    {
        int i2 = p[v2];
        if(i2 != -1 && v2 != v && i2 == i)
            return false;
    }
    return true;
};

auto dia_constraint = [](const auto& p, auto v, auto i){
    for(int v2 = 0; v2 < GAMESCALE; ++v2)
    {
        auto i2 = p[v2];
        if(i2 != -1 && v2 != v && abs(i2 - i) == abs(v2 - v))
            return false;
    }
    return true;
};

using checkFunc_t = std::function<bool(const State& p, int v, int i)>;
vector<checkFunc_t> Constraints = {col_constraint, dia_constraint};
inline bool is_domain_empty(int * domain)
{
    for(int i = 0; i < GAMESCALE; ++i)
        if(domain[i] >= 0)
            return false;
    return true;
}

bool fc2(int level, int v, int i)
{
    for(int j = 0; j < GAMESCALE; ++j)
    {
        if(domains[v][j] >= 0) domains[v][j] = -level;
        if(domains[j][i] >= 0) domains[j][i] = -level;
        if(v+j < GAMESCALE)
        {
            if(i + j < GAMESCALE && domains[v+j][i+j] >= 0) domains[v+j][i+j] = -level;
            if(i - j >= 0 && domains[v+j][i-j] >= 0) domains[v+j][i-j] = -level;
        }
    }
    for(int i = v + 1; i < GAMESCALE; ++i)
    {
        if(is_domain_empty(domains[i]))
        {
//            cout << "DWO happend at variable " << v << endl;
            return true;
        }
    }
    return false;
}
bool fc(int level, int v, int i)
{
    for(int j = v + 1; j < GAMESCALE; ++j)
    {
        bool dwo = true;
        for(int k = 0; k < GAMESCALE; ++k)
        {
            if(domains[j][k] < 0)
                continue;
            if(k == i || abs(j - v) == abs(k - i))
                domains[j][k] = -level;
            dwo = false;
        }
        if(dwo)
            return true;
    }
    return false;
}

void BT(int level)
{
    auto v = level - 1;
    if(v == GAMESCALE)
    {
        print(X);
//        exit(0);
        return;
    }
    for(auto i: domains[v])
    {
        X[v] = i;
        bool satify = true;
        for(const auto c: Constraints)
        {
            if(!c(X, v, i))
            {
                satify = false;
                break;
            }
        }
        if(satify)
        {
            BT(level+1);
        }
    }
    X[v] = -1;
}

void FC(int level)
{
//    cout << " ------- In FC level --------" << level << endl;
    auto v = level - 1;
    if(v == GAMESCALE)
    {
        print(X);
//        exit(0);
        return;
    }
    for(int i = 0; i < GAMESCALE; ++i)
    {
//        cout << "---> Testing variable " << v << " with assign " << i << endl;
        if(domains[v][i] < 0) continue;
        X[v] = i;
        auto DWO = fc(level, v, i);
        if(!DWO)
        {
            FC(level+1);
        }
//        cout << "Restoring <----" << endl;
        for(int j = 0; j < GAMESCALE; ++j)
        {
            for(int k = 0; k < GAMESCALE; ++k)
            {
//                    cout << "Restore " << k << " to domain " << j << endl;
                if(domains[j][k] == -level)
                    domains[j][k] = k;
            }
        }
//        cout << "Restore over" << endl;
    }
    X[v] = -1;
}

int main()
{
    init();
    auto s = chrono::steady_clock::now();
//    BT(1);
    auto dis = chrono::steady_clock::now() - s;
    init();
    auto s2 = chrono::steady_clock::now();
    FC(1);
    auto dis2 = chrono::steady_clock::now() - s2;
    int i = GAMESCALE;
    printf("%d, %ld, %ld, %d\n", i, dis.count() / 1000, dis2.count() / 1000, cnt);
    return 0;
}
