#include <iostream>
#include <array>
#include <set>
#include <list>
#include <functional>
#include <vector>
#include <sstream>
#include <chrono>
#define GAMESCALE 14

using namespace std;

stringstream ss;
#define cout ss

using State = std::vector<int>;

void print(const State& X)
{
    return;
    for(auto& i: X)
    {
        printf("%d ,", i);
    }
    printf("\n");
}

using domain_t = std::array<std::array<int, GAMESCALE>, GAMESCALE>;
domain_t domains;

void initDomain()
{
    for(int i = 0; i < GAMESCALE; ++i)
    {
        for(int j = 0; j < GAMESCALE; ++j)
        {
            domains[i][j] = j;
        }
    }
}

std::list<int> V;
State X(GAMESCALE, -1);

auto col_constraint =  [](const auto& p, auto v, auto i){
//    cout << "col constraint" << endl;
    for(int v2 = 0; v2 < GAMESCALE; ++v2)
    {
        int i2 = p[v2];
        if(i2 != -1 && v2 != v && i2 == i)
            return false;
    }
    return true;
};

auto dia_constraint = [](const auto& p, auto v, auto i){
//    cout << "dia constraint" << endl;
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
bool is_domain_empty(const std::array<int, GAMESCALE>& domain)
{
    for(int i = 0; i < GAMESCALE; ++i)
        if(domain[i] >= 0)
            return false;
    return true;
}

bool fc(State&p, int level)
{
    for(auto v: V)
    {
        int pv = p[v];
        for(int i = 0; i < GAMESCALE; ++i)
        {
            if(domains[v][i] < 0) continue;
            for(const auto& c: Constraints)
            {
                p[v] = i;
//                cout << "Checking variable " << v << " assign " << i << " with ";
                if(!c(p, v, i))
                {
//                    cout << "deleted" << endl;
                    domains[v][i] = -level;
                    break;
                }
            }
        }
        p[v] = pv;
        if(is_domain_empty(domains[v]))
        {
//            cout << "DWO happend at variable " << v << endl;
            return true;
        }
    }
    return false;
}

void BT(int level)
{
    if(V.empty())
    {
        print(X);
//        exit(0);
        return;
    }
    auto v = V.front();
    V.pop_front();
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
    V.push_front(v);
    X[v] = -1;
}

void FC(int level)
{
//    cout << " ------- In FC level --------" << level << endl;
    if(V.empty())
    {
        print(X);
//        exit(0);
        return;
    }
    auto v = V.front();
    V.pop_front();
    for(int i = 0; i < GAMESCALE; ++i)
    {
//        cout << "---> Testing variable " << v << " with assign " << i << endl;
        if(domains[v][i] < 0) continue;
        X[v] = i;
        auto DWO = fc(X, level);
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
    V.push_front(v);
    X[v] = -1;
}

int main()
{
    initDomain();
    for(int i = 0; i < GAMESCALE; ++i)
        V.push_back(i);
    auto s = chrono::steady_clock::now();
    BT(1);
    auto dis = chrono::steady_clock::now() - s;
    printf("%ld\n", dis.count() / 1000000);

    V.clear();
    initDomain();
    for(int i = 0; i < GAMESCALE; ++i)
        V.push_back(i);
    s = chrono::steady_clock::now();
    FC(1);
    dis = chrono::steady_clock::now() - s;
    printf("%ld\n", dis.count() / 1000000);
    return 0;
}
