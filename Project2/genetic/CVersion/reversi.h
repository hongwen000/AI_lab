#ifndef REVERSI_H
#define REVERSI_H
#include <cstdint>
#include <array>
#include "cstdlib"
#include "cstdint"
#include "climits"
using namespace std;
#include <utility>
#include <functional>
#define BLACK 1
#define WHITE -1
#define THINKINGLEVEL 5

//#define USE_GENE_DEBUG
//#define USE_DEBUG
#define GENE_CNT 13
using chrom_t = array<double, GENE_CNT>;
using chrom_t = array<double, GENE_CNT>;
int pk(const chrom_t& g1, const chrom_t & g2);
#endif //
