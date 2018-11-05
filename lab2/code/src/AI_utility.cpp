//
// Created by 李新锐 on 05/10/2018.
//

#include "AI_utility.h"
#include "csv.h"

FileData_t readFile(const Str& filen)
{
    FileData_t ret;
    io::CSVReader<51> in(filen);
    Vec<Str> s(51);
    while(in.read_row(s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8],s[9],s[10],s[11],s[12],s[13],s[14],s[15],s[16],s[17],s[18],s[19],s[20],s[21],s[22],s[23],s[24],s[25],s[26],s[27],s[28],s[29],s[30],s[31],s[32],s[33],s[34],s[35],s[36],s[37],s[38],s[39],s[40],s[41],s[42],s[43],s[44],s[45],s[46],s[47],s[48],s[49],s[50]))
    {
        ret.push_back(s);
//        for(const auto& w : s)
//        {
//            std::cout << w << ", ";
//        }
//        std::cout << std::endl;
    }
    return ret;
}

Eigen::MatrixXd readProject(const Str& filen)
{
    Vec<Vec<double>> file;
    io::CSVReader<51> in(filen);
    Vec<double> s(51);
    while(in.read_row(s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8],s[9],s[10],s[11],s[12],s[13],s[14],s[15],s[16],s[17],s[18],s[19],s[20],s[21],s[22],s[23],s[24],s[25],s[26],s[27],s[28],s[29],s[30],s[31],s[32],s[33],s[34],s[35],s[36],s[37],s[38],s[39],s[40],s[41],s[42],s[43],s[44],s[45],s[46],s[47],s[48],s[49],s[50]))
    {
        file.push_back(s);
//        for(const auto& w : s)
//        {
//            std::cout << w << ", ";
//        }
//        std::cout << std::endl;
    }
    Eigen::MatrixXd ret(file[0].size(), file.size());
    for(Eigen::Index j = 0; j < ret.cols(); ++j)
    {
        for(Eigen::Index i = 0; i < ret.rows(); ++i)
        {
            ret(i, j) = file[j][i];
        }
    }
    return ret;
}
Eigen::MatrixXd readTest(const Str& filen)
{
    Vec<Vec<double>> file;
    io::CSVReader<50> in(filen);
    Vec<double> s(51);
    while(in.read_row(s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8],s[9],s[10],s[11],s[12],s[13],s[14],s[15],s[16],s[17],s[18],s[19],s[20],s[21],s[22],s[23],s[24],s[25],s[26],s[27],s[28],s[29],s[30],s[31],s[32],s[33],s[34],s[35],s[36],s[37],s[38],s[39],s[40],s[41],s[42],s[43],s[44],s[45],s[46],s[47],s[48],s[49]))
    {
        file.push_back(s);
//        for(const auto& w : s)
//        {
//            std::cout << w << ", ";
//        }
//        std::cout << std::endl;
    }
    s[50] = 0;
    Eigen::MatrixXd ret(file[0].size(), file.size());
    for(Eigen::Index j = 0; j < ret.cols(); ++j)
    {
        for(Eigen::Index i = 0; i < ret.rows(); ++i)
        {
            ret(i, j) = file[j][i];
        }
    }
    return ret;
}
