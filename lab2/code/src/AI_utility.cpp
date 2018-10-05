//
// Created by 李新锐 on 05/10/2018.
//

#include "AI_utility.h"
#include "csv.h"

FileData_t readFile(const Str& filen)
{
    FileData_t ret;
    io::CSVReader<7> in(filen);
    Vec<Str> s(7);
    while(in.read_row(s[0], s[1], s[2], s[3], s[4], s[5], s[6]))
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
