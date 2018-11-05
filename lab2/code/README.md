## 构建说明

- 本项目使用cmake构建系统

- 在ArchLinux上编译通过，gcc版本8.2.1，C++语言标准为C++17

- 使用到的第三方库有Eigen3h和range-v3
- 随机森林的Target为test_regression_tree.cpp - T，5
- 回归树的Target为test_regression_tree.cpp -T, 4
- 建议使用release版本构建，否则性能会很低

## 文件说明

.
├── CMakeLists.txt				cmake项目文件
├── CMakeLists.txt.user
├── data						数据文件
│   ├── Car_test.csv
│   ├── Car_train.csv
│   └── readme.txt
├── inc							头文件
│   ├── AI_utility.h			项目公共头文件，主要是库的引用和类型定义
│   ├── array_view.h			针对vector<vector>实现的视图类
│   ├── csv.h					csv读取库
│   ├── decision_tree.h			决策树类（上次实验代码）
│   ├── regression_tree.h		回归树类（本次实验主要的代码实现在这里）
│   └── matrix_view.h			针对Eigen::Matrix实现的视图类
├── main.cpp					和实验结果展示相关的测试样例
├── readme.txt
├── src							头文件对应的cpp
│   ├── AI_utility.cpp			
│   ├── regression_tree.cpp
│   └── matrix_view.cpp
└── unit_test					针对代码的各个部分编写的测试样例			
​    ├── CMakeLists.txt
​    ├── test_AI_utility.cpp
​    ├── test_array_view.cpp
​    ├── test_decison_tree.cpp
​    ├── test_regression_tree.cpp
​    ├── test_eigen3.cpp
​    ├── test_matrix_view.cpp
​    └── test_ranges.cpp

10 directories, 28 files
