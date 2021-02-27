//
// Created by zlc on 2021/2/25.
//

#include <fstream>          // 文件输入输出控制
#include <iostream>         // 标准输入输出
#include <sstream>
#include <vector>
#include <stdlib.h>
#include "Eigen/Dense"
#include "FusionEKF.h"
#include "ground_truth_package.h"
#include "measurement_package.h"


using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


// Usage instructions: ./ExtendedKF path/to/input.txt output.txt
// ./ExtendedKF ../data/obj_pose-laser-radar-synthetic-input.txt output.txt
void check_arguments(int argc, char* *argv)
{
    string usage_instructions = "Usage instructions: ";
    usage_instructions += argv[0];
    usage_instructions += " path/to/input.txt output.txt";

    bool has_valid_args = false;

    // make sure the user has provided input and output files.
    if( 1 == argc )
    {
        cerr << usage_instructions << endl;
    }
    else if( 2 == argc )
    {
        cerr << "Please include an output file.\n" << usage_instructions << endl;
    }
    else if( 3 == argc )
    {
        has_valid_args = true;
    }
    else if( argc > 3 )
    {
        cerr << "Too many arguments.\n" << usage_instructions << endl;
    }

    if( !has_valid_args )
    {
        exit(EXIT_FAILURE);
    }
}

void check_files(ifstream& in_file, string& in_name,
                 ofstream& out_file, string& out_name)
{
    if( !in_file.is_open() )
    {
        cerr << "Cannot open input file: " << in_name << endl;
        exit(EXIT_FAILURE);
    }

    if( !out_file.is_open() )
    {
        cerr << "Cannot open output file: " << out_name << endl;
        exit(EXIT_FAILURE);
    }
}


int main(int argc, char* *argv)
{
    check_arguments(argc, argv);

    string in_file_name_ = argv[1];
    ifstream in_file_(in_file_name_.c_str(), ifstream::in);
    // c_str()函数返回一个指向正规C字符串的指针常量, 内容与本string串相同。
    // 这是为了与c语言兼容，在c语言中没有string类型，故必须通过string类对象的成员函数c_str()把string 对象转换成c中的字符串样式。
    // 1.c_str是一个内容为字符串指向字符数组的临时指针；
    // 2.c_str返回的是一个可读不可改的常指针；

    string out_file_name_ = argv[2];
    ofstream out_file_(out_file_name_.c_str(), ofstream::out);

    check_arguments()

}


