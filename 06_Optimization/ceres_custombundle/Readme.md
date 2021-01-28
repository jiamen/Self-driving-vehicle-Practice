
// common/flags/文件夹下的 CommandArgs 用来解析用户输入的参数，同时对程序需要的参数提供默认值及文档说明
// BundleParams这个类定义Bundle Adjustment 使用的所有参数，也调用了CommandArgs类型的变量。
// 由于CommandArgs这个类的存在，我们可以直接在程序后面使用-help来查看程序所有参数的含义，使用方式可以参考改程序中BundleParams类型的写法。


#To build the code : 
mkdir build

cd ./build

cmake ..

make

#How to run the code :

cd ./build

./ceres_customBundle -input ../data/problem-16-22106-pre.txt

#see more detail settings by :
./ceres_customBundle -help
