#include "BALProblem.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>


#include "tools/random.h"
#include "tools/rotation.h"


typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;


/** 读文件 */
template < typename T >
void FscanfOrDie(FILE *fptr, const char *format, T *value)
{
    int num_scanned = fscanf(fptr, format, value);
    if( num_scanned != 1 )
        std::cerr << "Invalid UW data file. ";
}


/** 随机添加噪声 */
void PerturbPoint3(const double sigma, double* point)
{
    for(int i = 0; i < 3; ++i)
        point[i] += RandNormal() * sigma;
}

/** 取一个数组的中位数，用于归一化; */
double Median(std::vector<double>* data)
{
    int n = data->size();

    std::vector<double>::iterator mid_point = data->begin() + n/2;
    std::nth_element(data->begin(), mid_point, data->end());

    return *mid_point;
}


/** 构造函数：读取txt文件,一种方式是以旋转向量的方式读取，另一种方式是把旋转向量转换成四元数，然后以四元数读取
 ** 其实主要是旋转矩阵的变换，其它问题都没变     */
BALProblem::BALProblem(const std::string& filename, bool use_quaternions)
{
    FILE* fptr = fopen(filename.c_str(), "r");
    
    if (NULL == fptr)
    {
        std::cerr << "Error: unable to open file " << filename;
        return;
    }

    // This wil die horribly on invalid files. Them's the breaks.
    FscanfOrDie(fptr, "%d", &num_cameras_);         // 相机的数量
    FscanfOrDie(fptr, "%d", &num_points_);          // 观测点的数量
    FscanfOrDie(fptr, "%d", &num_observations_);    // 像素的数量

    std::cout << "Header: " << num_cameras_     // 16
              << " " << num_points_             // 22106
              << " " << num_observations_       // 83718    这里的数据来自于data/problem-16-22106-pre.txt 的第一行 3个数
              << std::endl;

    point_index_  = new int[num_observations_];              // 观测点的索引值 83718
    camera_index_ = new int[num_observations_];             // 相机的索引值   83718
    observations_ = new double[2 * num_observations_];      // 像素坐标的索引值  83718*2

    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;   // 参数数量，9个参数分别代表什么在projection.h头文件中有标注
    parameters_ = new double[num_parameters_];              // 开辟一片能够包含所有参数的大空间，空间首地址是parameters_，66462个double空间

    /************************对txt文件按列读取***********************/
    // 第i个相机观测第j个路标以及观测得到的像素坐标
    for (int i = 0; i < num_observations_; ++ i)
    {
        FscanfOrDie(fptr, "%d", camera_index_ + i);     // 课本第258页第i个相机 观测 第j个路标 所看到的像素坐标
        FscanfOrDie(fptr, "%d", point_index_ + i);      // 第j个路标
        for (int j = 0; j < 2; ++ j)
        {
            FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);          // 每一行后面2个像素坐标
        }
    }

    // 读到txt后半部分,读取所有优化变量的值     data/problem-16-22106-pre.txt 文件下的 66462 个参数： 16*9+22106*3 = 66462
    for (int i = 0; i < num_parameters_; ++ i)
    {
        FscanfOrDie(fptr, "%lf", parameters_ + i);
    }

    fclose(fptr);           // 关闭文件,读取文件之后一定要关闭


    /*************************************是否使用四元数**************************************/
    use_quaternions_ = use_quaternions;
    if ( use_quaternions )
    {
        std::cout<<"***********************break*************************"<<std::endl;
        // Switch the angle-axis rotations to quaternions. 转换旋转向量到四元数
        num_parameters_ = 10 * num_cameras_ + 3 * num_points_;          // 如果使用四元数的话，更改优化变量的总个数
        double* quaternion_parameters = new double[num_parameters_];    // 重新开辟一片能够包含所有参数的大空间，空间首地址是quaternion_parameters
        double* original_cursor = parameters_;                          // 原来旋转向量起始位置,parameters是在BALProblem.h定义的
        double* quaternion_cursor = quaternion_parameters;              // 四元数起始位置  复制一份
        for (int i = 0; i < num_cameras_; ++ i)
        {
            AngleAxisToQuaternion(original_cursor, quaternion_cursor);  // 将旋转向量转换成四元数目
            quaternion_cursor += 4;         // 四元数是四位，每次前进四个位置
            original_cursor += 3;           // 角轴是三位，每次前进三个位置
            for (int j = 4; j < 10; ++ j)
            {
                // 原来用角轴表示总共是9维，现在把旋转向量转换成四元数，维数变成10维，除了角轴剩下的向量保持不变即可.
                *quaternion_cursor++ = *original_cursor++;
            }
        }

        // Copy the rest of the points. 复制3D点的坐标
        for (int i = 0; i < 3 * num_points_; ++ i)
        {
            *quaternion_cursor++ = *original_cursor++;
        }
        // Swap in the quaternion parameters.
        delete []parameters_;                   // 之前旋转向量形式new出来的数组一定要释放
        parameters_ = quaternion_parameters;    // 重新采用四元数形式下新new出来的数组
    }
}


void BALProblem::WriteToFile(const std::string& filename) const
{
    FILE* fptr = fopen(filename.c_str(),"w");
    // 检查文件是否打开
    if(NULL == fptr)
    {
        std::cerr << "Error: unable to open file " << filename;
        return;
    }
    fprintf(fptr, "%d %d %d %d\n", num_cameras_, num_cameras_, num_points_, num_observations_);

    for(int i = 0; i < num_observations_; ++ i)
    {
        fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);  // 相机的索引以及第i个相机观测第j个路标点
        for(int j = 0; j < 2; ++ j)
        {
            fprintf(fptr, " %g", observations_[2*i + j]);   // 观测点的像素坐标
        }
        fprintf(fptr,"\n");
    }

    for(int i = 0; i < num_cameras(); ++i)
    {
        double angleaxis[9];
        if(use_quaternions_)
        {
            //OutPut in angle-axis format.
            QuaternionToAngleAxis(parameters_ + 10 * i, angleaxis);     // 如果使用四元数则要转换成角轴
            memcpy(angleaxis + 3, parameters_ + 10 * i + 4, 6 * sizeof(double));
            // 角轴3位, 四元数四维, 整个相机用四元数表示的话是10维, 步长为6.
        }
        else
        {
            memcpy(angleaxis, parameters_ + 9 * i, 9 * sizeof(double));   // 如果直接以旋转向量表示的话,长度为9,步长为9.
        }
        // 每个9维相机参数的循环控制输出，注意每个值有个换行
        for(int j = 0; j < 9; ++ j)
        {
            fprintf(fptr, "%.16g\n", angleaxis[j]);
        }
    }

    const double* points = parameters_ + camera_block_size() * num_cameras_;
    // point指针指向路标的首位置
    for(int i = 0; i < num_points(); ++ i)      // 以路标点的总个数作为控制循环
    {
        const double* point = points + i * point_block_size();      // 指针指向首位置
        for(int j = 0; j < point_block_size(); ++ j)
        {
            fprintf(fptr,"%.16g\n", point[j]);
        }
    }

    fclose(fptr);
}

// Write the problem to a PLY file for inspection in Meshlab or CloudCompare
/** 将问题写入ply文件，以便在Meshlab或CloudCompare中进行检查 */
void BALProblem::WriteToPLYFile(const std::string& filename) const
{
    std::ofstream of(filename.c_str());     // 创建一个文件输出流对象，输出至文件名

    of  << "ply"
        << '\n' << "format ascii 1.0"
        << '\n' << "element vertex " << num_cameras_ + num_points_
        << '\n' << "property float x"
        << '\n' << "property float y"
        << '\n' << "property float z"
        << '\n' << "property uchar red"
        << '\n' << "property uchar green"
        << '\n' << "property uchar blue"
        << '\n' << "end_header" << std::endl;

    // Export extrinsic data (i.e. camera centers) as green points.
    double angle_axis[3];
    double center[3];
    for(int i = 0; i < num_cameras(); ++ i)
    {
        const double* camera = cameras() + camera_block_size() * i;     // cameras()返回所有参数中相机参数的起始位置
        // cameras()返回parameters; camera_block_size有两种表达方式：10？9
        CameraToAngelAxisAndCenter(camera, angle_axis, center);         // center解析出来的 相机原点在世界坐标系下的坐标承接数组
        of << center[0] << ' ' << center[1] << ' ' << center[2]
           << " 0 255 0" << '\n';            // 用绿色表示
    }

    // Export the structure (i.e. 3D Points) as white points.   把用到的22106个点全部导出保存到initial.ply中
    const double* points = parameters_ + camera_block_size() * num_cameras_;
    for(int i = 0; i < num_points(); ++ i)
    {
        const double* point = points + i * point_block_size();
        for(int j = 0; j < point_block_size(); ++ j)
        {
            of << point[j] << ' ';
        }
        of << "255 255 255\n";
    }

    of.close();
}


// camera数据中的旋转向量以及平移向量 解析 相机世界坐标系的姿态,(依旧是旋转向量)和位置（世界坐标系下的XYZ）
// 具体参数说明：
//    · camera要解析的相机参数，前三维旋转，接着三维平移
//    · angle_axis解析出的相机姿态 承接数组，也是旋转向量形式
//    · center解析出来的 相机原点在世界坐标系下的坐标承接数组
void BALProblem::CameraToAngelAxisAndCenter(const double* camera, 
                                            double* angle_axis,
                                            double* center) const
{
    VectorRef angle_axis_ref(angle_axis,3);
    if( use_quaternions_ )
    {
        QuaternionToAngleAxis(camera, angle_axis);       // 将四元数转换成角轴
    }
    else
    {
        angle_axis_ref = ConstVectorRef(camera,3);  // 将camera的前三维复制过去
    }

    // c = -R't    R是R_cw  -R = R_wc
    // 如何计算center
    // center是相机原点在世界坐标系下的定义
    // PW_center:世界坐标系下的相机坐标
    // PC_center:相机坐标系下的相机原点坐标（0,0,0）
    // 根据相机坐标系与世界坐标系的转换关系：PW_center×R+t=PC_center
    // PW_center= -R't
    // 旋转向量的反向过程（求逆）和旋转向量取负一样。
    Eigen::VectorXd inverse_rotation = -angle_axis_ref;
    /*************调用inline void AngleAxisRotatePoint
     *(const T angle_axis[3], const T pt[3], T result[3]) 世界坐标转换成相机坐标*/
    AngleAxisRotatePoint(inverse_rotation.data(),    // -R'
                         camera + camera_block_size() - 6,  // 先偏移到末尾再往回倒6个 translation t
                         center);
    VectorRef(center,3) *= -1.0;                // 整个相机投影模型投影到光心的后面，所以乘以-1;
}


/**上一个过程的逆操作
*   根据世界坐标系下的相机姿态和位置,生成camera数据
* @param angle_axis 旋转向量数据
* @param center 相机中心在世界坐标系下的位置坐标
* @param camera 承接数据的camera数组，由于这里只是生成旋转和平移，所以是camera的前6维
*/
void BALProblem::AngleAxisAndCenterToCamera(const double* angle_axis,
                                            const double* center,
                                            double* camera) const{
    ConstVectorRef angle_axis_ref(angle_axis,3);
    if(use_quaternions_)
    {
        AngleAxisToQuaternion(angle_axis,camera);
    }
    else
    {
        VectorRef(camera, 3) = angle_axis_ref;      // 得到camera数据的前三维
    }

    // PW_center×R+t=PC_center(0,0,0)
    // t = -R * c
    // (camera + camera_block_size() - 6)   camera中的平移部分t
    AngleAxisRotatePoint(angle_axis, center,camera+camera_block_size() - 6);        // 世界坐标转换成相机坐标
    VectorRef(camera + camera_block_size() - 6,3) *= -1.0;
}


/** 对数据进行归一化处理  */
void BALProblem::Normalize()
{
    // Compute the marginal median of the geometry
    std::vector<double> tmp(num_points_);
    Eigen::Vector3d median;
    double* points = mutable_points();  // 所有参数中 点参数的起始位置
    for(int i = 0; i < 3; ++ i)         // 外圈循环控制3D点的坐标xyz
    {
        for(int j = 0; j < num_points_; ++ j)
        {
            tmp[j] = points[3 * j + i];
        }
        median(i) = Median(&tmp);       // 得到x，y，z每一列的中位数
    }

    for(int i = 0; i < num_points_; ++i)
    {
        VectorRef point(points + 3 * i, 3);
        tmp[i] = (point - median).lpNorm<1>();      // https://blog.csdn.net/skybirdhua1989/article/details/17584797
    }

    const double median_absolute_deviation = Median(&tmp);      //对tmp求中值

    // Scale so that the median absolute deviation of the resulting
    // reconstruction is 100

    const double scale = 100.0 / median_absolute_deviation;
    // X = scale * (X - median)
    for(int i = 0; i < num_points_; ++i)
    {
        VectorRef point(points + 3 * i, 3);
        point = scale * (point - median);
    }

    // 直到这一步结束，发现到这里目的就是对路标点坐标做一些处理，为啥处理？为啥这么处理？
    double* cameras = mutable_cameras();
    double angle_axis[3];
    double center[3];               // 相机中心在世界坐标系下的坐标
    for(int i = 0; i < num_cameras_ ; ++i)
    {
        double* camera = cameras + camera_block_size() * i;
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        // center = scale * (center - median)
        VectorRef(center,3) = scale * (VectorRef(center,3)-median);
        AngleAxisAndCenterToCamera(angle_axis, center,camera);
    }
}


/***********添加噪声（rotation translation point）************************************/
void BALProblem::Perturb(const double rotation_sigma, 
                         const double translation_sigma,
                         const double point_sigma)
{
    assert(point_sigma >= 0.0);
    assert(rotation_sigma >= 0.0);
    assert(translation_sigma >= 0.0);

    double* points = mutable_points();
    if(point_sigma > 0)
    {
        for(int i = 0; i < num_points_; ++ i)
        {
            PerturbPoint3(point_sigma, points + 3 * i);
        }
    }

    for(int i = 0; i < num_cameras_; ++ i)
    {
        double* camera = mutable_cameras() + camera_block_size() * i;

        double angle_axis[3];
        double center[3];
        // Perturb in the rotation of the camera in the angle-axis
        // representation
        CameraToAngelAxisAndCenter(camera, angle_axis, center);     // 主要是通过camera以及angle_axis求center(相机原点在世界坐标系下的坐标)
        if(rotation_sigma > 0.0)        // 检查旋转噪声
        {
            PerturbPoint3(rotation_sigma, angle_axis);
        }
        AngleAxisAndCenterToCamera(angle_axis, center,camera);      // 求取camera中的旋转平移数组

        if(translation_sigma > 0.0)     // 检查平移中添加的噪声
            PerturbPoint3(translation_sigma, camera + camera_block_size() - 6);
    }
}

