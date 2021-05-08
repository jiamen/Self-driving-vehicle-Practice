//
// Created by zlc on 2021/1/15.
//

#include <iostream>
#include <pcl/io/pcd_io.h>

#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>


// 步骤1：读取点云信息 读取PCD文件中的点云信息
pcl::PointCloud<pcl::PointXYZ>::Ptr read_cloud_point(std::string const &file_path)
{
    // Loading first scan
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if( -1 == pcl::io::loadPCDFile<pcl::PointXYZ>(file_path, *cloud) )
    {
        PCL_ERROR("Couldn't read the pcd file\n");
        return nullptr;
    }

    return cloud;
}

// 步骤5：将配准以后的点云图可视化
void visualizer(pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud)
{
    // Initializing point cloud visualizer  初始化点云图可视化
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_final(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer_final->setBackgroundColor(0, 0, 0);

    // Coloring and visualizing target cloud (red).                 点云1 cloud1  目标点云（即我们已有的高精度地图）用红点绘制
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target_cloud, 255, 0, 0);
    viewer_final->addPointCloud<pcl::PointXYZ>(target_cloud, target_color, "target cloud");
    viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");


    // color and visualizing transformed input cloud (green).       点云2 cloud2  输入点云 用绿点绘制
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> output_color(output_cloud, 0, 255, 0);
    viewer_final->addPointCloud<pcl::PointXYZ>(output_cloud, output_color, "output cloud");
    viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "output cloud");


    // Starting visualizer
    viewer_final->addCoordinateSystem(1.0, "global");
    viewer_final->initCameraParameters();

    // Wait until visualizer window is closed.
    while ( !viewer_final->wasStopped() )
    {
        viewer_final->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}


int main(int argc, char* *argv)
{
    // 步骤1：读取点云信息 读取PCD文件中的点云信息
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud = read_cloud_point(argv[1]);           // 目标地图点云,点云1
    std::cout << "Loaded " << target_cloud->size() << " data points from cloud1.pcd" << std::endl;   // Loaded 112586 data points from cloud1.pcd

    auto input_cloud = read_cloud_point(argv[2]);           // 输入点云，待配准点云
    std::cout << "Loaded " << input_cloud->size() << " data points from cloud2.pcd" << std::endl;    // Loaded 112624 data points from cloud2.pcd

    // 步骤2：过滤输入点云,降采样
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);      // 定义过滤后的点云
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;      // 这是降采样的网格大小设置，不是NDT算法中的网格设置
    approximate_voxel_filter.setLeafSize(0.2, 0.2, 0.2);

    approximate_voxel_filter.setInputCloud(input_cloud);
    approximate_voxel_filter.filter(*filtered_cloud);                   // 得到过滤后的点云
    std::cout << "Filter cloud contains " << filtered_cloud->size() << " data points from cloud2.pcd" << std::endl;
    // Filtered cloud contains 12433 data points from cloud2.pcd      cloud2数量减少为10%, cloud1不变

    // 步骤3：初始化NDT并且设置NDT参数
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setTransformationEpsilon(0.01);        // 两次连续变换允许的最大的差值
    ndt.setStepSize(0.1);                     // 设置牛顿法 优化 的最大步长
    ndt.setResolution(1.0);                  // 将参考点云网格化，设置网格的大小, 这是NDT算法中的网格大小设置

    ndt.setMaximumIterations(35);          // 最大迭代次数
    ndt.setInputSource(filtered_cloud);         // 设置过滤后的源点云cloud2
    ndt.setInputTarget(target_cloud);           // 设置目标点云cloud1


    // 步骤:4：初始化变换参数并开始初始化
    Eigen::AngleAxisf init_rotation(0.6931, Eigen::Vector3f::UnitZ());
    Eigen::Translation3f init_translation(1.79387, 0.720047, 0);

    Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();       // NDT配准前需要提供变换初始位姿
    // 保存配准以后的点云图，输出到文件cloud3.pcd
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZ>);

    std::cout << "init_guess before: " << init_guess << std::endl;
    ndt.align(*output_cloud, init_guess);
    std::cout << "init_guess after: " << init_guess << std::endl;

    // hasConverged（） 获取收敛状态，注意，只要迭代过程符合上述三个终止条件之一，该函数返回true
    std::cout << "Normal Distribution Transform has converged:" << ndt.hasConverged()
              << "score: " << ndt.getFitnessScore() << std::endl;
    // Normal Distribution Transform has converged:1score: 0.638694
    // getFitnessScore（）用于获取迭代结束后目标点云和配准后的点云的最近点之间距离的均值。


    // getFinalTransformation () 获取最终的配准的转化矩阵，即原始点云到目标点云的刚体变换，返回Matrix4数据类型。
    std::cout << "ndt.getFinalTransformation() : " << ndt.getFinalTransformation() << endl;

    pcl::transformPointCloud(*input_cloud, *output_cloud, ndt.getFinalTransformation());
    pcl::io::savePCDFileASCII("../cloud3.pcd", *output_cloud);


    visualizer(target_cloud, output_cloud);


    return 0;
}

