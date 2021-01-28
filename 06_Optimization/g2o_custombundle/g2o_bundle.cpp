//
// Created by zlc on 2021/1/25.
//

#include <Eigen/StdVector>
#include <Eigen/Core>

#include <iostream>
#include <stdint.h>

#include <unordered_set>
#include <memory>
#include <vector>
#include <stdlib.h>

#include "g2o/stuff/sampler.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_dogleg.h"

#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#include "g2o/solvers/structure_only/structure_only_solver.h"

#include "common/BundleParams.h"
#include "common/BALProblem.h"
#include "g2o_bal_class.h"              // 自定义的类


using namespace Eigen;
using namespace std;

typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;
typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BalBlockSolver;


// △△△ 第 6 个调用函数 ☆☆☆
void WriteToBALProblem(BALProblem* bal_problem, g2o::SparseOptimizer* optimizer)
{
    const int num_points  = bal_problem->num_points();          // 22106
    const int num_cameras = bal_problem->num_cameras();         // 16
    const int camera_block_size = bal_problem->camera_block_size();
    const int point_block_size  = bal_problem->point_block_size();

    double* raw_cameras = bal_problem->mutable_cameras();
    for( int i=0; i<num_cameras; ++ i )
    {
        VertexCameraBAL* pCamera = dynamic_cast<VertexCameraBAL*>(optimizer->vertex(i));
        Eigen::VectorXd NewCameraVec = pCamera->estimate();
        memcpy(raw_cameras + i*camera_block_size, NewCameraVec.data(), sizeof(double) * camera_block_size);
    }

    double* raw_points = bal_problem->mutable_points();
    for( int j=0; j<num_points; ++ j )
    {
        VertexPointBAL* pPoint = dynamic_cast<VertexPointBAL*>(optimizer->vertex(j + num_cameras));
        Eigen::Vector3d NewPointVec = pPoint->estimate();
        memcpy(raw_points + j*point_block_size, NewPointVec.data(), sizeof(double) * point_block_size);
    }
}


// △△△ 第 3 个调用函数 ☆☆☆ set up the vertexes and edges for the bundle adjustment.
void BuildProblem(const BALProblem* bal_problem, g2o::SparseOptimizer* optimizer, const BundleParams& params)
{
    const int num_points = bal_problem->num_points();
    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();     // 10 ? 9
    const int point_block_size  = bal_problem->point_block_size();      // 3

    // Set camera vertex with initial value in the dataset.
    const double* raw_cameras = bal_problem->cameras();         // 16*9+22106*3 = 66462 个优化变量的空间起始位置地址
    for( int i=0; i<num_cameras; ++ i )
    {
        ConstVectorRef temVecCamera(raw_cameras + camera_block_size * i, camera_block_size);
        VertexCameraBAL* pCamera = new VertexCameraBAL();
        pCamera->setEstimate(temVecCamera);     // initial value for the camera i ...
        pCamera->setId(i);                      // set id for each camera vertex

        // remeber to add vertex into optimizer.
        optimizer->addVertex(pCamera);
    }

    // Set point vertex with initial value in the dataset.
    const double* raw_points = bal_problem->points();
    // const int point_block_size = bal_problem->points();
    for( int j=0; j<num_points; ++ j )
    {
        ConstVectorRef temVecPoint(raw_points + point_block_size * j, point_block_size);
        VertexPointBAL* pPoint = new VertexPointBAL();
        pPoint->setEstimate(temVecPoint);       // initial value for the camera i ...
        pPoint->setId(j + num_cameras);                       // set id for each camera vertex

        // remeber to add vertex into optimizer.
        pPoint->setMarginalized(true);
        optimizer->addVertex(pPoint);
    }


    // Set edges for graph.
    const int num_observations = bal_problem->num_observations();
    const double* observations = bal_problem->observations();        // pointer for the first observation...
    for( int i=0; i < num_observations; ++ i )
    {
        EdgeObservationBAL* bal_edge = new EdgeObservationBAL();

        const int camera_id = bal_problem->camera_index()[i];       // get id for the camera;
        const int point_id  = bal_problem->point_index()[i] + num_cameras;  // get id for the point.

        if( params.robustify )
        {
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            rk->setDelta(1.0);
            bal_edge->setRobustKernel(rk);
        }

        // set the vertex by the ids for an edge observation
        bal_edge->setVertex(0, dynamic_cast<VertexCameraBAL*>(optimizer->vertex(camera_id)) );
        bal_edge->setVertex(1, dynamic_cast<VertexPointBAL*>(optimizer->vertex(point_id)) );
        bal_edge->setInformation(Eigen::Matrix2d::Identity());
        bal_edge->setMeasurement(Eigen::Vector2d(observations[2*i + 0], observations[2*i + 1]) );

        optimizer->addEdge(bal_edge);
    }
}

// △△△ 第 2 个调用函数 ☆☆☆
void SetSolverOptionsFromFlags(BALProblem* bal_problem, const BundleParams& params, g2o::SparseOptimizer* optimizer)
{
    /* 变量使用顺序：linearSolver -> solver_ptr -> solver -> optimizer， 确定 求解方法 和 H△x=g具体的函数形式 */

    g2o::LinearSolver<BalBlockSolver::PoseMatrixType>* linearSolver = 0;
    printf("params.linear_solver = %s\n", params.linear_solver.c_str());
    /** 确定是稠密舒尔补还是稀疏舒尔补，即确定 H△x = g 的  求解方法  ，课本P252页 */
    if ( params.linear_solver == "dense_schur" )        // 默认使用稠密计算方法
    {
        linearSolver = new g2o::LinearSolverDense<BalBlockSolver::PoseMatrixType>();
    }
    else if ( params.linear_solver == "sparse_schur" )
    {
        linearSolver = new g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>();
        // 让solver对矩阵进行排序保持稀疏性
        dynamic_cast< g2o::LinearSolverCholmod<BalBlockSolver::PoseMatrixType>* > (linearSolver)->setBlockOrdering(true);
        // AMD ordering, only needed for sparse cholesky solver
    }


    BalBlockSolver* solver_ptr;
    solver_ptr = new BalBlockSolver(linearSolver);

    // SetLinearSolver(solver_ptr, params);                     // △△△ 第 4 个调用函数
    // SetMinimizerOptions(solver_ptr, params, optimizer);      // △△△ 第 5 个调用函数

    /** 确定H△x = g的  具体形式  ， 课本P248 */
    g2o::OptimizationAlgorithmWithHessian* solver;
    printf("params.trust_region_strategy = %s\n", params.trust_region_strategy.c_str());
    if( params.trust_region_strategy == "levenberg_marquardt" ) // 默认使用LM下降法
    {
        solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    }
    else if( params.trust_region_strategy == "dogleg" )
    {
        solver = new g2o::OptimizationAlgorithmDogleg(solver_ptr);
    }
    else
    {
        std::cout << "Please check your trust_region_strategy parameter again..." << std::endl;
        exit(EXIT_FAILURE);
    }

    optimizer->setAlgorithm(solver);
}


// △△△ 第 1 个调用函数 ☆☆☆
void SolveProblem(const char* filename, const BundleParams& params)
{
    BALProblem bal_problem(filename);                   // 根据输入的problem-16-22106-pre.txt建立优化问题

    // show some information here ...
    std::cout << "bal problem file loaded ..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;


    // store the initial 3D cloud points and camera pose ...
    if( !params.initial_ply.empty() )   // initial_ply是string类
    {
        bal_problem.WriteToPLYFile(params.initial_ply);         // initial.ply文件就是在这里得到的
    }

    std::cout << "beginning problem..." << std::endl;


    /** 第1步：数据预处理 */
    // add some noise for the initial value
    srand(params.random_seed);
    bal_problem.Normalize();        // 对数据进行归一化
    bal_problem.Perturb(params.rotation_sigma, params.translation_sigma,
                        params.point_sigma);        // 给数据加上噪声（相机旋转、相机平移、路标点）
    std::cout << "Normalization complete..." << std::endl;


    /** 第2步：构建BA问题 */
    g2o::SparseOptimizer optimizer;
    // △△△ 第 2 个调用函数 ☆☆☆  BA问题会得到损失函数（cost function），根据选定的方法（LM,BN,orDogLeg）确定H△x=g具体的增量方程 和 求解方法
    SetSolverOptionsFromFlags(&bal_problem, params, &optimizer);        // 使用用户的输入参数来设置优化求解
    // △△△ 第 3 个调用函数 ☆☆☆
    BuildProblem(&bal_problem, &optimizer, params);                     // 完成对BA问题目标函数的构造


    /** 第3步：执行优化 */
    // perform the optimization
    std::cout << "begin optimization ..." << std::endl;
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(params.num_iterations);


    std::cout << "Optimization complete ..." << std::endl;
    // write the optimization data into BALProblem class
    WriteToBALProblem(&bal_problem, &optimizer);                    // △△△ 第 6 个调用函数 ☆☆☆

    // write the result into a .ply file.
    if( !params.final_ply.empty() )
    {
        bal_problem.WriteToPLYFile(params.final_ply);
    }
}


int main(int argc, char* *argv)
{
    BundleParams params(argc, argv);                // set the parameters here.

    if( params.input.empty() )
    {
        std::cout << "Usage: bundle_adjuster - input <path for dataset>";
        return 1;
    }

    SolveProblem(params.input.c_str(), params);

    return 0;
}




