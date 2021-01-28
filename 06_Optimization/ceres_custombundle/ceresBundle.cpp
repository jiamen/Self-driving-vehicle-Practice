//
// Created by zlc on 2021/1/24.
//

#include <iostream>
#include <fstream>
#include "ceres/ceres.h"

#include "SnavelyReprojectionError.h"
#include "common/BALProblem.h"
#include "common/BundleParams.h"

using namespace ceres;


// 第4个调用函数
void SetLinearSolver(ceres::Solver::Options* options, const BundleParams& params)
{
    CHECK(ceres::StringToLinearSolverType(params.linear_solver, &options->linear_solver_type));
    CHECK(ceres::StringToSparseLinearAlgebraLibraryType(params.linear_solver, &options->sparse_linear_algebra_library_type));
    CHECK(ceres::StringToDenseLinearAlgebraLibraryType(params.linear_solver, &options->dense_linear_algebra_library_type));
    // options->num_linear_solver_threads = params.num_threads;
    options->num_threads = params.num_threads;
}

// 第
void SetMinimizerOptions(Solver::Options* options, const BundleParams& params)
{
    options->max_num_iterations = params.num_iterations;
    options->minimizer_progress_to_stdout = true;
    options->num_threads = params.num_threads;
    // options->eta = params.eta;
    // options->max_solver_time_in_seconds = params.max_solver_time;

    CHECK(StringToTrustRegionStrategyType(params.trust_region_strategy,
                                                    &options->trust_region_strategy_type));
}

//
void SetSolveOptionsFromFlags(BALProblem* bal_problem, const BundleParams& params, Solver::Options*options)
{
    SetMinimizerOptions(options, params);               // 第 3 个调用函数
    SetLinearSolver(options, params);                   // 第 4 个调用函数

}

void BuildProblem(BALProblem* bal_problem, Problem* problem, const BundleParams& params)
{
    const int point_block_size  = bal_problem->point_block_size();
    const int camera_block_size = bal_problem->camera_block_size();
    double* points  = bal_problem->mutable_points();
    double* cameras = bal_problem->mutable_cameras();

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ..., u_n], where each u_i is two dimensional, the x and y position of the observation.
    const double* observations = bal_problem->observations();

    for(int i=0; i<bal_problem->num_observations(); ++ i)
    {
        CostFunction* cost_function;

        // Each Residual block takes a point and a camera as input and outputs a 2 dimensional Residual
        cost_function = SnavelyReprojectionError::Create(observations[2*i + 0], observations[2*i + 1]);

        // If enabled use Huber's loss function.
        LossFunction* loss_function = params->robustify ? new HuberLoss(1.0) : NULL;

        // Each observation corresponds to a camera and a point which are identified by camera_index()[i] and point_index()[i] respectively.
        double* camera = cameras + camera_block_size * bal_problem->camera_index()[i];
        double* point = points + point_block_size * bal_problem->point_index()[i];

        problem->AddResidualBlock(cost_function, loss_function, camera, point);
    }
}


void SolveProblem(const char* filename, const BundleParams& params)
{
    BALProblem bal_problem(filename);

    // show some information here ...
    std::cout << "bal problem file loaded ..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    // store the initial 3D cloud points and camera pose.
    if ( !params.initial_ply.empty() )
    {
        bal_problem.WriteToFile(params.initial_ply);
    }

    std::cout << "beginning problem ..." << std::endl;

    // add some noise for the initial std::endl
    srand(params.random_seed);
    bal_problem.Normalize();
    bal_problem.Perturb(params.rotation_sigma, params.translation_sigma,
                        params.point_sigma);

    std::cout << "Normalization complete..." << std::endl;

    Problem problem;
    BuildProblem(&bal_problem, &problem, params);                 // 第 1 个调用函数

    std::cout << "the problem is successfully build ..." << std::endl;

    Solver::Options options;
    SetSolverOptionsFromFlags();                                  // 第 2 个调用函数
}


int main(int argc, char* *argv)
{
    BundleParams params(argc, argv);        // set the parameters here

    google::InitGoogleLogging(argv[0]);
    std::cout << params.input << std::endl;

    if( params.input.empty() )
    {
        std::cout << "Usage: bundle_adjuster -input <path for dataset>";
        return 1;
    }

    SolveProblem(params.input.c_str(), params);

    return 0;
}

