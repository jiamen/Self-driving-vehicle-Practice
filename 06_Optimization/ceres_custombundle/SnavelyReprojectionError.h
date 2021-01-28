//
// Created by zlc on 2021/1/24.
//

#ifndef _SnavelyReprojection_h_
#define _SnavelyReprojection_h_

#include <iostream>
#include "ceres/ceres.h"

#include "common/tools/rotation.h"
#include "common/projection.h"

class SnavelyReprojectionError
{
private:
    double observed_x;
    double observed_y;


public:
    SnavelyReprojectionError(double observation_x, double observation_y):observed_x(observation_x), observed_y(observation_y){}

    template<class T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        // camera[0, 1, 2] are the angle-axis rotation
        T predictions[2];
        CamProjectionWithDistortion(camera, point, predictions);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y)
    {
        //这里面残差是二维 待优化的是相机位姿以及3d点 所以自动求导模板中的数字2,9,3分别是它们的维度
        return ( new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(observed_x, observed_y)) );
        // 这里参数的意思解释如下：
        // 每个残差值依赖于空间点的位置（3个参数）和相机参数（9个参数）。
        // 9个相机参数包含用Rodriques’ axis-angle表示的旋转向量（3个参数），平移参数（3个参数），以及焦距和两个径向畸变参数(3个参数)。
    }
};

#endif  // SnavelyReprojection.h