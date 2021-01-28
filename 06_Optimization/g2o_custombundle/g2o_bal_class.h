//
// Created by zlc on 2021/1/25.
//

#ifndef _G2O_CUSTOMBUNDLE_G2O_BAL_CLASS_H_
#define _G2O_CUSTOMBUNDLE_G2O_BAL_CLASS_H_


#include <Eigen/Core>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"

#include "ceres/autodiff.h"

#include "common/tools/rotation.h"
#include "common/projection.h"


// 定义相机位姿节点
class VertexCameraBAL : public g2o::BaseVertex<9, Eigen::VectorXd>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 在生成定长的Matrix或Vector对象时，需要开辟内存，调用默认构造函数，通常x86下的指针是32位，内存位数没对齐就会导致程序运行出错。
    // 而对于动态变量(例如Eigen::VectorXd)会动态分配内存，因此会自动地进行内存对齐。
    VertexCameraBAL() {}

    virtual bool read(std::istream & /*is*/ )
    {
        return false;
    }

    virtual bool write(std::ostream & /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {

    }

    virtual void oplusImpl( const double* update )
    {
        Eigen::VectorXd::ConstMapType v( update, VertexCameraBAL::Dimension );
        _estimate += v;
    }
};

// 定义路标节点
class VertexPointBAL : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPointBAL() { }
    virtual bool read( std::istream& is )
    {
        return false;
    }

    virtual bool write( std::ostream& os ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {  }

    virtual void oplusImpl( const double* update ) override
    {
        Eigen::Vector3d::ConstMapType v(update);
        _estimate += v;
    }
};


// 下面开始定义边来表示节点之间的关系, 每一条边都代表一个代价函数 , 代价函数在P246，P259页也有
class EdgeObservationBAL : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexCameraBAL, VertexPointBAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeObservationBAL() { }

    virtual bool read( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write( std::ostream& /*is*/ ) const
    {
        return false;
    }

    // 覆盖基类函数，使用operator()计算代价
    virtual void computeError() override        // The virtual function comes from the Edge base class. Must define if you use edge.
    {
        const VertexCameraBAL* cam  = static_cast<const VertexCameraBAL*> ( vertex(0) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*>  ( vertex(1) );

        ( *this ) ( cam->estimate().data(), point->estimate().data(), _error.data() );
        // return the current estimate of the vertex            // _error是base_edge类中的保护成员变量
    }


    // 为了使用 Ceres 求导功能而定义的函数，让本类成为 拟函数类， 为了避免复杂的求导运算，我们借助Ceres库中的Autodiff（自动求导）功能
    template<typename T>
    bool operator() (const T* camera, const T* point, T* residuals) const
    {
        T predictions[2];
        CamProjectionWithDistortion( camera, point, predictions );        // 点point通过相机位姿参数camera后得到的 预测值predictions
        residuals[0] = predictions[0] - T( measurement()(0) );     // measurement : accessor functions for the measurement represented by the edge
        residuals[1] = predictions[1] - T( measurement()(1) );     // 残差 = 预测值 - 测量值

        return true;
    }

    virtual void linearizeOplus() override
    {
        // use numeric Jacobians
        // BaseBinaryEdge<2, Vector2d, VertexCameraBAL, VertexPointBAL>::linearizeOplus();
        // return;

        // using autodiff from ceres. Otherwise, the system will use g2o numerical diff for Jacobians.
        const VertexCameraBAL* cam  = static_cast<const VertexCameraBAL*> ( vertex(0) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*>  ( vertex(1) );
        typedef ceres::internal::AutoDiff<EdgeObservationBAL, double, VertexCameraBAL::Dimension, VertexPointBAL::Dimension> BalAutoDiff;

        Eigen::Matrix<double, Dimension, VertexCameraBAL::Dimension, Eigen::RowMajor> dError_dCamera;
        Eigen::Matrix<double, Dimension, VertexPointBAL::Dimension,  Eigen::RowMajor> dError_dPoint;
        double* parameters[] = { const_cast<double*>( cam->estimate().data() ), const_cast<double*>( point->estimate().data() ) };
        double* jacobians[] = { dError_dCamera.data(), dError_dPoint.data() };
        double value[Dimension];

        // Ceres 中的自动求导函数用法，需要提供 operator() 函数成员
        bool diffState = BalAutoDiff::Differentiate( *this, parameters, Dimension, value, jacobians );

        // copy over the jacobians(convert row-major -> column-major)
        if( diffState )
        {
            _jacobianOplusXi = dError_dCamera;
            _jacobianOplusXj = dError_dPoint;
        }
        else
        {
            assert( 0 && "Error while differentiating" );
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
        }
    }
};


#endif // _G2O_CUSTOMBUNDLE_G2O_BAL_CLASS_H_
