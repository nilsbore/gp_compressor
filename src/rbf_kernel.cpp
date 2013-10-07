#include "rbf_kernel.h"

using namespace Eigen;

rbf_kernel::rbf_kernel(double sigmaf_sq, double l_sq) :
    sigmaf_sq(sigmaf_sq), l_sq(l_sq)
{
}

// squared exponential coviariance, should use matern instead
double rbf_kernel::kernel_function(const Vector2d& xi, const Vector2d& xj)
{
    return sigmaf_sq*exp(-0.5f / l_sq * (xi - xj).squaredNorm());
}

// fast computation of kernel derivatives
void rbf_kernel::kernels_fast(ArrayXXd& K_dx, ArrayXXd& K_dy, const MatrixXd& X, const MatrixXd& BV)
{
    K_dx.resize(BV.cols(), X.cols());
    K_dy.resize(BV.cols(), X.cols());
    MatrixXd offset;
    ArrayXd exppart;
    ArrayXd temp;
    for (int i = 0; i < BV.cols(); ++i) {
        offset = X - BV.col(i).replicate(1, X.cols());
        temp = offset.colwise().squaredNorm();
        exppart = -sigmaf_sq/l_sq*(-0.5/l_sq*temp).exp();
        K_dx.row(i) = offset.row(0).array()*exppart.transpose();
        K_dy.row(i) = offset.row(1).array()*exppart.transpose();
    }
}

// the differential kernel vector with respect to x
void rbf_kernel::kernel_dx(MatrixXd& k_dx, const Vector2d& x, const MatrixXd& BV)
{
    k_dx.resize(BV.cols(), 2);
    RowVector2d offset;
    for (int i = 0; i < BV.cols(); ++i) {
        offset =  (x - BV.col(i)).transpose();
        k_dx.row(i) = -sigmaf_sq/l_sq*offset*exp(-0.5f/l_sq*offset.squaredNorm());
    }
}

// fast computation of covariance matrix
void rbf_kernel::construct_covariance_fast(MatrixXd& K, const MatrixXd& X, const MatrixXd& BV)
{
    K.resize(BV.cols(), X.cols());
    MatrixXd rep;
    ArrayXd temp;
    for (int i = 0; i < BV.cols(); ++i) {
        rep = BV.col(i).replicate(1, X.cols()); // typically more cols in X than in Xb
        temp = (X - rep).colwise().squaredNorm();
        K.row(i) = sigmaf_sq*(-0.5f/l_sq*temp).exp();
    }
}
