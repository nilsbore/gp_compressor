#ifndef RBF_KERNEL_H
#define RBF_KERNEL_H

#include <Eigen/Dense>


class rbf_kernel
{
private:
    double sigmaf_sq;
    double l_sq; // length parameter, how far points influence each other
public:
    double kernel_function(const Eigen::Vector2d& xi, const Eigen::Vector2d& xj);
    void kernel_dx(Eigen::MatrixXd& k_dx, const Eigen::Vector2d& x, const Eigen::MatrixXd& BV);
    void kernels_fast(Eigen::ArrayXXd& K_dx, Eigen::ArrayXXd& K_dy, const Eigen::MatrixXd& X, const Eigen::MatrixXd& BV);
    void construct_covariance_fast(Eigen::MatrixXd& K, const Eigen::MatrixXd& X, const Eigen::MatrixXd& BV);
    rbf_kernel(double sigmaf_sq = 1e-2f, double l_sq = 0.08*0.08);
};

#endif // RBF_KERNEL_H
