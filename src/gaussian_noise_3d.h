#ifndef GAUSSIAN_NOISE_3D_H
#define GAUSSIAN_NOISE_3D_H

#include <Eigen/Dense>

class gaussian_noise_3d
{
private:
    double s20;
public:
    void dx_ln(Eigen::RowVectorXd& q, const Eigen::VectorXd& y, const Eigen::VectorXd& x, double sigma_x);
    double dx2_ln(const Eigen::VectorXd& y, const Eigen::VectorXd& x, double sigma_x);
    gaussian_noise_3d(double s20);
};

#endif // GAUSSIAN_NOISE_3D_H
