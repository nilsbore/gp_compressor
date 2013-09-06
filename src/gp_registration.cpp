#include "gp_registration.h"

using namespace Eigen;

gp_registration::gp_registration(pointcloud::ConstPtr cloud, float res, int sz) : gp_compressor(cloud, res, sz),
    mean1(0), mean2(0), accumulated_weight(0)
{
    covariance.setZero();
    project_cloud();
    std::cout << "Number of patches: " << S.size() << std::endl;
    train_processes();
}

void gp_registration::add_cloud(pointcloud::ConstPtr other_cloud)
{

}

void gp_registration::add_derivative(const Vector3d& x, const Vector3d& dx)
{
    weight = dx.norm();
    if (weight == 0.0f) {
        return;
    }
    dx *= step/weight;

    Vector3d cx = x + dx;

    accumulated_weight += weight;
    double alpha = weight / accumulated_weight;

    Vector3d diff1 = x - mean1;
    Vector3d diff2 = cx - mean2;
    covariance = (1.0f - alpha)*(covariance + alpha * (diff2 * diff1.transpose()));

    mean1 += alpha*diff1;
    mean2 += alpha*diff2;
}

void gp_registration::get_transformation(Matrix3d& R, Vector3d& t)
{
    JacobiSVD<Matrix3d> svd(covariance_, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Matrix3d& u = svd.matrixU();
    const Matrix3d& v = svd.matrixV();
    Matrix3d s;
    s.setIdentity();
    if (u.determinant()*v.determinant() < 0.0f) {
        s(2, 2) = -1.0f;
    }
    R = u * s * v.transpose();
    t = mean2_ - r*mean1_;
}
