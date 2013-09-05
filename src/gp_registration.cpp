#include "gp_registration.h"

using namespace Eigen;

gp_registration::gp_registration(pointcloud::ConstPtr cloud, float res, int sz) : gp_compressor(cloud, res, sz)
{
    project_cloud();
    std::cout << "Number of patches: " << S.size() << std::endl;
    train_processes();
}

void gp_registration::add_cloud(pointcloud::ConstPtr other_cloud)
{

}

void gp_registration::add(const Vector3f& x, const Vector3f& dx, const Vector3f& corresponding_point)
{
    if (weight==0.0f)
        return;

    ++no_of_samples_;
    accumulated_weight_ += weight;
    float alpha = weight/accumulated_weight_;

    Eigen::Vector3f diff1 = point - mean1_, diff2 = corresponding_point - mean2_;
    covariance_ = (1.0f-alpha)*(covariance_ + alpha * (diff2 * diff1.transpose()));

    mean1_ += alpha*(diff1);
    mean2_ += alpha*(diff2);
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
