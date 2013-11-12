#include "gaussian_noise_3d.h"

using namespace Eigen;

gaussian_noise_3d::gaussian_noise_3d(double s20) : s20(s20)
{

}

// d/dx ln P(y|x)
void gaussian_noise_3d::dx_ln(RowVectorXd& q, const VectorXd& y, const VectorXd& x, double sigma_x)
{
    q = (y - x).transpose()/(s20 + sigma_x);
}

// d2/dx2 ln P(y|x)
double gaussian_noise_3d::dx2_ln(const VectorXd& y, const VectorXd& x, double sigma_x)
{
    return -1.0f/(s20 + sigma_x);
}

