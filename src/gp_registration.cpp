#include "gp_registration.h"

using namespace Eigen;

gp_registration::gp_registration()
{
}

double gp_compressor::kernel_dx(double x, double xk)
{

}

void gp_compressor::likelihood_dx(Vector2d& x, double y, Matrix3d& dx)
{
    VectorXf k;
    construct_covariance(x, BV);
    VectorXf k_dx(BV.cols());
    for (int i = 0; i < BV.cols(); ++i) {
        k_dx(i) = kernel_dx(x, BV.col(i));
    }
    double sigma_dx = 2*k.transpose()*C*k_dx;
    double sigma = s20 + k.transpose()*C*k_dx;
    double nomroot = y - alpha.transpose()*k;
    double nom = nomroot*nomroot;
    double expf = exp(-0.5f*nom/sigma);
    dx(0) = sigma_dx*expf +
            sigma*-0.5*(sigma_dx*nom - sigma*2.0f*alpha.transpose()*k_dx*nomroot)*expf;


    dx(2) = -nomroot/sigma*expf;
}
