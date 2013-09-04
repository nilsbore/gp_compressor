#include "gaussian_process.h"

#include <iostream>

using namespace Eigen;

// hyperparameters
gaussian_process::gaussian_process(double sigmaf, double l, double sigman) :
    sigmaf_sq(sigmaf*sigmaf), l_sq(l*l), sigman_sq(sigman*sigman)
{

}

// maybe call this training
void gaussian_process::add_measurements(const MatrixXd& nX, const VectorXd& y)
{
    X = nX;
    //K.resize(y.rows(), y.rows());
    covariance_matrix(K, X, X, true); // true for adding diagonal noise
    MatrixXd C = K;
    C.diagonal().array() += sigman_sq;
    chol = C.llt();
    //std::cout << y.transpose() << std::endl;
    alpha = chol.solve(y); // compute (K + sigman^2)^-1*y
    //std::cout << alpha.transpose() << std::endl;
}

void gaussian_process::predict_measurements(VectorXd& f_star, const MatrixXd& X_star, VectorXd& V_star)
{
    MatrixXd K_star;
    covariance_matrix(K_star, X, X_star);
    f_star = K_star.transpose()*alpha;
    // do we need marked lower triangular here?
    //MatrixXd v = chol.matrixL().marked<LowerTriangular>().solveTriangular(K_star);
    MatrixXd v = chol.matrixL().solve(K_star);
    //MatrixXd K_starstar;
    //covariance_matrix(K_starstar, X_star, X_star);
    V_star.resize(X_star.rows());
    MatrixXd k_starstar(1, 1);
    for (int m = 0; m < X_star.rows(); ++m) {
        covariance_matrix(k_starstar, X_star.row(m), X_star.row(m));
        V_star(m) = k_starstar(0) - v.col(m).transpose()*v.col(m);
    }
    //V_star = K_starstar - v.transpose()*v; // diagonal the local variances
}

double gaussian_process::squared_exp_distance(const Vector2d& xi, const Vector2d& xj)
{
    return sigmaf_sq * exp(- 0.5f / l_sq * (xi - xj).squaredNorm());
}

void gaussian_process::covariance_matrix(MatrixXd& C, const MatrixXd& Xi, const MatrixXd& Xj, bool training)
{
    C.resize(Xi.rows(), Xj.rows());
    for (int i = 0; i < Xi.rows(); ++i) {
        for (int j = 0; j < Xj.rows(); ++j) {
            C(i, j) = squared_exp_distance(Xi.row(i).transpose(),
                                           Xj.row(j).transpose());
            if (training && i == j) {
                C(i, j) += sigman_sq;
            }
        }
    }
}
