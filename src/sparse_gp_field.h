#ifndef SPARSE_GP_FIELD_H
#define SPARSE_GP_FIELD_H

#include <Eigen/Dense>
#include <vector>

template <class Kernel, class Noise>
class sparse_gp_field
{
public:
    typedef Kernel kernel_type;
    typedef Noise noise_type;
private:
    // parameters of covariance function:
    kernel_type kernel; // the gp kernel type used
    noise_type noise; // the noise model used
    int total_count; // How many points have I seen?
    int current_size; // how many inducing points do I have
    int capacity; // maximum number of inducing points
    double s20; // measurement noise, should be in noise instead
    double eps_tol; // error tolerance
    Eigen::MatrixXd alpha; // Alpha and C are the parameters of the GP
    // Alpha is NxDout
    Eigen::MatrixXd C;
    Eigen::MatrixXd Q; // Inverse Gram Matrix (K_t).  C and Q are NxN
    Eigen::MatrixXd BV; // The Basis Vectors, BV is 2xN
    void add(const Eigen::VectorXd& X, const Eigen::VectorXd& y);
    void delete_bv(int loc);
    void predict(Eigen::VectorXd& f_star, const Eigen::VectorXd& X_star, double& sigma, bool conf);
    void construct_covariance(Eigen::VectorXd& K, const Eigen::Vector2d& X, const Eigen::MatrixXd& Xv);
    void shuffle(std::vector<int>& ind, int n);
    void likelihood_dx(Eigen::Vector3d& dx, const Eigen::VectorXd& x, const Eigen::VectorXd& y);
    double likelihood(const Eigen::Vector2d& x, const Eigen::VectorXd& y);
public:
    void reset();
    int size();
    void add_measurements(const Eigen::MatrixXd& X,const Eigen::MatrixXd& Y);
    void predict_measurements(Eigen::MatrixXd& f_star, const Eigen::MatrixXd& X_star,
                              Eigen::VectorXd& sigconf, bool conf = false);
    void compute_derivatives(Eigen::MatrixXd& dX, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
    void compute_likelihoods(Eigen::VectorXd& l, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
    //sparse_gp_field(int capacity = 30, double s0 = 1e-1f, double sigmaf = 1e-2f, double l = 0.08*0.08);
    sparse_gp_field(int capacity = 100, double s0 = 1e2f);
};

#include "sparse_gp_field.hpp"

#endif // SPARSE_GP_FIELD_H
