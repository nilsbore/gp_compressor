#ifndef SPARSE_GP_H
#define SPARSE_GP_H

#include <Eigen/Dense>
#include <vector>

template <class Kernel, class Noise>
class sparse_gp
{
public:
    typedef Kernel kernel_type;
    typedef Noise noise_type;
private:
    // parameters of covariance function:
    double sigmaf_sq;
    double l_sq;
    kernel_type kernel;
    noise_type noise;
    //private:
    int total_count; // How many points have I seen?
    int current_size; // how many inducing points do I have
    int capacity; // maximum number of inducing points
    double s20;
    double eps_tol; // error tolerance
    Eigen::VectorXd alpha; // Alpha and C are the parameters of the GP
    // Alpha is NxDout
    Eigen::MatrixXd C;
    Eigen::MatrixXd Q; // Inverse Gram Matrix (K_t).  C and Q are NxN
    Eigen::MatrixXd BV; // The Basis Vectors, BV is 2xN
    void add(const Eigen::VectorXd& X, double y);
    void delete_bv(int loc);
    double predict(const Eigen::VectorXd& X_star, double& sigma, bool conf);
    void construct_covariance(Eigen::VectorXd& K, const Eigen::Vector2d& X, const Eigen::MatrixXd& Xv);
    void shuffle(std::vector<int>& ind, int n);
    void likelihood_dx(Eigen::Vector3d& dx, const Eigen::Vector2d& x, double y);
    double likelihood(const Eigen::Vector2d& x, double y);
public:
    void train_parameters(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    void compute_derivatives_fast(Eigen::MatrixXd& dX, const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    int size();
    void add_measurements(const Eigen::MatrixXd& X,const Eigen::VectorXd& y);
    void predict_measurements(Eigen::VectorXd& f_star, const Eigen::MatrixXd& X_star,
                              Eigen::VectorXd& sigconf, bool conf = false);
    double log_prob(const Eigen::VectorXd& X_star, const Eigen::VectorXd& f_star);
    void compute_derivatives(Eigen::MatrixXd& dX, const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    void compute_likelihoods(Eigen::VectorXd& l, const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    //sparse_gp(int capacity = 30, double s0 = 1e-1f, double sigmaf = 1e-2f, double l = 0.08*0.08);
    sparse_gp(int capacity = 100, double s0 = 1e-1f);
};

#include "sparse_gp.hpp"

#endif // SPARSE_GP_H
