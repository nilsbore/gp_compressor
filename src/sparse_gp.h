#ifndef SPARSE_GP_H
#define SPARSE_GP_H

#include <Eigen/Dense>
#include <vector>

class sparse_gp
{
private:
    // parameters of covariance function:
    double sigmaf_sq;
    double l_sq;
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
    double kernel_function(const Eigen::Vector2d& xi, const Eigen::Vector2d& xj);
    void shuffle(std::vector<int>& ind, int n);
    void kernel_dx(Eigen::MatrixXd k_dx, const Eigen::Vector2d& x);
public:
    void add_measurements(const Eigen::MatrixXd& X,const Eigen::VectorXd& y);
    void predict_measurements(Eigen::VectorXd& f_star, const Eigen::MatrixXd& X_star,
                              Eigen::VectorXd& sigconf, bool conf = false);
    double log_prob(const Eigen::VectorXd& X_star, const Eigen::VectorXd& f_star);
    void likelihood_dx(Eigen::Vector3d& dx, const Eigen::Vector2d& x, double y);
    sparse_gp(int capacity = 20, double s0 = 1e-2f, double sigmaf = 1e-2f, double l = 0.08*0.08);
};

#endif // SPARSE_GP_H
