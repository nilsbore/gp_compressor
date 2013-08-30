#ifndef SPARSE_GP_H
#define SPARSE_GP_H

#include <Eigen/Dense>
#include <vector>

class sparse_gp
{
private:
    // parameters of covariance function:
    float sigmaf_sq;
    float l_sq;
    //private:
    int total_count; // How many points have I seen?
    int current_size; // how many inducing points do I have
    int capacity; // maximum number of inducing points
    float s20;
    float eps_tol; // error tolerance
    Eigen::VectorXf alpha; // Alpha and C are the parameters of the GP
    // Alpha is NxDout
    Eigen::MatrixXf C;
    Eigen::MatrixXf Q; // Inverse Gram Matrix.  C and Q are NxN
    Eigen::MatrixXf BV; // The Basis Vectors, BV is 2xN
    void add(const Eigen::VectorXf& X, float y);
    void delete_bv(int loc);
    float predict(const Eigen::VectorXf& X_star, float& sigma, bool conf);
    void construct_covariance(Eigen::VectorXf& K, const Eigen::Vector2f& X, const Eigen::MatrixXf& Xv);
    float kernel_function(const Eigen::Vector2f& xi, const Eigen::Vector2f& xj);
public:
    void add_measurements(const Eigen::MatrixXf& X,const Eigen::VectorXf& y);
    void predict_measurements(Eigen::VectorXf& f_star, const Eigen::MatrixXf& X_star,
                              Eigen::VectorXf& sigconf, bool conf);
    double log_prob(const Eigen::VectorXf& X_star, const Eigen::VectorXf& f_star);
    sparse_gp(int capacity, float s20, float sigmaf, float l);
};

#endif // SPARSE_GP_H
