#ifndef GAUSSIAN_PROCESS_H
#define GAUSSIAN_PROCESS_H

#include <Eigen/Dense>

class gaussian_process
{
private:
    float sigmaf_sq;
    float l_sq;
    float sigman_sq;
    Eigen::VectorXf alpha;
    Eigen::MatrixXf K;
    Eigen::MatrixXf X;
    Eigen::LLT<Eigen::MatrixXf> chol;
    void covariance_matrix(Eigen::MatrixXf& C, const Eigen::MatrixXf& Xi, const Eigen::MatrixXf& Xj, bool training = false);
    float squared_exp_distance(const Eigen::Vector2f& xi, const Eigen::Vector2f& xj);
public:
    void evaluate_points(Eigen::VectorXf& f_star, Eigen::VectorXf& V_star, const Eigen::MatrixXf& X_star);
    void add_measurements(const Eigen::MatrixXf& nX, const Eigen::VectorXf& y);
    gaussian_process(float sigmaf = 0.05, float l = 3, float sigman = 0.04);
    //gaussian_process(float sigmaf, float l, float sigman);
};

#endif // GAUSSIAN_PROCESS_H
