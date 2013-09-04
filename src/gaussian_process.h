#ifndef GAUSSIAN_PROCESS_H
#define GAUSSIAN_PROCESS_H

#include <Eigen/Dense>

class gaussian_process
{
private:
    double sigmaf_sq;
    double l_sq;
    double sigman_sq;
    Eigen::VectorXd alpha;
    Eigen::MatrixXd K;
    Eigen::MatrixXd X;
    Eigen::LLT<Eigen::MatrixXd> chol;
    void covariance_matrix(Eigen::MatrixXd& C, const Eigen::MatrixXd& Xi, const Eigen::MatrixXd& Xj, bool training = false);
    double squared_exp_distance(const Eigen::Vector2d& xi, const Eigen::Vector2d& xj);
public:
    void predict_measurements(Eigen::VectorXd& f_star, const Eigen::MatrixXd& X_star, Eigen::VectorXd& V_star);
    void add_measurements(const Eigen::MatrixXd& nX, const Eigen::VectorXd& y);
    gaussian_process(double sigmaf = 0.05, double l = 3, double sigman = 0.04);
    //gaussian_process(double sigmaf, double l, double sigman);
};

#endif // GAUSSIAN_PROCESS_H
