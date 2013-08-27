#ifndef ORTHOGONAL_MATCHING_PURSUIT_H
#define ORTHOGONAL_MATCHING_PURSUIT_H

#include <Eigen/Dense>
#include <vector>

class orthogonal_matching_pursuit
{
private:
    const Eigen::MatrixXf& D;
    //Eigen::MatrixXf D_squared;
    int words_max;
    float proj_error;
    // indices of patches using a dictinary entry in one algorithm pass
    std::vector<std::vector<int>>& L; // HA DESSA I KLASSEN ISTÄLLET!!!
    // the indices in I and X that these entries are associated to
    std::vector<std::vector<int>>& Lk;
    int max_abs_coeff(const Eigen::ArrayXf& array);
public:
    int omp_match_vector(Eigen::VectorXf& Xi, Eigen::VectorXi& Ii,
                        Eigen::VectorXf s, Eigen::VectorXf vmask, int i);
    int mp_match_vector(Eigen::VectorXf& Xi, Eigen::VectorXi& Ii,
                        Eigen::VectorXf s, Eigen::VectorXf vmask, int i);
    orthogonal_matching_pursuit(const Eigen::MatrixXf &D, int words_max, float proj_error,
                                std::vector<std::vector<int>>& L, std::vector<std::vector<int>>& Lk);
};

#endif // ORTHOGONAL_MATCHING_PURSUIT_H
