#ifndef KSVD_DECOMPOSITION_H
#define KSVD_DECOMPOSITION_H

#include <Eigen/Dense>
#include <vector>

class ksvd_decomposition
{
private:
    Eigen::MatrixXf& X; // weights of the code words used
    Eigen::MatrixXi& I; // indices of code words
    Eigen::MatrixXf& D; // dictionary
    std::vector<int>& number_words;
    const Eigen::MatrixXf& S; // horizontally stacked set of vectors to be compressed
    const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& W; // binary mask
    int dict_size; // size of the dictionary
    int words_max; // maximum number of words for one entry
    float proj_error; // the error at which the orthogonal matching pursuit stops
    // the difference between iteration errors when the algorithm terminates
    float stop_diff;
    int l; // length of vectors in S, D
    int n; // number of vectors in S, e.g. number of patches
    // indices of patches using a dictinary entry in one algorithm pass
    std::vector<std::vector<int>> L;
    // the indices in I and X that these entries are associated to
    std::vector<std::vector<int>> Lk;
    // inidices of unused dictinary entries in one algorithm pass
    std::vector<int> unused;
    void decompose();
    int compute_code();
    float nipals_largest_singular(const Eigen::MatrixXf& A, Eigen::VectorXf& u, Eigen::VectorXf& v);
    void optimize_dictionary();
    float compute_error();
    void replace_unused();
    void randomize_positions(std::vector<int>& rtn, int m);
public:
    ksvd_decomposition(Eigen::MatrixXf& X, Eigen::MatrixXi& I, Eigen::MatrixXf& D,
                       std::vector<int>& number_words, const Eigen::MatrixXf& S,
                       const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>& W,
                       int dict_size, int words_max, float proj_error, float stop_diff);
};

#endif // KSVD_DECOMPOSITION_H
