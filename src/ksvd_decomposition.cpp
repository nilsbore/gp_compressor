#include "ksvd_decomposition.h"
#include "orthogonal_matching_pursuit.h"

#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace Eigen;

ksvd_decomposition::ksvd_decomposition(MatrixXf& X, MatrixXi& I, MatrixXf& D, std::vector<int>& number_words,
                                       const MatrixXf& S, const Array<bool, Dynamic, Dynamic>& W,
                                       int dict_size, int words_max, float proj_error, float stop_diff) :
    X(X), I(I), D(D), S(S), W(W), number_words(number_words),
    dict_size(dict_size), words_max(words_max), proj_error(proj_error),
    stop_diff(stop_diff), l(S.rows()), n(S.cols())
{
    X.resize(words_max, n);
    I.resize(words_max, n);
    D.resize(l, dict_size);
    number_words.resize(n);
    L.resize(dict_size);
    Lk.resize(dict_size);

    decompose();
}

void ksvd_decomposition::decompose()
{
    int mean_words;
    float error = 0;
    float last_error;
    // to initialize all positions of dictionary
    for (int j = 0; j < dict_size; ++j) {
        unused.push_back(j);
    }
    while (true) {
        replace_unused();
        mean_words = compute_code();
        optimize_dictionary();
        last_error = error;
        error = compute_error();
        std::cout << "Mean of used words: " << mean_words << std::endl;
        std::cout << "Mean error: " << error << std::endl;
        std::cout << "Unused words: " << unused.size() << std::endl;
        if (fabs(error - last_error) < stop_diff) {
            break;
        }
    }
}

int ksvd_decomposition::compute_code()
{
    orthogonal_matching_pursuit omp(D, words_max, proj_error, L, Lk);
    int mean_words = 0;
    VectorXf Xi(words_max);
    VectorXi Ii(words_max);
    for (int i = 0; i < n; ++i) {
        number_words[i] = omp.omp_match_vector(Xi, Ii, S.col(i), W.col(i).cast<float>(), i);
        X.col(i) = Xi;
        I.col(i) = Ii;
        mean_words += number_words[i];
    }
    return mean_words / n;
}

float ksvd_decomposition::nipals_largest_singular(const MatrixXf& A, VectorXf& u, VectorXf& v)
{
    float threshold = 0.01;
    int max_iter = 20;
    float lambda = 0;
    u.resize(A.rows());
    u.setOnes();
    u /= sqrt(float(u.rows()));
    float old;
    for (int m = 0; m < max_iter; ++m) {
        v = A.transpose()*u;
        v.normalize();
        u = A*v;
        old = lambda;
        lambda = u.squaredNorm();
        //std::cout << (lambda - old)/lambda << " "; // for checking convergence threshold
        if ((lambda - old) < threshold*lambda) {
            break;
        }
    }
    //std::cout << std::endl; // for checking convergence threshold
    u /= sqrt(lambda);
    return sqrt(lambda);
}

void ksvd_decomposition::optimize_dictionary()
{
    MatrixXf SL;
    ArrayXXf WL;
    MatrixXf DXL;
    RowVectorXf XLj;
    VectorXf U(l);
    VectorXf V;
    float sigma;
    int ind;

    for (int j = 0; j < dict_size; ++j) {
        if (L[j].size() == 0) { // if this happens we should probably randomize a new vector
            unused.push_back(j);
            continue;
        }
        SL.resize(l, L[j].size());
        WL.resize(l, L[j].size()); // maybe test to have the mask as bool instead
        DXL.resize(l, L[j].size());
        DXL.setZero();
        XLj.resize(L[j].size());
        for (int i = 0; i < L[j].size(); ++i) {
            ind = L[j][i];
            //WL.col(i) = Array<bool, Dynamic, 1>::Map(masks[ind].data(), l).cast<float>();
            WL.col(i) = W.col(ind).cast<float>();
            //SL.col(i) = WL.col(i)*ArrayXf::Map(patches[ind].data(), l);
            SL.col(i) = WL.col(i)*S.col(ind).array();
            XLj(i) = X(Lk[j][i], ind);
            for (int k = 0; k < number_words[ind]; ++k) { // find a neater way to do this
                DXL.col(i) += X(k, ind)*D.col(I(k, ind));
            }
        }
        /*JacobiSVD<MatrixXf> svd(SL - (WL*(DXL-D.col(j)*XLj).array()).matrix(),
                                ComputeThinU | ComputeThinV);
        VectorXf U1 = svd.matrixU().col(0);
        VectorXf V1 = svd.matrixV().col(0);
        float s1 = svd.singularValues()(0);
        VectorXf U2, V2;
        float s2 = nipals_largest_singular(SL - (WL*(DXL-D.col(j)*XLj).array()).matrix(), U2, V2);
        std::cout << s1 - s2 << std::endl;
        std::cout << U1.transpose() - U2.transpose() << std::endl;
        std::cout << V1.transpose() - V2.transpose() << std::endl;
        exit(0);*/

        /*JacobiSVD<MatrixXf> svd(SL - (WL*(DXL-D.col(j)*XLj).array()).matrix(),
                                        ComputeThinU | ComputeThinV);
        U = svd.matrixU().col(0);
        V = svd.matrixV().col(0);
        sigma = svd.singularValues()(0);*/
        sigma = nipals_largest_singular(SL - (WL*(DXL-D.col(j)*XLj).array()).matrix(), U, V);
        D.col(j) = U;
        XLj = sigma*V;
        for (int i = 0; i < L[j].size(); ++i) {
            X(Lk[j][i], L[j][i]) = XLj(i);
        }
        L[j].clear();
        Lk[j].clear();
    }
}

float ksvd_decomposition::compute_error()
{
    ArrayXf mask(l);
    VectorXf residual(l);
    float error = 0;
    for (int i = 0; i < n; ++i)  { // mean squared norm of columns, running average
        //mask = Array<bool, Dynamic, 1>::Map(masks[i].data(), l).cast<float>();
        mask = W.col(i).cast<float>();
        //residual = mask*ArrayXf::Map(patches[i].data(), l);
        residual = mask*S.col(i).array();
        for (int k = 0; k < number_words[i]; ++k) {
            residual.array() -= X(k, i)*mask*D.col(I(k, i)).array();
        }
        error += residual.squaredNorm();
    }
    return error / float(n);
}

void ksvd_decomposition::replace_unused()
{
    std::vector<int> rnd;
    randomize_positions(rnd, unused.size());
    VectorXf s(l);
    for (int j = 0; j < unused.size(); ++j) {
        //temp = VectorXf::Map(patches[rnd[j]].data(), sz*sz);
        s = S.col(rnd[j]);
        float norm = s.norm();
        if (norm > 0) {
            s /= norm;
        }
        D.col(unused[j]) = s;
    }
    unused.clear();
}

void ksvd_decomposition::randomize_positions(std::vector<int>& rtn, int m)
{
    rtn.clear();
    if (m == 0) {
        return;
    }
    rtn.resize(m);
    std::srand(std::time(0)); // use current time as seed for random generator
    int ind;
    for (int j = 0; j < m; ++j) {
        do {
            ind = std::rand() % n;
        }
        while (std::find(rtn.begin(), rtn.end(), ind) != rtn.end());
        rtn[j] = ind;
    }
}
