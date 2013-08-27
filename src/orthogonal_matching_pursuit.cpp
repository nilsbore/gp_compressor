#include "orthogonal_matching_pursuit.h"

#include <iostream>

using namespace Eigen;

// sending the dictionary in here might be bad for parallelization
orthogonal_matching_pursuit::orthogonal_matching_pursuit(const MatrixXf& D,
                                                         int words_max,
                                                         float proj_error,
                                                         std::vector<std::vector<int>>& L, std::vector<std::vector<int>>& Lk) :
    D(D), words_max(words_max), proj_error(proj_error), L(L), Lk(Lk)
{
    //D_squared.array() *= D.array();
}

int orthogonal_matching_pursuit::max_abs_coeff(const ArrayXf& array)
{
    float max = 0;
    float val;
    int ind = -1;
    //std::cout << "rows: " << array.rows() << ", cols: " << array.cols() << std::endl;
    for (int m = 0; m < array.rows(); ++m) {
        val = array(m);
        if (!isnan(val) && !isinf(val) && fabs(val) > max) {
            ind = m;
            max = fabs(val);
        }
    }
    return ind;
}

int orthogonal_matching_pursuit::omp_match_vector(VectorXf& Xi, VectorXi& Ii, VectorXf s, VectorXf vmask, int i)
{
    Xi.resize(words_max);
    Ii.resize(words_max);

    ArrayXf mask = vmask.array();
    VectorXf residual(vmask.rows());
    ArrayXf weights(D.cols());

    VectorXf bk;
    VectorXf vk;
    double alphak, beta;
    MatrixXf Akinv(words_max, words_max);
    VectorXf xk(D.rows());

    residual = mask*s.array();
    int ind, k;
    for (k = 0; k < words_max; ++k) {
        if (residual.squaredNorm() < proj_error) { // we could stop if weight too small also
            break;
        }
        weights = residual.transpose()*D;
        for (int m = 0; m < k; ++m) {
            weights(Ii(m)) = 0;
        }
        weights.abs().maxCoeff(&ind);
        Xi(k) = weights(ind);
        Ii(k) = ind;

        L[ind].push_back(i); // this is done because we do it in the opposite way her
        Lk[ind].push_back(k); // when recreating, focus lies on recreating the patches fast

        if (k > 0) {
            // hitta korrelationen med de tidigare
            if (k == 1) { // flytta upp
                Akinv(0, 0) = 1;
            }
            else {
                beta = 1.0f/(1.0f - vk.transpose()*bk); // we can reuse the one from the prev iter here
                Akinv.block(0, 0, k-1, k-1) += beta*bk*bk.transpose(); // kanske kolla k > 1
                Akinv.col(k-1).head(k-1) = -beta*bk;
                Akinv.row(k-1).head(k-1) = -beta*bk.transpose();
                Akinv(k-1, k-1) = beta; // fortfarande assignment för k == 1
            }

            vk.resize(k);
            xk = mask.array()*D.col(ind).array();
            for (int m = 0; m < k; ++m) {
                // xk is needed for the masking;
                vk(m) = xk.transpose()*D.col(Ii(m));
            }

            bk = Akinv.block(0, 0, k, k)*vk;
            alphak = Xi(k)/(1.0f - vk.transpose()*bk); // assuming || x_k+1 || = 1
            Xi(k) = alphak;
            Xi.head(k) -= alphak*bk; // head på bk?
        }
        residual = mask*s.array();
        for (int m = 0; m <= k; ++m) {
            residual.array() -= Xi(m)*mask*D.col(Ii(m)).array(); // this is unclear why it works so good, norm of masked vector not 1
        }
    }

    return k;
}

int orthogonal_matching_pursuit::mp_match_vector(VectorXf& Xi, VectorXi& Ii, VectorXf s, VectorXf vmask, int i)
{
    Xi.resize(words_max);
    Ii.resize(words_max);

    ArrayXf mask = vmask.array();
    VectorXf residual(vmask.rows());
    ArrayXf weights(D.cols());

    residual = mask*s.array();
    int ind, k;
    for (k = 0; k < words_max; ++k) {
        if (residual.squaredNorm() < proj_error) {
            break;
        }
        weights = residual.transpose()*D;
        for (int m = 0; m < k; ++m) {
            weights(Ii(m)) = 0;
        }
        weights.abs().maxCoeff(&ind);
        Xi(k) = weights(ind);
        Ii(k) = ind;
        residual.array() -= Xi(k)*mask*D.col(ind).array();
        L[ind].push_back(i); // this is done because we do it in the opposite way her
        Lk[ind].push_back(k); // when recreating, focus lies on recreating the patches fast
    }

    return k;
}

//int orthogonal_matching_pursuit::match_vector(VectorXf& Xi, VectorXi& Ii, VectorXf s, VectorXf vmask, int i)
//{
//    if (vmask.sum() == 0) {
//        return 0;
//    }
//    Xi.resize(words_max);
//    Ii.resize(words_max);

//    ArrayXf mask = vmask.array();
//    VectorXf residual(vmask.rows());
//    ArrayXf weights(D.cols());

//    VectorXf bk;
//    VectorXf vk;
//    double alphak, beta;
//    MatrixXf Akinv(words_max, words_max);
//    VectorXf xk(D.rows());

//    ArrayXf norms = vmask.transpose()*D_squared;
//    norms = norms.sqrt();

//    residual = mask*s.array();
//    int ind, k;
//    for (k = 0; k < words_max; ++k) {
//        if (residual.squaredNorm() < proj_error) { // we could stop if weight too small also
//            break;
//        }
//        weights = residual.transpose()*D;
//        weights /= norms;
//        for (int m = 0; m < k; ++m) {
//            weights(Ii(m)) = 0;
//        }
//        ind = max_abs_coeff(weights);
//        if (ind == -1) {
//            break;
//        }
//        Xi(k) = weights(ind);
//        Ii(k) = ind;

//        L[ind].push_back(i); // this is done because we do it in the opposite way her
//        Lk[ind].push_back(k); // when recreating, focus lies on recreating the patches fast

//        //norms(k) = xk.norm();

//        if (k > 0) {
//            // hitta korrelationen med de tidigare
//            if (k == 1) { // flytta upp
//                Akinv(0, 0) = 1;
//            }
//            else {
//                beta = 1.0f/(1.0f - vk.transpose()*bk); // we can reuse the one from the prev iter here
//                Akinv.block(0, 0, k-1, k-1) += beta*bk*bk.transpose(); // kanske kolla k > 1
//                Akinv.col(k-1).head(k-1) = -beta*bk;
//                Akinv.row(k-1).head(k-1) = -beta*bk.transpose();
//                Akinv(k-1, k-1) = beta; // fortfarande assignment för k == 1
//            }

//            vk.resize(k);
//            xk = mask.array()*D.col(ind).array() / norms(ind);
//            for (int m = 0; m < k; ++m) {
//                // xk is needed for the masking
//                //vk(m) = 1.0f / (norms_t(Ii(k)) * norms_t(Ii(m))) * xk.transpose()*D.col(Ii(m));
//                vk(m) = 1.0f / norms(Ii(m))*xk.transpose()*D.col(Ii(m));
//            }

//            bk = Akinv.block(0, 0, k, k)*vk;
//            alphak = Xi(k)/(1.0 - vk.transpose()*bk); // assuming || x_k+1 || = 1
//            Xi(k) = alphak;
//            Xi.head(k) -= alphak*bk; // head på bk?
//        }
//        residual = mask*s.array();
//        for (int m = 0; m <= k; ++m) {
//            residual.array() -= Xi(m)/norms(Ii(m))*mask*D.col(Ii(m)).array();
//        }
//    }
//    for (int m = 0; m <= k; ++m) {
//        //Xi(m) /= norms(Ii(m));
//    }

//    return k;
//}
