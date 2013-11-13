#include "sparse_gp_field.h"

#include <ctime>
#include <stdio.h> // until we removed the old debug printouts
#include <iostream>

#include "octave_convenience.h"

#define DEBUG 0

using namespace Eigen;

template <class Kernel, class Noise>
sparse_gp_field<Kernel, Noise>::sparse_gp_field(int capacity, double s0) :
    kernel(), noise(s0), capacity(capacity), s20(s0),
    eps_tol(1e-4f), current_size(0), total_count(0) // 1e-6f
{

}

template <class Kernel, class Noise>
int sparse_gp_field<Kernel, Noise>::size()
{
    return current_size;
}

// shuffle the data so that all neighbouring points aren't added at once
template <class Kernel, class Noise>
void sparse_gp_field<Kernel, Noise>::shuffle(std::vector<int>& ind, int n)
{
    ind.resize(n);
    for (int i = 0; i < n; ++i) {
        ind[i] = i;
    }
    int temp, r;
    for (int i = n-1; i > 0; --i) {
        r = rand() % i;
        temp = ind[i];
        ind[i] = ind[r];
        ind[r] = temp;
    }
}

//Add a chunk of data
template <class Kernel, class Noise>
void sparse_gp_field<Kernel, Noise>::add_measurements(const MatrixXd& X, const MatrixXd& Y)
{
    std::vector<int> ind;
    shuffle(ind, Y.rows());

    for (int i = 0; i < X.rows(); i++) {
        add(X.row(ind[i]).transpose(), Y.row(ind[i]).transpose()); // row transpose
    }

}

//Add this input and output to the GP
template <class Kernel, class Noise>
void sparse_gp_field<Kernel, Noise>::add(const VectorXd& X, const VectorXd& y)
{

    total_count++;
    if (DEBUG) {
        printf("\t\tsparse_gp_field adding %dth point to sparse_gp_field (current_size = %d):",total_count,current_size);
    }
    double kstar = kernel.kernel_function(X, X);

    if (current_size == 0){//First point is easy
        alpha.resize(1, y.rows());
        C.resize(1, 1);
        Q.resize(1, 1);
        //Equations 2.46 with q, r, and s collapsed
        alpha = y.transpose() / (kstar + s20); // resize alpha?
        C(0, 0) = -1.0f / (kstar + s20);
        Q(0, 0) = 1.0f / kstar; // K_1^-1

        current_size = 1;
        BV = X;
        if (DEBUG) {
            printf("First\n");
        }
    }
    else {   //We already have data
        //perform the kernel
        VectorXd k;
        // this could be done a lot faster if we had a cc_fast for BV
        construct_covariance(k, X, BV);

        VectorXd m = alpha.transpose()*k;
        double s2 = kstar + k.transpose()*C*k;

        if (s2 < 1e-12f) {//For numerical stability..from Csato's Matlab code?
            if (DEBUG) {
                printf("s2 %lf ",s2);
            }
        }

        //Update scalars
        //page 33 - Assumes Gaussian noise, no assumptions on kernel?
        double r = noise.dx2_ln(y, m, s2); // -1.0f / (s20 + s2);
        //printf("out %d m %d r %f\n",out.Nrows(),m.Ncols(),r);
        RowVectorXd q;
        noise.dx_ln(q, y, m, s2); // y and m now vectors, will have to be redone

        //projection onto current BV
        VectorXd e_hat = Q*k;//Appendix G, section c
        //residual length
        // below equation 3.5

        double gamma = kstar - k.transpose()*e_hat;//Ibid

        if (gamma < 1e-12f) {//Numerical instability?
            if (DEBUG) {
                printf(" Gamma %lf ",gamma);
            }
            gamma = 0;
        }

        // these if statements should probably be the other way around, gamma < eps_tol ==> full update
        // No, sparse and full seem to be interchanged in Csato's thesis
        if (gamma < eps_tol && capacity != -1) {//Nearly singular, do a sparse update (e_tol), makes sure Q not singular
            if (DEBUG) {
                printf("Sparse %lf \n",gamma);
            }
            double eta = 1/(1 + gamma*r);//Ibid
            VectorXd s_hat = C*k + e_hat;//Appendix G section e
            alpha += s_hat*(q * eta);//Appendix G section f
            C += r*eta*s_hat*s_hat.transpose();//Ibid
        }
        else { //Full update
            if (DEBUG) {
                printf("Full!\n");
            }

            //s is of length N+1
            VectorXd s(k.rows() + 1);
            s.head(k.rows()) = C*k; // + e_hat, no only when sparse, OK!
            s(s.rows() - 1) = 1.0f; // shouldn't this be e_hat instead?

            //Add a row to alpha
            alpha.conservativeResize(alpha.rows() + 1, alpha.cols());
            alpha.row(alpha.rows() - 1).setZero();
            //Update alpha
            alpha += s*q;//Equations 2.46

            //Add Row and Column to C
            C.conservativeResize(C.rows() + 1, C.cols() + 1);
            C.row(C.rows() - 1).setZero();
            C.col(C.cols() - 1).setZero();
            //Update C
            C += r*s*s.transpose();//Ibid

            //Save the data, N++
            BV.conservativeResize(BV.rows(), current_size + 1);
            BV.col(current_size) = X;
            current_size++;

            //Add row and column to Gram Matrix
            Q.conservativeResize(Q.rows() + 1, Q.cols() + 1);
            Q.row(Q.rows() - 1).setZero();
            Q.col(Q.cols() - 1).setZero();

            //Add one more to ehat
            e_hat.conservativeResize(e_hat.rows() + 1);
            e_hat(e_hat.rows() - 1) = -1.0f;
            //Update gram matrix
            Q += 1.0f/gamma*e_hat*e_hat.transpose();//Equation 3.5

        }

        //Delete BVs if necessary...maybe only 2 per iteration?
        while (current_size > capacity && capacity > 0) { //We're too big!
            double minscore = 0, score;
            int minloc = -1;
            //Find the minimum score
            for (int i = 0; i < current_size; i++) {
                // Equation 3.26
                score = alpha.row(i).squaredNorm()/(Q(i, i) + C(i, i));
                if (i == 0 || score < minscore) {
                    minscore = score;
                    minloc = i;
                }
            }
            //Delete it
            delete_bv(minloc);
            if (DEBUG) {
                printf("Deleting for size %d\n", current_size);
            }
        }

        //Delete for geometric reasons - Loop?
        double minscore = 0, score;
        int minloc = -1;
        while (minscore < 1e-9f && current_size > 1) {
            for (int i = 0; i < current_size; i++) {
                score = 1.0f/Q(i, i);
                if (i == 0 || score < minscore) {
                    minscore = score;
                    minloc = i;
                }
            }
            if (minscore < 1e-9f) {
                delete_bv(minloc);
                if (DEBUG) {
                    printf("Deleting for geometry\n");
                }
            }
        }

    }
    if (isnan(C(0, 0))) {
        printf("sparse_gp_field::C has become Nan\n");
    }

}

//Delete a BV.  Very messy
template <class Kernel, class Noise>
void sparse_gp_field<Kernel, Noise>::delete_bv(int loc)
{
    //First swap loc to the last spot
    RowVectorXd alphastar = alpha.row(loc);
    alpha.row(loc) = alpha.row(alpha.rows() - 1);
    alpha.conservativeResize(alpha.rows() - 1, alpha.cols());

    //Now C
    double cstar = C(loc, loc);
    VectorXd Cstar = C.col(loc);
    Cstar(loc) = Cstar(Cstar.rows() - 1);
    Cstar.conservativeResize(Cstar.rows() - 1);

    VectorXd Crep = C.col(C.cols() - 1);
    Crep(loc) = Crep(Crep.rows() - 1);
    C.row(loc) = Crep.transpose();
    C.col(loc) = Crep;
    C.conservativeResize(C.rows() - 1, C.cols() - 1);

    //and Q
    double qstar = Q(loc, loc);
    VectorXd Qstar = Q.col(loc);
    Qstar(loc) = Qstar(Qstar.rows() - 1);
    Qstar.conservativeResize(Qstar.rows() - 1);
    VectorXd Qrep = Q.col(Q.cols() - 1);
    Qrep(loc) = Qrep(Qrep.rows() - 1);
    Q.row(loc) = Qrep.transpose();
    Q.col(loc) = Qrep;
    Q.conservativeResize(Q.rows() - 1, Q.cols() - 1);

    //Ok, now do the actual removal  Appendix G section g
    VectorXd qc = (qstar + cstar)*(Qstar + Cstar);
    for (int i = 0; i < alpha.cols(); ++i) {
        alpha.col(i) -= alphastar(i)*qc;
    }
    C += (Qstar * Qstar.transpose()) / qstar -
            ((Qstar + Cstar)*(Qstar + Cstar).transpose()) / (qstar + cstar);
    Q -= (Qstar * Qstar.transpose()) / qstar;

    //And the BV
    BV.col(loc) = BV.col(BV.cols() - 1);
    BV.conservativeResize(BV.rows(), BV.cols() - 1);

    current_size--;
}


//Predict on a chunk of data.
template <class Kernel, class Noise>
void sparse_gp_field<Kernel, Noise>::predict_measurements(MatrixXd& f_star, const MatrixXd& X_star, VectorXd& sigconf, bool conf)
{
    //printf("sparse_gp_field::Predicting on %d points\n",in.Ncols());
    std::cout << "Alpha.cols: " << alpha.cols() << std::endl;
    f_star.resize(X_star.rows(), alpha.cols());
    sigconf.resize(X_star.rows());
    VectorXd temp(alpha.cols());
    for (int c = 0; c < X_star.rows(); c++) {
        predict(temp, X_star.row(c).transpose(), sigconf(c), conf); // row transpose
        f_star.row(c) = temp.transpose();
    }
}

//Predict the output and uncertainty for this input.
// Make this work for several test inputs
template <class Kernel, class Noise>
void sparse_gp_field<Kernel, Noise>::predict(VectorXd& f_star, const VectorXd& X_star, double& sigma, bool conf)
{
    double kstar = kernel.kernel_function(X_star, X_star);
    VectorXd k;
    construct_covariance(k, X_star, BV);

    //f_star.resize(alpha.cols());
    if (current_size == 0) {
        f_star.setZero(); // gaussian around 0
        sigma = kstar + s20;
        if (DEBUG) {
            printf("No training points added before prediction\n");
        }
    }
    else {
        f_star = alpha.transpose()*k; //Page 33
        sigma = s20 + kstar + k.transpose()*C*k; //Ibid..needs s2 from page 19
        // here we predict y and not f, adding s20
    }

    if (sigma < 0) { //Numerical instability?
        printf("sparse_gp_field:: sigma (%lf) < 0!\n",sigma);
        sigma = 0;
    }

    //Switch to a confidence (0-100)
    if (conf) {
        //Normalize to one
        sigma /= kstar + s20;
        //switch diretion
        sigma = 100.0f*(1.0f - sigma);
    }
    else {
        sigma = sqrt(sigma);
    }

}

template <class Kernel, class Noise>
void sparse_gp_field<Kernel, Noise>::compute_likelihoods(VectorXd& l, const MatrixXd& X, const MatrixXd& Y)
{
    l.resize(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        l(i) = likelihood(X.row(i).transpose(), Y.row(i).transpose());
    }
}

// likelihood, without log for derivatives
template <class Kernel, class Noise>
double sparse_gp_field<Kernel, Noise>::likelihood(const Vector2d& x, const VectorXd& y)
{
    //This is pretty much prediction
    //double kstar = kernel_function(x, x);
    double kstar = kernel.kernel_function(x, x);
    VectorXd k;
    construct_covariance(k, x, BV);
    VectorXd mu(y.rows());
    double sigma;
    if (current_size == 0) { // no measurements, only prior gaussian around 0
        mu.setZero();
        sigma = kstar + s20; // maybe integrate this into likelihood_dx also
    }
    else {
        mu = alpha.transpose()*k; //Page 33
        sigma = s20 + kstar + k.transpose()*C*k;//Ibid..needs s2 from page 19
    }
    return 1.0f/sqrt(pow(2.0f*M_PI, double(y.rows()))*sigma)*exp(-0.5f/sigma*(y - mu).squaredNorm());
}

template <class Kernel, class Noise>
void sparse_gp_field<Kernel, Noise>::compute_derivatives(MatrixXd& dX, const MatrixXd& X, const MatrixXd& Y)
{
    dX.resize(X.rows(), 3);
    Vector3d dx;
    for (int i = 0; i < X.rows(); ++i) {
        likelihood_dx(dx, X.row(i).transpose(), Y.row(i).transpose());
        dX.row(i) = dx.transpose();
    }
}

// THIS NEEDS SOME SPEEDUP, PROBABLY BY COMPUTING SEVERAL AT ONCE
// the differential likelihood with respect to x and y
template <class Kernel, class Noise>
void sparse_gp_field<Kernel, Noise>::likelihood_dx(Vector3d& dx, const VectorXd& x, const VectorXd& y)
{
    VectorXd k;
    double k_star = kernel.kernel_function(x, x);
    construct_covariance(k, x, BV);
    MatrixXd k_dx;
    MatrixXd k_star_dx;
    //std::cout << "BV height " << BV.rows() << ", width " << BV.cols() << std::endl;
    //kernel_dx(k_dx, x);
    kernel.kernel_dx(k_dx, x, BV);
    //MatrixXd temp = x;
    kernel.kernel_dx(k_star_dx, x, x);
    Array2d sigma_dx = 2.0f*k_dx.transpose()*C*k + k_star_dx.transpose(); // with or without k_star?
    double sigma = s20 + k.transpose()*C*k + k_star;
    double sqrtsigma = sqrt(sigma);
    VectorXd offset = y - alpha.transpose()*k;
    double exppart = 0.5f/(sigma*sqrtsigma)*exp(-0.5f/sigma * offset.squaredNorm());
    Array2d firstpart = -sigma_dx;
    Array2d secondpart = 2.0f*k_dx.transpose()*alpha*offset;
    Array2d thirdpart = sigma_dx/sigma * offset.squaredNorm();
    dx(0) = 0;
    dx.tail<2>() = exppart*(firstpart + secondpart + thirdpart);
    if (isnan(dx(0)) || isnan(dx(1))) {
        //breakpoint();
    }

}

// kernel function, to be separated out later
template <class Kernel, class Noise>
void sparse_gp_field<Kernel, Noise>::construct_covariance(VectorXd& K, const Vector2d& X, const MatrixXd& Xv)
{
    K.resize(Xv.cols());
    for (int i = 0; i < Xv.cols(); ++i) {
        //K(i) = kernel_function(X, Xv.col(i));
        K(i) = kernel.kernel_function(X, Xv.col(i));
    }
}

// reset all parameters of gp so that it can be trained again
template <class Kernel, class Noise>
void sparse_gp_field<Kernel, Noise>::reset()
{
    total_count = 0;
    current_size = 0;
    alpha.resize(0, 0); // just to empty memory
    C.resize(0, 0);
    Q.resize(0, 0);
    BV.resize(0, 0);
}
