#include "sparse_gp.h"

#include <ctime>
#include <stdio.h> // until we removed the old debug printouts
#include <iostream>

#define DEBUG 1

using namespace Eigen;

sparse_gp::sparse_gp(int capacity, double s0, double sigmaf, double l) :
    capacity(capacity), s20(s0), sigmaf_sq(sigmaf), l_sq(l),
    eps_tol(1e-6f), current_size(0), total_count(0)
{
    std::cout << "new gaussian process object!" << std::endl;
}

void sparse_gp::shuffle(std::vector<int>& ind, int n)
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
void sparse_gp::add_measurements(const MatrixXd& X,const VectorXd& y)
{
    std::vector<int> ind;
    shuffle(ind, y.rows());

    /*std::cout << "[ ";
    for (int i = 0; i < X.rows(); ++i) {
        std::cout << X.row(ind[i]);
        if (i < X.rows() - 1) {
            std::cout << " ;" << std::endl;
        }
        else {
            std::cout << " ]" << std::endl;
        }
    }

    std::cout << "[ ";
    for (int i = 0; i < y.rows(); ++i) {
        std::cout << y(ind[i]) << " ";
    }
    std::cout << "]" << std::endl;*/

    for (int i = 0; i < X.rows(); i++) {
        add(X.row(ind[i]).transpose(), y(ind[i])); // row transpose
    }

}

//Add this input and output to the GP
void sparse_gp::add(const VectorXd& X, double y)
{

    total_count++;
    if (DEBUG) {
        printf("\t\tsparse_gp adding %dth point to sparse_gp (current_size = %d):",total_count,current_size);
    }
    double kstar = kernel_function(X, X);
    std::cout << "kstar: " << kstar << std::endl;

    if (current_size == 0){//First point is easy
        alpha.resize(1);
        C.resize(1, 1);
        Q.resize(1, 1);
        //Equations 2.46 with q, r, and s collapsed
        alpha(0) = y / (kstar + s20); // resize alpha?
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
        construct_covariance(k, X, BV);

        double m = alpha.transpose()*k;
        double s2 = kstar + k.transpose()*C*k;

        if (s2 < 1e-12f) {//For numerical stability..from Csato's Matlab code?
            if (DEBUG) {
                printf("s2 %lf ",s2);
            }
            //s2 = 1e-12f; // this is where the nan comes from
        }

        //Update scalars
        //page 33 - Assumes Gaussian noise, no assumptions on kernel?
        double r = -1.0f / (s20 + s2); // should check these ones out
        //printf("out %d m %d r %f\n",out.Nrows(),m.Ncols(),r);
        double q = -r*(y - m);

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
            C += r*eta*(s_hat*s_hat.transpose());//Ibid,  TRY TO REMOVE PARENTHESIS FOR SPEED
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
            alpha.conservativeResize(alpha.rows() + 1);
            alpha(alpha.rows() - 1) = 0;
            //Update alpha
            alpha += q*s;//Equations 2.46

            //Add Row and Column to C
            C.conservativeResize(C.rows() + 1, C.cols() + 1);
            C.row(C.rows() - 1).setZero();
            C.col(C.cols() - 1).setZero();
            //Update C
            C += r*(s*s.transpose());//Ibid,  TRY TO REMOVE PARENTHESIS FOR SPEED

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
            Q += 1.0f/gamma*(e_hat*e_hat.transpose());//Equation 3.5, TRY TO REMOVE PARENTHESIS FOR SPEED

        }

        MatrixXd Q_inv = Q.inverse(); // DEBUG!!!

        //Delete BVs if necessary...maybe only 2 per iteration?
        while (current_size > capacity && capacity > 0) { //We're too big!
            double minscore = 0, score;
            int minloc = -1;
            //Find the minimum score
            for (int i = 0; i < current_size; i++) {
                // Equation 3.26
                score = alpha(i)*alpha(i)/(Q(i, i) + C(i, i)); // SumSquare()?
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
        while(minscore < 1e-9f && current_size > 1) {
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
        printf("sparse_gp::C has become Nan\n");
    }

}

//Delete a BV.  Very messy
void sparse_gp::delete_bv(int loc)
{
    //First swap loc to the last spot
    double alphastar = alpha(loc);
    alpha(loc) = alpha(alpha.rows() - 1);
    alpha.conservativeResize(alpha.rows() - 1);

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
    //VectorXd qc = (Qstar + Cstar)/(qstar + cstar);
    alpha -= alphastar/(qstar + cstar)*(Qstar + Cstar);
    C += (Qstar * Qstar.transpose()) / qstar -
            ((Qstar + Cstar)*(Qstar + Cstar).transpose()) / (qstar + cstar);
    Q -= (Qstar * Qstar.transpose()) / qstar;

    //And the BV
    BV.col(loc) = BV.col(BV.cols() - 1);
    BV.conservativeResize(BV.rows(), BV.cols() - 1);

    current_size--;
}


//Predict on a chunk of data.
void sparse_gp::predict_measurements(VectorXd& f_star, const MatrixXd& X_star, VectorXd& sigconf, bool conf)
{
    //printf("sparse_gp::Predicting on %d points\n",in.Ncols());
    f_star.resize(X_star.rows());
    sigconf.resize(X_star.rows());
    for (int c = 0; c < X_star.rows(); c++) {
        f_star(c) = predict(X_star.row(c).transpose(), sigconf(c), conf); // row transpose
    }
}

//Predict the output and uncertainty for this input.
// Make this work for several test inputs
double sparse_gp::predict(const VectorXd& X_star, double& sigma, bool conf)
{
    double kstar = kernel_function(X_star, X_star);
    VectorXd k;
    construct_covariance(k, X_star, BV);

    double f_star;
    if (current_size == 0) {
        sigma = kstar + s20;
        //We don't actually know the correct output dimensionality
        //So return nothing.
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
        printf("sparse_gp:: sigma (%lf) < 0!\n",sigma);
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

    return f_star;
}


//Log probability of this data
//Sigma should really be a matrix...
//Should this use Q or -C?  Should prediction use which?
double sparse_gp::log_prob(const VectorXd& X_star, const VectorXd& f_star)
{
    int dout = f_star.rows();
    static const double logsqrt2pi = 0.5f*log(2.0f*M_PI);
    double sigma;
    MatrixXd Sigma(dout, dout);
    Sigma.setIdentity(); //For now
    VectorXd mu(dout);

    //This is pretty much prediction
    double kstar = kernel_function(X_star, X_star);
    VectorXd k;
    construct_covariance(k, X_star, BV);
    if (current_size == 0) {
        sigma = kstar + s20;
        for (int i = 0; i < dout; i++) {
            mu(i) = 0;
        }
    }
    else {
        mu = (k.transpose()*alpha).transpose();//Page 33
        sigma = s20 + kstar + k.transpose()*C*k;//Ibid..needs s2 from page 19
        //printf("Making sigma: %lf %lf %lf %lf\n",s20,kstar,k(1),C(1,1));
    }
    Sigma *= sigma;
    double cent2 = (f_star - mu).squaredNorm(); //SumSquare()?

    // in eigen, we have to use decomposition to get log determinant
    long double log_det_cov = log(Sigma.determinant());

    long double lp = -double(dout)*logsqrt2pi - 0.5f*log_det_cov - 0.5f*cent2/sigma;
    //printf("\tCalculating log prob %Lf, Sigma = %lf, cent = %lf\n",lp,sigma,cent2);
    //if(C.Nrows()==1)
    //printf("\t\tC has one entry: %lf, k = %lf\n",C(1,1),k(1));
    return lp;
}

// kernel function, to be separated out later
void sparse_gp::construct_covariance(VectorXd& K, const Vector2d& X, const MatrixXd& Xv)
{
    K.resize(Xv.cols());
    for (int i = 0; i < Xv.cols(); ++i) {
        K(i) = kernel_function(X, Xv.col(i));
    }
}

// squared exponential coviariance, should use matern instead
double sparse_gp::kernel_function(const Vector2d& xi, const Vector2d& xj)
{
    return sigmaf_sq*exp(-0.5f / l_sq * (xi - xj).squaredNorm());
}

// linear kernel
/*double sparse_gp::kernel_function(const Vector2d& xi, const Vector2d& xj)
{
    return xi.transpose()*xj;
}*/

// polynomial kernel
/*double sparse_gp::kernel_function(const Vector2d& xi, const Vector2d& xj)
{
  double d = xi.rows();
  double resp = 1;
  double inner = xi.transpose()*xj;
  for (int i = 0; i < scales.Ncols(); i++) {
      resp += pow(inner / (d*scales(i)), i);
  }
  return resp;
}*/

// rbf kernel
/*double sparse_gp::kernel(const Vector2d& xi, const Vector2d& xj)
{
    int d = xi.rows();
    if (d != widths.cols()) {//Expand if necessary
        //printf("RBFKernel:  Resizing width to %d\n",(int)d);
        double wtmp = widths(0);
        widths.resize(d);
        for (int i = 0; i < d; i++) {
            widths(i) = wtmp;
        }
    }
    //I think this bumps up against numerical stability issues.
    return A*exp(-0.5f/d * (SP(a-b,widths.t())).squaredNorm());
}*/
