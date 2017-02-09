#include "linearregression.h"

using namespace std;

int main()
{
    /**
    *file should be saved in project directory
    *in arma::mat format, this contain values for Linear Regression model (format n*n)
    */
    arma::mat X;
    X.load("armaX.mat");
    /**
    *file should be saved in project directory in arma::mat format
    *this contains, result of these values (format n*1)
    */
    arma::mat Y;
    Y.load("armaY.mat");
    arma::vec Yv=Y;

    linearregression lr(X,Yv);
    /**
    * enter value of alpha used for this Regression
    */
    double alpha;
    std::cin>>alpha;
    /**
    *enter number of iterations
    */
    int iters;
    std::cin>>iters;

    lr.TrainModel(alpha,iters);


    return 0;
}
