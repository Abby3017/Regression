#include <math.h>
#include <iostream>

#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#ifndef ARMA_INCLUDES
#include <armadillo>
#endif // ARMA_INCLUDES

class linearregression
{
    public:
    /** Default constructor */
    linearregression();
        /**
    *  **Creates Model**
    * @param X = matrix consisting sample set for the model
    * @param Y = vector consisting result of each feature list in X
    * have two constructor one copy constructor and a parameterized constructor
    */
    linearregression(const arma::mat& X,
                     const arma::vec& Y);

    linearregression(const linearregression& linearregression);

    /**
    *  **Train model**
    * @param alpha = learning rate of model
    * @param iters = number of iteration specified by user to run algo
    */
    void TrainModel(double alpha, int iters);

    double predict(arma::vec& X);

    static arma::vec subtractT(const arma::mat& predict,const arma::vec& Y);

    static arma::vec multiply(const arma::vec& H, const arma::mat& X, arma::uword theta);

    arma::vec gradient_descent(const arma::mat& X, const arma::vec& Y, const double alpha, int iters);
    /* {
        const int m = X.n_rows;
    arma::mat Xtemp =X;
    arma::colvec Ins(X.n_rows);
    Ins.fill(1.0);
    Xtemp.insert_cols(0,Ins);

    arma::vec thetavec(Xtemp.n_cols);
    arma::vec thetatemp(arma::size(thetavec));
    thetatemp.fill(1.0);
for(int i=0;i<iters;i++)
  {
     thetavec = thetatemp;
     arma::mat Xth;
     arma::colvec thetaT=arma::trans(thetavec);
     Xth= Xtemp * thetaT;
     arma::vec H;
     H = linearregression::subtractT(Xth,Y);
   for(arma::uword thetaNum = 1;thetaNum < thetavec.n_rows ;thetaNum++)
   {
      arma::vec resTemp = linearregression::multiply(H,Xtemp,thetaNum);  //thetanum starts from 1 as 0 is stored as 1.0
      double restemp = arma::as_scalar(arma::accu(resTemp));
      double res = ((1.0)/(2*m)) * restemp;
      thetatemp(thetaNum) = thetatemp(thetaNum) - res;
   }
  }

    return thetavec;

}*/
    private:

    arma::mat X;

    arma::vec Y;

    arma::vec theta;
};

#endif // LINEARREGRESSION_H
