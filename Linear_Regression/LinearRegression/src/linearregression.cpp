#include "linearregression.h"

linearregression::linearregression()
{
    //ctor
}

/**
*@file linearregression.cpp
*here functions for multiple Linear Regression is defined
*/

linearregression::linearregression(const arma::mat& X,const arma::vec& Y)

{
  this->X=X;
  this->Y=Y;
}

linearregression::linearregression(const linearregression& linearregression)
{
    this->X=linearregression.X;
    this->Y=linearregression.Y;
}


 arma::vec linearregression::multiply(const arma::vec& H, const arma::mat& X, arma::uword theta)
{
    arma::vec temp(H.n_rows);

    for(arma::uword i=0;i < H.n_rows; i++ )
    {
        temp(i)=H(i) * X(i,theta); //col vector n*1
    }
    return temp;
}

   arma::vec linearregression::subtractT(const arma::mat& predict,const arma::vec& Y)
   {
    arma::vec temp(Y.n_rows);
    arma::vec X(predict);
    for(arma::uword i=0; i < Y.n_rows;i++)
        temp(i)=X(i)- Y(i);
    return temp;
   }


   arma::vec gradient_descent(const arma::mat& X, const arma::vec& Y, const double alpha, int iters)
     {
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

}

void linearregression::TrainModel(double alpha,int iterations)
{
    arma::vec J(iterations);

    this->theta = gradient_descent(this->X,this->Y,alpha,iterations);
     //cost function values of linear regression can be calculated
    //cout theta value
}

double linearregression::predict(arma::vec &X)
{
    arma::rowvec thetat = arma::trans(this->theta);
    double res = arma::as_scalar(thetat * X);
    return res;
}
