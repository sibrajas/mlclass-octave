function J=cst(X,y,theta)

% X- design matrix
% y- actual values
%theta - coefficients

m=size(X,1);
prediction = X*theta;
sqrderror  = (prediction-y).^2;
J=0.5/m*sum(sqrderror);
