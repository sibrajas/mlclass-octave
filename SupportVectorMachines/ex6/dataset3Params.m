function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C1 = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma1 = [0.01 0.03 0.1 0.3 1 3 10 30];
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
cost=zeros(size(C1,2),size(sigma1,2));
mincost=Inf;
for i=1:size(C1,2),
  for j=1:size(sigma1,2),
     model= svmTrain(X, y, C1(i), @(x1, x2) gaussianKernel(x1, x2, sigma1(j)));
     predictions=svmPredict(model,Xval);
     cost(i,j)=mean(double(predictions~=yval));
     if(cost(i,j)<mincost)
        mincost=cost(i,j);
        C=C1(i);
        sigma=sigma1(j);
     endif
     %fprintf('Cost with C=%f and sigma=%f is %f\n',C1(i),sigma1(j),cost(i,j));
   end
end
fprintf('MinCost with C=%f and sigma=%f is %f\n',C,sigma,mincost);



% =========================================================================

end
