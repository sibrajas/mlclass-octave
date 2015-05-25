function [learnCurve_train,learnCurve_val]=learnCurve_rand_samples(X_poly,y,X_poly_val,yval,lambda)
%lambda=0.01;
m=12;
mv=4;
iter=50;
[error_train_final]=zeros(m,1);
[error_val_final]=zeros(m,1);
for i=1:iter,
[s r]=sort(rand(size(X_poly,1),1));
[s rval]=sort(rand(size(X_poly_val,1),1));

%r=rand(1,size(X_poly,1));
%rval=rand(1,size(X_poly_val,1));
[error_train,error_val]=learningCurve(X_poly(r(1:m),:),y(r(1:m)),X_poly_val(rval(1:mv),:),yval(rval(1:mv)),lambda);

error_train_final=error_train_final.+error_train;
error_val_final=error_val_final+error_val;
end
error_train_final=error_train_final/iter;
error_val_final=error_val_final/iter;
plot(1:m, error_train_final, 1:m, error_val_final);
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')
