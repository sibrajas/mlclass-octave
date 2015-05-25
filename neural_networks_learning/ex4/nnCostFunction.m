function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1=[ones(m,1) X];
z2=a1*Theta1';
a2=[ones(size(z2,1),1) sigmoid(z2)];  %mx25
z3=a2*Theta2';
a3=sigmoid(z3);  %mxnum_labels 

h=a3;

%[x,p]=max(h,[],2);  Cost function is to calculate the error so no point predicting from h
I=eye(num_labels);
Y=ones(m,num_labels);
for i=1:m,
Y(i,:)=I(y(i),:);
end
temp1=Theta1(:,2:end).^2;
temp2=Theta2(:,2:end).^2;
penalise=lambda/(2*m)*sum([temp1(:); temp2(:)]);
J=-(1/m)*sum(sum(Y.*log(h)+(1-Y).*log(1-h),2))+penalise;   % this multiply should be .* since a*b is matmultiply and a.*b is multiply each corresponding element


% -------------------------------------------------------------
%Part 2 - BackPROP

%for t=1:m,
%a1=[1 X(t,:)];
%z2=a1*Theta1';
%a2=sigmoid(z2);  %mx25
%z3=[1 a2]*Theta2';
%a3=sigmoid(z3);  %mxnum_labels 

%yk=zeros(1,num_labels);   %or size(a3,2)

delta_3=(a3-Y);  % 1 x 10
%size((delta_3*Theta2)(2:end))
%size(sigmoidGradient(z2))
%pause;
%D2=
%delta_2=;
delta_2=((delta_3*Theta2).*sigmoidGradient([ones(size(z2,1),1) z2]))(:,2:end);   %1 x 25
%size(delta_2(2:end));
%size(sigmoidGradient(z2))
%pause;
D1=delta_2'*a1;
D2=(delta_3)'*a2;

temp3=Theta1(:,2:end);
temp3=[zeros(size(temp3,1),1) temp3];
temp4=Theta2(:,2:end);
temp4=[zeros(size(temp4,1),1) temp4];

Theta1_grad=D1./m+lambda/m*temp3;
Theta2_grad=D2./m+lambda/m*temp4; 
%size(D2)   zeros(size(D2,1),1) 
%size(Theta2_grad)
%size(D1)
%size(Theta1_grad)
%pause;


%end
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
