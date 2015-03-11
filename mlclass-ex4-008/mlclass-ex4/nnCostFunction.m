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


% Add ones to the X data matrix
X = [ones(m, 1) X];
hiddenLayer = sigmoid(X*transpose(Theta1));
hiddenLayer = [ones(m, 1) hiddenLayer];
outputLayer = sigmoid(hiddenLayer*transpose(Theta2));


newY = zeros(size(y, 1) , num_labels);
for n = 1 : size(y, 1)
  labelNumber = y(n);
  label = zeros(1, num_labels);
  label(1, labelNumber) = 1;
  newY(n,:) = label;
end
  
leftPart = - newY.*log(outputLayer);
rightPart = -(1-newY).*log(1-outputLayer);
sumPart = leftPart + rightPart;
J = sum(sum(sumPart))/m;


%After Regulartization
theta1Part = Theta1(:, 2: size(Theta1,2));
theta2Part = Theta2(:, 2: size(Theta2,2));

theta1Sum =  sum(sum(theta1Part.*theta1Part));
theta2Sum =  sum(sum(theta2Part.*theta2Part));

J  =  J + lambda/(2*m)*(theta1Sum+theta2Sum);


captitalDelta1 = zeros(size(Theta1));%25*401
captitalDelta2 = zeros(size(Theta2));%10*26

%Feedforward and Backpropagation
for t = 1 : m
  a1 = X(t,:); %1*401
  a2 = sigmoid(a1*transpose(Theta1));%1*25
  a2 = [ones(1, 1) a2];%1*26
  a3 = sigmoid(a2*transpose(Theta2));%1*10
  
  tLabel = y(t,:);
  tLabelVec = zeros(1, num_labels);
  tLabelVec(1, tLabel) = 1;
  
  delta3 = a3 - tLabelVec; %1*10
  delta3 = transpose(delta3);%10*1
  
  gz2 = sigmoidGradient(a1*transpose(Theta1));%1*25
  
  gz2 = [ones(1, 1) gz2];%1*26
  delta2 = transpose(Theta2)*delta3.*transpose(gz2); %26*1
  delta2 = delta2(2:end);%25*1
  
  
  
  captitalDelta1 = captitalDelta1 + delta2*a1;
  captitalDelta2 = captitalDelta2 + delta3*a2;

  
  
  
  
  
end
  
Theta1_grad = captitalDelta1/m;%25*401
Theta2_grad = captitalDelta2/m;%10*26

firstColumnTheta1 = Theta1_grad(:,1);
firstColumnTheta2 = Theta2_grad(:,1);

Theta1_grad = Theta1_grad + lambda*Theta1/m;
Theta2_grad = Theta2_grad + lambda*Theta2/m;

Theta1_grad(:,1) =firstColumnTheta1;

Theta2_grad(:,1) =firstColumnTheta2;








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
