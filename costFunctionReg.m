function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y); % number of training examples
n = size(X,2);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
M = 1/m;
M1 = 2*m;
L = eye(n);
L(1,1) = 0;

h = sigmoid(X*theta);
y_one = -y'*log(h);
y_zeros = (1.-y)'*log(1.-h);

value = lambda/M1;
theta_L = L*theta;
theta_square = theta_L.^2;
sumof = sum(theta_square);
regularizing_term = value.*sumof;
J_1 = M.*(y_one - y_zeros);
J = J_1 + regularizing_term;

value1 = lambda/m;
regularizing_term_grad = value1.*theta_L;
hypo = h.-y;
hypo1 = X'*hypo;
grad = M.*hypo1 + regularizing_term_grad;

% =============================================================

end
