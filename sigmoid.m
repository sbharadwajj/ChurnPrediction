function g = sigmoid(z)

g = zeros(size(z));

exponential = exp(-z);
value = 1 .+ exponential; 
g = 1./value;



% =============================================================

end
