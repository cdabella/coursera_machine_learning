function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
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

test_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
error = 10000;

for C_index = 1: length(test_vec)
    for sigma_index = 1: length(test_vec)
        C_test = test_vec(C_index);
        sigma_test = test_vec(sigma_index);
        model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test)); 
        predictions = svmPredict(model, Xval);
        test_error = mean(double(predictions ~= yval));
        
        if (test_error < error)
            error = test_error;
            C = C_test;
            sigma = sigma_test;
        endif
    end
end

C
sigma
% =========================================================================

end
