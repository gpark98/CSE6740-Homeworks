function [ U, V ] = myRecommender( rateMatrix, lowRank )
    % Please type your name here:
    name = 'Song, Le';
    disp(name); % Do not delete this line.

    % Parameters
    maxIter = 0; % Choose your own.
    learningRate = 0; % Choose your own.
    regularizer = 0; % Choose your own.
    
    % Random initialization:
    [n1, n2] = size(rateMatrix);
    U = rand(n1, lowRank) / lowRank;
    V = rand(n2, lowRank) / lowRank;

    % Gradient Descent:
    
    % IMPLEMENT YOUR CODE HERE.
end