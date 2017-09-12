close all
clear all
%% Problem 3
x = [2, 2.5, 3, 5, 9];
y = [-4.2, -5, 2, 1, 24.3];

% a) Make a plot of the data
scatter(x,y,'filled','b');
xlabel('x')
ylabel('y')
title('Problem 3')
hold on

% b) Determine and plot least squares line
% see equations 3.38, 3.37, and 3.36 in the book
A = [];
for i = 1:length(y),
   A(i,1) = x(i); 
   A(i,2) = 1;
end

c = (A'*A)^-1*A'*y'
y_lin = c(1)*x + c(2);
plot(x,y_lin,'g')
legend('Raw Data','Linear Regression')

% c) Assume 1st and last points are most accurate, formulate a weighting
% matrix and compute a weighted least-squares line that fits the data. Plot
% this line
v = [10,1,1,1,10]; %vector with weighting terms
W = diag(v); %create diagonal weighting matrix
c_w = (A'*W*A)^-1*A'*W*y'
y_lin_w = c_w(1)*x + c_w(2);
plot(x,y_lin_w,'r')
legend('Raw Data','Least Squares Line', 'Weighted Least Squares Line')

%% Problem 12
clear all
close all
f = [1, 1, 2, 3, 5, 8, 13];
m = 2;
N = length(f);
% a) Write down the data matrix A and the Grammian A'A using
%       i) the covariance method (assume m = 2)
A_c = [f(m), f(m-1);...
    f(m+1), f(m);...
    f(m+2), f(m+1);...
    f(m+3), f(m+2);...
    f(m+4), f(m+3);...
    f(m+5), f(m+4)]
R_c = A_c'*A_c
%       ii) the autocorrelation method (assume m = 2)
A_a = [f(1), 0;...
    f(m), f(m-1);...
    f(m+1), f(m);...
    f(m+2), f(m+1);...
    f(m+3), f(m+2);...
    f(m+4), f(m+3);...
    f(m+5), f(m+4);...
    0, f(m+5)]
R_a = A_a'*A_a

% b) Determine the least-squares coefficients for the predictor using the
% covariance method
d_c = [1, 2, 3, 5, 8, 13];
h_c = R_c^-1*A_c'*d_c'

% autocorrelation method
d_a = [1, 1, 2, 3, 5, 8, 13, 0];
h_a = R_a^-1*A_a'*d_a'

% c) Find Error
e_c = d_c' - A_c*h_c
e_a = d_a' - A_a*h_a

