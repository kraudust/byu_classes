function xdot = hw_7_pr_4_deriv(t,x)
m = 2.1;
L = 1;
g = 9.81;
R = 0.5;
x1 = x(1); %theta
x2 = x(2); %thetadot
xdot = [x2; (-g*R*x1*cos(x1) - R^2*x2^2*x1)/(R^2*x1^2 + (1/12)*L^2)];
end

