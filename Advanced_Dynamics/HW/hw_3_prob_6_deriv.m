function xdot = hw_3_prob_6_deriv(t,x)
%HW_1_PROB_6_DERIV This is the function for the derivatives of prob. 6
global R g
x1 = x(1);
x2 = x(2);
x1dot = x2;
x2dot = (R*x2^2-g*sin(x1))/(2*R-R*x1);
xdot = [x1dot; x2dot];
end

