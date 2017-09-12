function xdot = hw_3_prob_5_deriv(t,x)
%HW_1_PROB_5_DERIV This is the function for the derivatives of prob. 5
global R k m
x1 = x(1);
x2 = x(2);
x1dot = x2;
x2dot = 5*R*cos(t) + 25*x1*(sin(t))^2 - k*x1/m;
xdot = [x1dot; x2dot];
end

