function xdot = hw_3_prob_4_deriv(t, x)
%HW_1_PROB_4_DERIV This is the function for the derivatives of prob. 4
global w1 mu g
x1 = x(1);
x2 = x(2);

x1dot = x2;
x2dot = w1^2*x1-mu*sqrt(g^2 + 4*w1^2*x2^2);

xdot = [x1dot; x2dot];

end

