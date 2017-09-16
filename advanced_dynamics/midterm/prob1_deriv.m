function xdot = prob1_deriv(t,x)
%prob1_deriv: Derivatives of x1 and x2 for ode45 where x1 = r and x2 = rdot
global mu g k m r0 %parameters used in the ode defined in problem1.m
x1 = x(1);
x2 = x(2);
x1dot = x2;
x2dot = 60.84*(cos(6.5*t))^2*x1 - mu*sign(x2)*sqrt(g^2 + (-50.7*sin(6.5*t)*x1 + 15.6*cos(6.5*t)*x2)^2) - (k/m)*(x1-r0);
xdot = [x1dot; x2dot];
end