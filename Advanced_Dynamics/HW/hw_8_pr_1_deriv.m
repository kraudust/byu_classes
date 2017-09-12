function xdot = hw_8_pr_1_deriv(t,x)
g = 9.81;
l = 1;
x1 = x(1); %th
x2 = x(2); %phi
x3 = x(3); %thdot
x4 = x(4); %phidot
xdot = [...
    x3;...
    x4;...
    x4^2*sin(x1)*cos(x1)-(g/l)*sin(x1);...
    (-2*x4*x3*sin(x1)*cos(x1))*1/(sin(x1)^2)];

end

