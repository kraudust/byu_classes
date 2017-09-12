function [ xdot ] = odefunc2(my_t,x,qdd,tspan,k)
    disp(my_t)
    qd = x(8:14);
    qdd = interp1(tspan,qdd,my_t);
    xdot = [qd;qdd'];


end

