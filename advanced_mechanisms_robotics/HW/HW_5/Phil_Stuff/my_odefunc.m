function [ xdot ] = myode_func(my_t,x,right,tspan,qdes)
    
    Kp = [20 0 0 0 0 0 0;
          0 50 0 0 0 0 0;
          0 0 20 0 0 0 0;
          0 0 0 30 0 0 0;
          0 0 0 0 20 0 0;
          0 0 0 0 0 5 0;
          0 0 0 0 0 0 5];
      
    Kd = [10 0 0 0 0 0 0;
          0 20 0 0 0 0 0;
          0 0 2 0 0 0 0;
          0 0 0 10 0 0 0;
          0 0 0 0 2 0 0;
          0 0 0 0 0 2 0;
          0 0 0 0 0 0 2];
    
    disp(my_t)
    
    
    q = x(1:7);
    qd = x(8:14);
    
    %tau = interp1(tspan,tau,my_t);
    qdes = interp1(tspan,qdes,my_t);
    
    %PD controller
    tau = Kp*(qdes'-q) - Kd*qd
    
    qdd = right.accel(q',qd',tau');
    
    xdot = [qd;qdd];
end

