%% Problem 1 simulation to check if I did it right
m = 10;
l = 1;
g = 9.81;
x0 = [30*pi/180, 30*pi/180, 0 , 2*pi/30];
tspan = 0:0.01:30;
[t,x] = ode45(@hw_8_pr_1_deriv,tspan,x0);
F = m*l.*x(:,3).^2 + m*l.*x(:,4).^2.*sin(x(:,1)).^2 + m*g.*cos(x(:,1));
figure()
plot(t,x(:,1).*180/pi)
figure()
plot(t,x(:,2).*180/pi)
figure()
plot(t,F)