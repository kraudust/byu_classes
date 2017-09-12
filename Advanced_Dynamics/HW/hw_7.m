close all
clear all
clc

%% Problem 4
th0 = 18*pi/180;
thd0 = 0;
x0 = [th0; thd0];
tspan = 0:0.001:3;
[t,out] = ode45(@hw_7_pr_4_deriv, tspan, x0);

%convert to degrees
out(:,1) = out(:,1)*180/pi;
out(:,2) = out(:,2)*180/pi;

figure()
plot(t,out(:,1))
xlabel('Time (s)')
ylabel('Theta (deg)')

