%% Problem 1
clear all
close all
clc
% Part (c)
global mu g k m r0
mu = 0.12;   
g = 9.81; %m/s^2
m = 2; %kg
k = 1000; %N/m
r0 = 0.1; %m
x0 = [1.5*r0; 0]; %initial conditions r(0) = 1.5*r0 rdot(0) = 0

dt = 0.00001; %time between "measurements"
t = 0:dt:1; %time vector

[t,x] = ode45(@prob1_deriv,t,x0); %use ode45 to find solution
x1 = x(:,1); %position r(t) in m
x2 = x(:,2); %velocity rdot(t) in m/s 

figure()
subplot(2,1,1)
plot(t,x1)
title('Position')
xlabel('Time (sec)')
ylabel('r (m)')

subplot(2,1,2)
plot(t,x2)
title('Velocity')
xlabel('Time (sec)')
ylabel('rdot (m/s)')

% Part (d)
%Find maximum and minimum values of r and rdot over t = 0-1
rmax = max(x1)
rmin = min(x1)
rdotmax = max(x2)
rdotmin = min(x2)

%% Problem 3
clc
clear all
close all

L = 0.03; %m
R = .01; %m
w1 = 12; %rad/s
w2 = -7; %rad/s
m = 0.05; %kg
th = 30; %deg

F = sqrt(m^2*(2*R*w1*w2 - R*w1^2 - R*w2^2 - L*w1^2*cosd(th))^2 + ...
    m^2*L^2*w1^4*(sind(th))^2)
