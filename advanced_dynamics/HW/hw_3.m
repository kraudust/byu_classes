%% Problem 4
clear all
close all
clc
% Parameters
global w1 mu g
mu = 0.5;
x10 = .05; %initial position is 5 cm
x20 = 0; %starts from rest
x0 = [x10; x20];
w1 = 10; %omega one is 10 rad/s
g = 9.81; %m/s^2

dt = 0.001;
t = 0:dt:1;

[t,x] = ode45(@hw_3_prob_4_deriv,t,x0); %use ode45 to find solution
x1 = x(:,1); %position x(t) in m
x2 = x(:,2); %velocity v(t) in m/s 

%find where the bead reaches 30 cm (0.3 m) and make plots
index = x1 < 0.301;
t_shorter = t(index);
plot(t_shorter,x1(index), 'b');
hold on
plot(t_shorter,ones(length(t(index)))*0.3, 'r')
xlim([0 t_shorter(end)])
xlabel('Time (s)')
ylabel('Bead Location (m)')
title('Bead Location as a function of time')
legend('Bead Location','Goal of 30 cm')

t_shorter(end)

%% Problem 5
clear all
close all
clc

%Parameters
global m k R
m = 0.75; %kg
k = 25; %N/m
R = 1; %m
g = 9.81; %m/s^2
x10 = 0; %initial position
x20 = 0; %initial velocity
x0 = [x10; x20];
dt = 0.001;
dth = 0.001;
tf = acos((2*pi-5)/-5); %time at which theta is 2pi
t = 0:dt:tf; %generate time vector to solve at

%solve x1 and x2 as fun. of time
[t,x] = ode45(@hw_3_prob_5_deriv,t,x0); 
x1 = x(:,1); %position s in m
x2 = x(:,2); %velocity in m/s

%Find theta as a function of time
th = -5*cos(t) + 5;

%Find normal force
w = 5*sin(t);
wdot = 5*cos(t);
Ny = (2*x2.*w + x1.*wdot-R*w.^2)*m;

%make plots
figure(1) %s as a function of time for 1 revolution
plot(t,x1)
xlabel('Time (s)')
ylabel('s (m)')
title('s as a function of time')

figure(2) %s as a function of theta for 1 revolution
plot(th,x1)
xlabel('Theta (rad)')
ylabel('s (m)')
title('s as a function of theta')

figure(3)
plot(t,Ny)
xlabel('Time (sec)')
ylabel('N (Newtons)')
title('Normal force as a function of time')

figure(4)
plot(th,Ny)
xlabel('Theta (rad)')
ylabel('N (Newtons)')
title('Normal force as a function of theta')

%% Problem 6
clear all
close all
clc

%set up parameters
global g R
g = 9.81;
R = 7;
%R = 17;
x10 = 0; %initial angle
x20 = sqrt(g/(2*R)); %initial angular velocity
x0 = [x10;x20];
dt = 0.001;
tf = 15; %time to simulate to
t = 0:dt:tf;

%solve x1 and x2 as fun. of time
[t,x] = ode45(@hw_3_prob_6_deriv,t,x0); 
x1 = x(:,1); %position s in m
x2 = x(:,2); %velocity in m/s
plot(t,x1)
xlabel('Time (sec)')
ylabel('Theta (rad)')
title('Theta as a function of time')
