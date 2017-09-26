close all
clear all
clc

%Initial conditions in meters and radians
x0 = -5; 
y0 = -3; 
th0 = pi/2; 

%Noise characteristics on velocities experienced by the robot
alpha1 = 0.1;
alpha2 = 0.01;
alpha3 = 0.01;
alpha4 = 0.1;
alpha = [alpha1; alpha2; alpha3; alpha4];

%Landmark locations
xland = [6, -7, 6];
yland = [4, 8, -4];
m = [xland', yland'];
%Standard deviations of range and bearing sensor noise in meters & radians
sigma_range = 0.1;
sigma_phi = 0.05;

%Generate time vector (seconds)
Ts = 0.1; 
t = 0:Ts:20;

%Generate control inputs (linear and angular velocities)
v_c = 1 + 0.5*cos(2*pi*0.2*t);
omega_c = -0.2 + 2*cos(2*pi*0.6*t);
u = [v_c; omega_c];

%Generate true path using velocity motion model
xt = zeros(3,length(t));
xt(:,1) = [x0;y0;th0];

for i = 1:length(t)-1
    xt(:,i+1) = velocity_motion_model(u(:,i), xt(:,i), alpha, Ts);
end

%Simulate range and bearing measurements
[r, phi] = sim_measurements(m,sigma_range,sigma_phi,xt);
z = [r;phi];

%Run EKF to get state estimates
mu = zeros(3,length(t));
sigma = zeros(3,3,length(t));
sigma(:,:,1) = eye(3);
mu(:,1) = [x0;y0;th0];
for i = 1:length(t)-1
    [mu(:,i+1), sigma(:,:,i+1)] = extended_kalman_filter(mu(:,i), sigma(:,:,i), u(:,i), z(:,i+1), m, sigma_range, sigma_phi,Ts,alpha);
end

%Draw Robot
draw_robot([t;xt],m,r,phi);
