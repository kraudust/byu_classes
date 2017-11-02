close all
clear all
clc

global landmarks_seen sigma_r sigma_phi alpha Ts
%Initial conditions in meters and radians
% x0 = -5; 
% y0 = -3;
% th0 = pi/2; 
x0 = 0;
y0 = 0;
th0 = 0;

%Noise characteristics on velocities experienced by the robot
alpha1 = 0.1;
alpha2 = 0.01;
alpha3 = 0.01;
alpha4 = 0.1;
alpha = [alpha1; alpha2; alpha3; alpha4];

%Landmark locations
% xland = [6, -7, 6];
% yland = [4, 8, -4];
% lm = [xland', yland'];
num_landmarks = 100;
lm = 10-20*rand(num_landmarks,2);
landmarks_seen = zeros(num_landmarks,1);


%Standard deviations of range and bearing sensor noise in meters & radians
sigma_r = 0.1;
sigma_phi = 0.05;

%Generate time vector (seconds)
Ts = 0.1; 
t = 0:Ts:30;

%Generate control inputs (linear and angular velocities)
v_c = 1 + 0.25*cos(2*pi*0.2*t);
omega_c = -0.3 + 1.5*cos(2*pi*0.6*t);
u = [v_c; omega_c];

%Generate true path using velocity motion model
xt = zeros(3,length(t));
xt(:,1) = [x0;y0;th0];

for i = 1:length(t)-1
    xt(:,i+1) = velocity_motion_model(u(:,i), xt(:,i), alpha, Ts);
end

%Simulate range and bearing measurements
[r, phi] = sim_measurements(lm,sigma_r,sigma_phi,xt);
z = [r;phi];

%Run EKF to get state estimates
mu = zeros(3+2*num_landmarks,length(t)); %eq. 10.7 in probabilistic robotics
sigma = zeros(3+2*num_landmarks,3+2*num_landmarks,length(t));
%sigma(:,:,1) = eye(3);
sigma(:,:,1) = 10^10*eye(3+2*num_landmarks);
sigma(1,1,1) = 0;
sigma(2,2,1) = 0;
sigma(3,3,1) = 0;
mu(:,1) = 0;
for i = 1:length(t)-1
    [mu(:,i+1), sigma(:,:,i+1)] = ekf_slam(mu(:,i), sigma(:,:,i), u(:,i), z(:,i+1), lm);
end

%Draw Robot
draw_robot(t,xt,lm,r,phi,mu, sigma);

%Make Plots----------------------------------------------------------------
%States and State Estimates
figure()
subplot(3,1,1)
plot(t,xt(1,:))
hold on
plot(t,mu(1,:))
ylabel('x Position (m)')
legend('True State','EKF Estimate')

subplot(3,1,2)
plot(t,xt(2,:))
hold on
plot(t,mu(2,:))
ylabel('y Position (m)')
legend('True State','EKF Estimate')

subplot(3,1,3)
plot(t,xt(3,:))
hold on
plot(t,mu(3,:))
xlabel('Time (sec)')
ylabel('Heading (Rad)')
legend('True State','EKF Estimate')
suptitle('States and State Estimates')

%---------------------------------------
%Error
figure()
x_error = xt(1,:)-mu(1,:);
y_error = xt(2,:)-mu(2,:);
th_error = xt(3,:)-mu(3,:);

subplot(3,1,1)
plot(t,x_error)
hold on
x_upper = 2*reshape(sqrt(sigma(1,1,:)),1,length(t));
x_lower = -2*reshape(sqrt(sigma(1,1,:)),1,length(t));
plot(t,x_upper,'r')
plot(t,x_lower,'r')
ylabel('x Position Error (m)')

subplot(3,1,2)
plot(t,y_error)
hold on
y_upper = 2*reshape(sqrt(sigma(2,2,:)),1,length(t));
y_lower = -2*reshape(sqrt(sigma(2,2,:)),1,length(t));
plot(t,y_upper,'r')
plot(t,y_lower,'r')
ylabel('y Position Error (m)')

subplot(3,1,3)
plot(t,th_error)
hold on
th_upper = 2*reshape(sqrt(sigma(3,3,:)),1,length(t));
th_lower = -2*reshape(sqrt(sigma(3,3,:)),1,length(t));
plot(t,th_upper,'r')
plot(t,th_lower,'r')
xlabel('Time (sec)')
ylabel('Theta Angle Error (Rad)')
suptitle('Estimation Error')

figure()
surf(sigma(:,:,301))
view(0,90)
