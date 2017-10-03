close all
clear all
clc

%Initial conditions in meters and radians
x0 = -5; 
y0 = -3;
% x0= 0;
% y0 = 0;
th0 = pi/2; 

%Noise characteristics on velocities experienced by the robot
alpha1 = 0.1;
alpha2 = 0.01;
alpha3 = 0.01;
alpha4 = 0.1;
% n = 10;
% alpha1 = n*0.1;
% alpha2 = n*0.01;
% alpha3 = n*0.01;
% alpha4 = n*0.1;
alpha = [alpha1; alpha2; alpha3; alpha4];

%Landmark locations
% xland = [6, -7, 6];
% yland = [4, 8, -4];
% m = [xland', yland'];
m = 10-20*rand(10000,2);


%Standard deviations of range and bearing sensor noise in meters & radians
sigma_range = 0.1;
sigma_phi = 0.05;
% sigma_range = 10*0.1;
% sigma_phi = 10*0.05;

%Generate time vector (seconds)
Ts = 0.1; 
t = 0:Ts:20;

%Generate control inputs (linear and angular velocities)
v_c = 1 + 0.5*cos(2*pi*0.2*t);
omega_c = -0.2 + 2*cos(2*pi*0.6*t);
% v_c = 1 + 20*0.5*cos(2*pi*0.2*t);
% omega_c = -0.2 + 2*cos(2*pi*0.6*t);
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
K1norm = zeros(1,length(t));
K2norm = zeros(1,length(t));
K3norm = zeros(1,length(t));
sigma(:,:,1) = eye(3);
mu(:,1) = [x0;y0;th0];
for i = 1:length(t)-1
    [mu(:,i+1), sigma(:,:,i+1), K_t] = ...
        extended_kalman_filter(mu(:,i), sigma(:,:,i), u(:,i), z(:,i+1),...
        m, sigma_range, sigma_phi,Ts,alpha);
    K1norm(i) = norm(K_t(:,:,1),'fro');
    K2norm(i) = norm(K_t(:,:,2),'fro');
    K3norm(i) = norm(K_t(:,:,3),'fro');
    
end

%Draw Robot
draw_robot(t,xt,m,r,phi,mu);

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

% %-------------------------------------
% %Kalman Gains
% figure()
% subplot(3,1,1)
% plot(t, K1norm)
% ylabel('Landmark 1')
% subplot(3,1,2)
% plot(t,K2norm)
% ylabel('Landmark 2')
% subplot(3,1,3)
% plot(t,K3norm)
% ylabel('Landmark 3')
% xlabel('Time (sec)')
% suptitle('Frobenius Norms of Kalman Gains')
