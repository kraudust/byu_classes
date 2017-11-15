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
num_landmarks = 50;
lm = 10-20*rand(num_landmarks,2);


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
% xt(3,:) = wrapToPi(xt(3,:));
%Simulate range and bearing measurements
[r, phi] = sim_measurements(lm,sigma_r,sigma_phi,xt);
for i = 1:length(t)
    z(:, 1, i) = r(:,i);
    z(:, 2, i) = phi(:,i);
end

% Run fast SLAM ----------------------------------------------------------------------------------------

%Generate Particles
M = 100; %number of particles
rand_x = -10 + 20.*rand(1,M);
rand_y = -10 + 20.*rand(1,M);
rand_th = -pi + 2*pi.*rand(1,M);
chi0 = [0*rand_x', 0*rand_y', 0*rand_th'];
landmarks_seen = zeros(M,num_landmarks);
% Construct state structure Y (see eq. 13.11)
Y(1).x = chi0;
for i = 1:M
    Y(1).mu(:,:,i) = repmat([0, 0], [num_landmarks, 1]);
    Y(1).sigma(:,:,i) = repmat(10^10*eye(2), [num_landmarks, 1]);
end

w(:,1) = 1/M * ones(M,1);

% Run fast slam algorithm
ct = 1;
fov = pi/2;
for i = 1:length(t)-1
    [Y(i+1), w(:,i+1)] = fastSLAM(z(:,:,i), ct, u(:,i), Y(i));
    ct = ct + 1;
    if ct > num_landmarks
        ct = 1;
    end
    i
end

% Get index of heaviest weighted particle at each time step for visualization
% Get robot and landmark locations from heaviest weighted particle
for i = 1:length(t)
    [~, idx] = max(w(:,i));
%     x_est(:, i) = (Y(i).x(idx, :))'; %robot pose at each t: [x;y;th]
    x_est(:,i) = mean(Y(i).x,1);
%     lm_est(:, :, i) = Y(i).mu(:, :, idx); %landmark locations at each t: [x y; x y; ... ; x y]
    lm_est(:,:,i) = mean(Y(i).mu, 3);
%     lm_sigma(:,:, i) = Y(i).sigma(:, :, idx); %landmark covariance at each t: [2x2 sigma for lm1; 2x2 sigma for lm2 ...]
    lm_sigma(:,:,i) = mean(Y(i).sigma,3);
end

% Get robot and landmark locations from heaviest weighted particle

% Restructure particles for plotting (to match the draw_robot function I made in hw 4
for i = 1:length(t)
    particles(i, :, :) = (Y(i).x)';
end

%Draw Robot
draw_robot(t, xt, lm, r, phi, x_est, lm_est, lm_sigma, particles);

%Make Plots---------------------------------------------------------------------------------------------
%States and State Estimates
figure()
subplot(3,1,1)
plot(t,xt(1,:))
hold on
plot(t,x_est(1,:))
ylabel('x Position (m)')
legend('True State','EKF Estimate')

subplot(3,1,2)
plot(t,xt(2,:))
hold on
plot(t,x_est(2,:))
ylabel('y Position (m)')
legend('True State','EKF Estimate')

subplot(3,1,3)
plot(t,xt(3,:))
hold on
plot(t,x_est(3,:))
xlabel('Time (sec)')
ylabel('Heading (Rad)')
legend('True State','EKF Estimate')
suptitle('States and State Estimates')

% %---------------------------------------
% %Error
% figure()
% x_error = xt(1,:)-mu(1,:);
% y_error = xt(2,:)-mu(2,:);
% th_error = xt(3,:)-mu(3,:);
% 
% subplot(3,1,1)
% plot(t,x_error)
% hold on
% x_upper = 2*reshape(sqrt(sigma(1,1,:)),1,length(t));
% x_lower = -2*reshape(sqrt(sigma(1,1,:)),1,length(t));
% plot(t,x_upper,'r')
% plot(t,x_lower,'r')
% ylabel('x Position Error (m)')
% 
% subplot(3,1,2)
% plot(t,y_error)
% hold on
% y_upper = 2*reshape(sqrt(sigma(2,2,:)),1,length(t));
% y_lower = -2*reshape(sqrt(sigma(2,2,:)),1,length(t));
% plot(t,y_upper,'r')
% plot(t,y_lower,'r')
% ylabel('y Position Error (m)')
% 
% subplot(3,1,3)
% plot(t,th_error)
% hold on
% th_upper = 2*reshape(sqrt(sigma(3,3,:)),1,length(t));
% th_lower = -2*reshape(sqrt(sigma(3,3,:)),1,length(t));
% plot(t,th_upper,'r')
% plot(t,th_lower,'r')
% xlabel('Time (sec)')
% ylabel('Theta Angle Error (Rad)')
% suptitle('Estimation Error')
% 
% figure()
% surf(sigma(:,:,301))
% view(0,90)
