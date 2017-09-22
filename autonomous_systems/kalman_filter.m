clear all
close all
clc

%Vehicle Parameters
m = 100; %kg (Mass)
b = 20; %N-s/m (Linear Drag Coeficcient)
Ts = 0.05; %sample period in seconds

% Set Up Input to System
t = (0:Ts:50)';
u = ones(1001, 1);
for i = 1:length(t)
    if t(i) >= 0 && t(i) < 5
        u(i) = 50; % Newtons
    elseif t(i) >= 25 && t(i) < 30
        u(i) = -50; % Newtons
    else
        u(i) = 0;
    end
end

%u = sin(t);

% Set up Continuous State Space System
Ac = [0 1; 0 -b/m];
Bc = [0; 1/m];
Cc = [1 0]; % I can only measure position, not velocity
Dc = 0;

sys_c = ss(Ac, Bc, Cc, Dc);

% Convert to a Discrete State Space System
sys_d = c2d(sys_c, Ts);
[A, B, C, D] = ssdata(sys_d);

% Noise Characteristics
R = [0.0001, 0; 0, 0.01]; % Process Covariance
%R = [0.001, 0; 0, 0.1];
Q = 0.001; % Measurement Covariance (we are measuring position only)
%Q = 1;
% Initial Conditions
x0 = [0; 0];
%x0 = [30;30];
xt = zeros(2,length(t)); %true states
xt(:,1) = x0;

% Simulate the System to get True States
for i = 2:length(t)
    xt(:,i) = A*xt(:,i-1) + B*u(i-1) + R.^(0.5)*randn(2,1);
end

% Run Kalman Filter
sigma_0 = eye(2); %initialize covariance
mu_0 = [10*(rand()-0.5);5*(rand()-0.5)]; %initialize state estimate
mu = zeros(2,length(t)); %vector for state estimates
mu(:,1) = mu_0;
sigma = zeros(2,2,length(t)); %vector for covariance
sigma(:,:,1) = sigma_0;
K_t = zeros(2,length(t));
sigma_bart = zeros(2,2,length(t)-1);
sigma_predict_update = zeros(2,2,2*length(t)-1);

% Pull in McLain's test data for grading
clear Q R sigma_0 mu_0 xt mu sigma
load hw1_soln_data.mat %loads Q, R, Sig0
R =  [R(2,2), 0; 0, R(1,1)];%his position and velocity states are in the opposite order as mine
sigma_0 = [Sig0(2,2),0;0,Sig0(1,1)]; %his position and velocity states are in the opposite order as mine
sigma = zeros(2,2,length(t)); %vector for covariance
sigma(:,:,1) = sigma_0;
mu = zeros(2,length(t)); %vector for state estimates
mu_0 = [mu0(2);mu0(1)]; %his position and velocity states are in the opposite order as mine
mu(:,1) = mu_0;
xt = [xtr;vtr];

for i = 2:length(t)
    % Get sensor measurement
    %z_t = C*xt(:,i) + Q.^(0.5)*randn(1); %from true data with added noise
    z_t = z(i);%from Mclain
    [mu(:,i), sigma(:,:,i), K_t(:,i-1), sigma_bart(:,:,i-1)] = kalman_filter_func(mu(:,i-1), sigma(:,:,i-1), u(i-1), z_t, A, B, C, R, Q);
end
K_t(:,length(t)) = K_t(:,length(t)-1);

%So I can visualize the covariance after both prediction and measurement update steps
i = 1;
for j = 1:length(t)
    sigma_predict_update(:,:,i) =  sigma(:,:,j);
    if j < length(t)
        sigma_predict_update(:,:,i+1) = sigma_bart(:,:,j);
    end
    i = i + 2;
end

%Make Plots
figure()
plot(t,xt(1,:))
hold on
plot(t,xt(2,:))
plot(t,mu(1,:))
plot(t,mu(2,:))
xlabel('Time (sec)')
ylabel('State Magnitude')
title('State and State Estimates')
legend('Position (m)','Velocity (m/s)','Position Estimate (m)','Velocity Estimate (m/s)')

figure()
x_error = xt(1,:)-mu(1,:);
xdot_error = xt(2,:)-mu(2,:);
subplot(2,1,1)
plot(t,x_error)
hold on
x_err_mean = mean(x_error);
x_err_std = std(x_error);
x_upper = (x_err_mean + 2*x_err_std)*ones(length(t));
x_lower = (x_err_mean - 2*x_err_std)*ones(length(t));
plot(t,x_upper,'r')
plot(t,x_lower,'r')
xlabel('Time (sec)')
ylabel('Position Error (m)')

subplot(2,1,2)
plot(t,xdot_error)
hold on
xdot_err_mean = mean(xdot_error);
xdot_err_std = std(xdot_error);
xdot_upper = (xdot_err_mean + 2*xdot_err_std)*ones(length(t));
xdot_lower = (xdot_err_mean - 2*xdot_err_std)*ones(length(t));
plot(t,xdot_upper,'r')
plot(t,xdot_lower,'r')
xlabel('Time (sec)')
ylabel('Velocity Error (m/s)')
suptitle('Estimation Error')

figure()
tt = 0:Ts/2:50;
sig_pos = reshape(sigma_predict_update(1,1,:),length(sigma_predict_update),1);
plot(tt(1:200),sig_pos(1:200))
%hold on
%plot(reshape(sigma_predict_update(2,2,:),length(sigma_predict_update),1))
xlabel('Time Step')
ylabel('Covariance (m^2)')
title('Error Covariance vs. Time Step')
legend('Position Covariance')

figure()
plot(t, K_t(1,:))
hold on
plot(t,K_t(2,:))
xlabel('Time (sec)')
ylabel('Kalman Gains')
title('Kalman Gains vs. Time')
legend('Position gain', 'Velocity gain')
