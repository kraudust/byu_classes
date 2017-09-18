clear all
close all
clc

%Vehicle Parameters
m = 100; %kg (Mass)
b = 20; %N-s/m (Linear Drag Coeficcient)
Ts = 0.05; %sample period in seconds

% Set Up Input to System
t = (0:0.05:50)';
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
Q = 0.001; % Measurement Covariance (we are measuring position only)

% Initial Conditions
x0 = [0; 0];
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

for i = 2:length(t)
    % Get sensor measurement
    z_t = C*xt(:,i) + Q.^(0.5)*randn(1);
    [mu(:,i), sigma(:,:,i), K_t(:,i-1), sigma_bart] = kalman_filter_func(mu(:,i-1), sigma(:,:,i-1), u(i-1), z_t, A, B, C, R, Q);
end
K_t(:,length(t)) = K_t(:,length(t)-1);

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
plot(t,x_error)
hold on
plot(t,xdot_error)
%calculate mean and standard deviation of error to plot 95% confidence intervals
x_err_mean = mean(x_error);
x_err_std = std(x_error);
xdot_err_mean = mean(xdot_error);
xdot_err_std = std(xdot_error);
x_upper = (x_err_mean + 2*x_err_std)*ones(length(t));
x_lower = (x_err_mean - 2*x_err_std)*ones(length(t));
xdot_upper = (xdot_err_mean + 2*xdot_err_std)*ones(length(t));
xdot_lower = (xdot_err_mean - 2*xdot_err_std)*ones(length(t));
plot(t,x_upper)
plot(t,x_lower)
plot(t,xdot_upper)
plot(t,xdot_lower)
xlabel('Time (sec)')
ylabel('State Error Magnitude')
title('Estimation Error')
legend('Position Error (m)', 'Velocity Error (m/s)')

figure()
plot(t, reshape(sigma(1,1,:),length(t),1))
hold on
plot(t,reshape(sigma(2,2,:),length(t),1))
xlabel('Time (sec)')
ylabel('Error Covariance')
title('Error Covariance vs. Time')
legend('Position Covariance', 'Velocity Covariance')

figure()
plot(t, K_t(1,:))
hold on
plot(t,K_t(2,:))
xlabel('Time (sec)')
ylabel('Kalman Gains')
title('Kalman Gains vs. Time')
legend('Position gain', 'Velocity gain')
