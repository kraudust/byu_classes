% Vehicle Parameters
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
xt = zeros(2,1001); %true states
xt(:,1) = x0;

% Simulate the System to get True States
for i = 2:length(t)
    xt(:,i) = A*xt(:,i-1) + B*u(i-1) + sqrt(R)*randn(2,1);
end

figure()
plot(t,xt(1,:))
hold on
plot(t,xt(2,:))

% Run Kalman Filter


