close all
clear all
clc
%----------------------------Generate Truth Data---------------------------
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
% alpha3 = n*0.01;% K1norm = zeros(1,length(t));
% alpha4 = n*0.1;
alpha = [alpha1; alpha2; alpha3; alpha4];

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

%--------------------Generate Measurments----------------------------------
%Landmark locations
xland = [6, -7, 6];
yland = [4, 8, -4];
% xland = [-7];
% yland = [8];
lm = [xland', yland'];
% lm = 10-20*rand(5,2);

%Standard deviations of range and bearing sensor noise in meters & radians
sigma_range = 0.1;
sigma_phi = 0.05;
% sigma_range = 10*0.1;
% sigma_phi = 10*0.05;

%Simulate range and bearing measurements
[r, phi] = sim_measurements(lm,sigma_range,sigma_phi,xt);
z = [r;phi];


%Run Particle Filter to get state estimates--------------------------------
%Generate Particles
M = 1000; %number of particles
rand_x = -10 + 20.*rand(1,M);
rand_y = -10 + 20.*rand(1,M);
rand_th = -pi + 2*pi.*rand(1,M);
chi0 = [rand_x; rand_y; rand_th];
particles = zeros(length(t), 3, M);
particles(1,:,:) = chi0;
mu = zeros(3,length(t));
sigma = zeros(3,length(t));
mu(:,1) = mean(chi0,2);
sigma(:,1) = std(chi0,0,2);
for i = 1:length(t)-1
   [particles(i+1, :, :), mu(:,i+1), sigma(:,i+1)] = particle_filter(reshape(particles(i,:,:),3,M), u(:,i), z(:,i+1), lm, Ts, alpha, sigma_range, sigma_phi);
end

%Draw Robot
draw_robot(t,xt,lm,r,phi,mu, particles);

%Make Plots----------------------------------------------------------------
%States and State Estimates
figure()
subplot(3,1,1)
plot(t,xt(1,:))
hold on
plot(t,mu(1,:))
ylabel('x Position (m)')
legend('True State','MCL Estimate')

subplot(3,1,2)
plot(t,xt(2,:))
hold on
plot(t,mu(2,:))
ylabel('y Position (m)')
legend('True State','MCL Estimate')

subplot(3,1,3)
plot(t,xt(3,:))
hold on
plot(t,mu(3,:))
xlabel('Time (sec)')
ylabel('Heading (Rad)')
legend('True State','MCL Estimate')
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
x_upper = 2*sigma(1,:);
x_lower = -2*sigma(1,:);
plot(t,x_upper,'r')
plot(t,x_lower,'r')
ylabel('x Position Error (m)')

subplot(3,1,2)
plot(t,y_error)
hold on
y_upper = 2*sigma(2,:);
y_lower = -2*sigma(2,:);
plot(t,y_upper,'r')
plot(t,y_lower,'r')
ylabel('y Position Error (m)')

subplot(3,1,3)
plot(t,th_error)
hold on
th_upper = 2*sigma(3,:);
th_lower = -2*sigma(3,:);
plot(t,th_upper,'r')
plot(t,th_lower,'r')
xlabel('Time (sec)')
ylabel('Theta Angle Error (Rad)')
suptitle('Estimation Error')



