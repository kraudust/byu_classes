clear all
clc;

%% Problem 1(a)
% School Linux
%run ~/Desktop/rvctools/startup_rvc.m
% Personal laptop
%run C:\Users\Dustan\Desktop\rvctools\startup_rvc.m

%define the robotics toolbox Puma 560 arm
mdl_puma560;

%set the Coulomb friction terms to zero to help with numerical simulation
p560 = p560.nofriction;

%load the torque profile and open the simulink model
load puma560_torque_profile.mat
out = sim('sl_puma_hw6.slx');

%On my Computer
q = out.get('q_sim');
qd = out.get('qd_sim');
qdd = out.get('qdd_sim');
t_sim = out.get('t_sim');

%% Problem 1(b)
%D(q)*qdd + C(q,qd)*qd + G(q) = Tau
q0 = [0; 0; 0; 0; 0; 0];
qd0 = [0; 0; 0; 0; 0; 0];
%qdd0 = p560.accel(q0, qd0, torque(1,:));
% D0 = p560.inertia(q0.'); %inertia matrix
% C0 = p560.coriolis(q0.',qd0.'); %coriolis matrix
% G0 = p560.gravload(q0.').';
[t,x] = ode45(@p560_dyn_eqns, [0, 10], [q0, qd0],[],p560,torque,time);
q_ode45 = x(:,1:6);
qd_ode45 = x(:,7:12);
for i = 1:length(t)
    qdd_ode45(i,:) = p560.accel(q_ode45(i,:),qd_ode45(i,:),...
        interp1(time,torque,t(i)));
end

%% Problem 1(c) Make Plots
figure(1)
for i = 1:6,
    subplot(3,2,i)
    hold on
    plot(t_sim,q(:,i)*180/pi)
    plot(t,q_ode45(:,i)*180/pi)
    if i == 6 || i == 5,
        xlabel('t')
    end
    ylabel(strcat('q_',num2str(i),' (deg)'))
end

figure(2)
for i = 1:6,
    subplot(3,2,i)
    hold on
    plot(t_sim,qd(:,i)*180/pi)
    plot(t,qd_ode45(:,i)*180/pi)
    if i == 6 || i == 5,
        xlabel('t')
    end
    ylabel(strcat('qd_',num2str(i),' (deg)'))
end

figure(3)
for i = 1:6,
    subplot(3,2,i)
    hold on
    plot(t_sim,qdd(:,i)*180/pi)
    plot(t,qdd_ode45(:,i)*180/pi)
    if i == 6 || i == 5,
        xlabel('t')
    end
    ylabel(strcat('qdd_',num2str(i),' (deg)'))
end

% The ode45 simulation gave the same output as the simulink simulation with
% a little bit higher fidelity.