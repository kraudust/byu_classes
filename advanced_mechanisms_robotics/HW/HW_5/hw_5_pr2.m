clear all
close all

% School Linux
%run ~/Desktop/rvctools/startup_rvc.m
% Personal laptop
%run C:\Users\Dustan\Desktop\rvctools\startup_rvc.m

%% Problem 2b
load desired_accel.mat
qdd = q;
clear q
%initialize qd and q
qd = [0 0 0 0 0 0 0];
q = qd;
for i = 2:length(qdd),
   qd(i,:) =  qd(i-1,:) + qdd(i-1,:)*(t(i) - t(i-1));
   q(i,:) =  q(i-1,:) + qd(i-1,:)*(t(i) - t(i-1));
end

[left, right] = mdl_baxter('sim');
tau = zeros(size(qdd));
%calculating the mass, corilios and gravity terms
for i = 1:length(qdd),
    M = left.inertia(q(i,:));
    C = left.coriolis(q(i,:), qd(i,:));
    G = left.gravload(q(i,:));
    tau(i,:) = (M*qdd(i,:).' + C*qd(i,:).' + G.').';
end

figure()
plot(t,tau)
legend('1', '2', '3', '4', '5', '6', '7');
xlabel('Time (sec)')
ylabel('Joint Torque')
