%% Validating what I think about 7 
Jv1 = [0 0 0;
       0 0 0;
       1 0 0];
   
Jv2 = [0 0 0;
       0 0 0;
       1 1 0];
   
Jv3 = [0 0 0;
       0 0 0;
       1 1 1];
   
J1 = Jv1'*Jv1
J2 = Jv2'*Jv2
J3 = Jv3'*Jv3

J = J1 + J2 + J3

%% Problem 10
syms m l q1 q2 q3 qd1 qd2 qd3 real

Jvc1 = [l/2*cos(q1) 0 0;
        l/2*sin(q1) 0 0;
        0 0 0];
    
Jw1 = [0 0 0;
        0 0 0;
        1 0 0];
    
Jvc2 = [l*cos(q1) l/2*cos(q1+q2) 0;
        l/2*sin(q1) l/2*sin(q1+q2) 0;
        0 0 0];
    
Jw2 = [0 0 0;
        0 0 0;
        1 1 0];
    
Jvc3 = [l*cos(q1) l*cos(q1+q2) l/2*cos(q1+q2+q3);
        l/2*sin(q1) l*sin(q1+q2) l/2*sin(q1+q2+q3);
        0 0 0];
    
Jw3 = [0 0 0;
        0 0 0;
        1 1 1];
    
I = [m*l^2/12 0 0;
     0 0 0;
     0 0 m*l^2/12];
 
R01 = [cos(q1) -sin(q1) 0;
       sin(q1) cos(q1) 0;
       0 0 1];
   
R12 = [cos(q2) -sin(q2) 0;
       sin(q2) cos(q2) 0;
       0 0 1];
   
R23 = [cos(q3) -sin(q3) 0;
       sin(q3) cos(q3) 0;
       0 0 1];
   
R02 = R01*R12;
R03 = R01*R12*R23;

D = m*Jvc1'*Jvc1 + Jw1'*R01*I*R01'*Jw1 +...
    m*Jvc2'*Jvc2 + Jw2'*R02*I*R02'*Jw2 +...
    m*Jvc3'*Jvc3 + Jw3'*R03*I*R03'*Jw3;

% Christoffel symbols
q = [q1,q2,q3];
for i=1:3
    for j=1:3
        for k=1:3
            c(i,j,k) = .5*(diff(D(k,j),q(i)) + diff(D(k,i),q(j)) - diff(D(i,j),q(k)));
        end
    end
end

qd = [qd1,qd2,qd3];
for k=1:3
    for j=1:3
        for i=1:3
            C(k,j) = c(i,j,k)*qd(i);
        end
    end
end

D = simplify(D)
C = simplify(C)

%% Problem 2

[left, right] = mdl_baxter('sim');
 
load('torque_profile.mat');
load('phil_tau.mat');
%tau(:,7) = 0;

%t = t(1:500);
%tau = tau(1:500,:);

%qdes = zeros(length(t),7);
qdes = [sin(2*t)];


figure(1)
plot(t,qdes)
legend('1','2','3','4','5','6','7')

[tout,yout] = ode15s(@my_odefunc,t,[q0; qd0],[],right,t,qdes);

figure(2)
subplot(2,1,1)
plot(tout,yout(:,1:7))
legend('1','2','3','4','5','6','7')
title('Joint angles')
subplot(2,1,2)
plot(tout,yout(:,8:14))
legend('1','2','3','4','5','6','7')
title('Joint velocities')

%% Problem 2b
load('desired_accel.mat')
[left, right] = mdl_baxter('sim');
qdd = q;
q0 = zeros(7,1);
qd0 = zeros(7,1);

[tout,yout] = ode45(@odefunc2,t,[q0; qd0],[],qdd,t);

figure(2)
subplot(2,1,1)
plot(tout,yout(:,1:7))
legend('1','2','3','4','5','6','7')
title('Joint angles')
subplot(2,1,2)
plot(tout,yout(:,8:14))
legend('1','2','3','4','5','6','7')
title('Joint velocities')

q = yout(:,1:7);
qd = yout(:,8:14);


tau = zeros(size(qdd));
for i=1:length(qdd)
    i
    M = left.inertia(q(i,:));
    C = left.coriolis(q(i,:), qd(i,:));
    G = left.gravload(q(i,:));
    tau(i,:) = M*qdd(i,:)' + C*qd(i,:)' + G';
end

plot(tout,tau)