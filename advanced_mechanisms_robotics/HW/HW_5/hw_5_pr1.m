clear all
close all

% School Linux
%run ~/Desktop/rvctools/startup_rvc.m
% Personal laptop
%run C:\Users\Dustan\Desktop\rvctools\startup_rvc.m
%% Problem 1a) 7-3
syms x y z a b c rho
% note: a*b*c*rho = m
% using direct calculation
Ixx = rho*int(int(int(y^2 + z^2,x,0,a),y,0,b),z,0,c);
Iyy = rho*int(int(int(x^2 + z^2,x,0,a),y,0,b),z,0,c);
Izz = rho*int(int(int(x^2 + y^2,x,0,a),y,0,b),z,0,c);
Ixy = -rho*int(int(int(x*y,x,0,a),y,0,b),z,0,c);
Ixz = -rho*int(int(int(x*z,x,0,a),y,0,b),z,0,c);
Iyz = -rho*int(int(int(y*z,x,0,a),y,0,b),z,0,c);
Iyx = Ixy;
Izx = Ixz;
Izy = Iyz;
I_1 = [Ixx Ixy Ixz; Iyx Iyy Iyz; Izx Izy Izz];

% using parallel axis theorem (Tensor generalization from wikipedia)
Ixx = rho*int(int(int(y^2 + z^2,x,-a/2,a/2),y,-b/2,b/2),z,-c/2,c/2);
Iyy = rho*int(int(int(x^2 + z^2,x,-a/2,a/2),y,-b/2,b/2),z,-c/2,c/2);
Izz = rho*int(int(int(x^2 + y^2,x,-a/2,a/2),y,-b/2,b/2),z,-c/2,c/2);
Ixy = -rho*int(int(int(x*y,x,-a/2,a/2),y,-b/2,b/2),z,-c/2,c/2);
Ixz = -rho*int(int(int(x*z,x,-a/2,a/2),y,-b/2,b/2),z,-c/2,c/2);
Iyz = -rho*int(int(int(y*z,x,-a/2,a/2),y,-b/2,b/2),z,-c/2,c/2);
Iyx = Ixy;
Izx = Ixz;
Izy = Iyz;
Icom = [Ixx Ixy Ixz; Iyx Iyy Iyz; Izx Izy Izz];
R = [-a/2 -b/2 -c/2];
I_2 = simplify(Icom + a*b*c*rho*((R*R.')*eye(3) - R.'*R));

%% Problem 1c) 7-7

% Part a
Ixx = (1/12)*(1+0.25^2);
Iyy = Ixx;
Izz = (1/12)*(0.25^2 + 0.25^2);

% Part b
Jv1 = [0 0 0; 0 0 0; 1 0 0];
Jv2 = [0 0 0; 0 1 0; 1 0 0];
Jv3 = [0 0 -1; 0 1 0; 1 0 0];

Dq = Jv1.'*Jv1 + Jv2.'*Jv2 + Jv3.'*Jv3;

%% Problem 1d) 7-10
clear all
L = 0.4; %link length
m = 1; %link mass
b = 0.1;
h = 0.1;
rho = m/(L*b*h);
% Make robots
Lk_1 = Link('revolute', 'offset', 0, 'd', 0, 'a', L/2, 'alpha', 0);

Lk_2(1) = Link('revolute', 'offset', 0, 'd', 0, 'a', L, 'alpha', 0);
Lk_2(2) = Link('revolute', 'offset', 0, 'd', 0, 'a', L/2, 'alpha', 0);

Lk_3(1) = Link('revolute', 'offset', 0, 'd', 0, 'a', L, 'alpha', 0);
Lk_3(2) = Link('revolute', 'offset', 0, 'd', 0, 'a', L, 'alpha', 0);
Lk_3(3) = Link('revolute', 'offset', 0, 'd', 0, 'a', L/2, 'alpha', 0);

% Make the first robot to get Jacobian for the c.o.m. of the first link
bot1 = SerialLink(Lk_1, 'name', 'com1');
% Make the 2nd robot to get Jacobian for the c.o.m. of the 2nd link
bot2 = SerialLink(Lk_2, 'name', 'com2');
% Make the 3rd robot to get Jacobian for the c.o.m. of the 3rd link
bot3 = SerialLink(Lk_3, 'name', 'com3');

syms q q1 q2 q3
q = [q1, q2, q3];
% Jacobian to c.o.m. of 1st link
J1 = [bot1.jacob0([q(1)]), [0;0;0;0;0;0], [0;0;0;0;0;0]];
Jv1 = J1(1:3,:);
Jw1 = J1(4:6,:);
% Jacobian to c.o.m. of 2nd link
J2 = [bot2.jacob0([q(1) q(2)]), [0;0;0;0;0;0]];
Jv2 = J2(1:3,:);
Jw2 = J2(4:6,:);
% Jacobian to c.o.m. of 3rd link
J3 = bot3.jacob0([q(1) q(2) q(3)]);
Jv3 = J3(1:3,:);
Jw3 = J3(4:6,:);

%Inertia tensor for each link at the c.o.m.
Ixx = (1/12)*(b^2 + h^2);
Iyy = (1/12)*(L^2 + b^2);
Izz = Iyy;
I1 = [Ixx 0 0; 0 Iyy 0; 0 0 Izz];
I2 = I1;
I3 = I1;

%Get rotation matrices for each c.o.m. tensor to the base frame
T1 = bot1.fkine([q(1)]);
R1 = T1(1:3,1:3);

T2 = bot2.fkine([q(1) q(2)]);
R2 = T2(1:3,1:3);

T3 = bot3.fkine([q(1) q(2) q(3)]);
R3 = T3(1:3,1:3);

% Calculate D(q)
D1q = m*Jv1.'*Jv1 + Jw1.'*R1*I1*R1.'*Jw1;
D2q = m*Jv2.'*Jv2 + Jw2.'*R2*I2*R2.'*Jw2;
D3q = m*Jv3.'*Jv3 + Jw3.'*R3*I3*R3.'*Jw3;
Dq = simplify(D1q + D2q + D3q)
C = sym(zeros(3));

% Calculate C(q, qdot)
for k = 1:3, 
    for j = 1:3, 
        for i = 1:3,
            C(k,j) = C(k,j) + (1/2)*(diff(Dq(k,i),q(j)) + ...
                diff(Dq(k,i),q(j)) - diff(Dq(i,j),q(k))); 
        end
    end
end
C = simplify(C)

% Calculate g(q)
g = [0; 9.81; 0];
rc = [T1(1:3,4), T2(1:3,4), T3(1:3,4)];
P = sym(0);
for i = 1:3, 
    P = P + m*g.'*rc(:,i);
end
P = simplify(P);
gq = sym(zeros(3,1));
for i = 1:3,
    gq(i) = diff(P,q(i));
    gq(i) = simplify(gq(i));
end

gq

syms Tau Tau1 Tau2 Tau3
Tau = [Tau1; Tau2; Tau3];