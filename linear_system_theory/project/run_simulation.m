clear
close all
clc

%% System Parameters
m = 12;
g = 9.81;
a = 0.5;  %side lengths x-dir
b = 1.00; %side lengths y-dir
c = 0.25; %side lengths z-dir
Ixx = (1/12)*m*(b^2 + c^2);
Iyy = (1/12)*m*(a^2 + c^2);
Izz = (1/12)*m*(a^2 + b^2);
Ixz = 0;
Ixy = 0;
Iyz = 0;
I = [Ixx, 0, 0; 0, Iyy, 0; 0, 0, Izz];
% x = [p, q,  r,  psi, th, phi, px,  py,   pz, u, v, w]';
x0 = [0;  0;  0;  0;   0;  0;   5;  3;  1;   0; 0; 0];
xh0 = [0;  0;  0;  0;   0;  0;   0;  0;  0;   0; 0; 0];
T = [];
d1x = 0.2;
d1y = 0.4;
d1z = 0.1;
d = [d1x, d1y, d1z];

%% Find u_eq (set xdot = 0) Aeq*u_eq = beq -> u_eq = inv(Aeq)*beq
%Assuming forces and torques applied at 1 position
%Aeq*[Tx;Ty;Tz;Fx;Fy;Fz] = beq;
Aeq = [...
    1, 0, 0, 0, -d1z, d1y;...
    0, 1, 0, d1z, 0, -d1x;...
    0, 0, 1, -d1y, d1x, 0;...
    0, 0, 0, 1/m, 0, 0;...
    0, 0, 0, 0, 1/m, 0;...
    0, 0, 0, 0, 0, 1/m];
beq = [0;0;0;0;0;g];

u_eq = Aeq\beq;

%check = object_deriv([], [xh0;x0], I, m, u_eq(1:3)', u_eq(4:6)', d, [], [],'n',[],[], [],[])
%% Linearize System
syms Mx My Mz wx wy wz Vx Vy Vz psi th phi Fx Fy Fz px py pz Tx Ty Tz

% Angular Velocity
F_ang_acc = [...
    Mx - Ixy*wx*wz + Ixz*wx*wy + (Iyy - Izz)*wy*wz + Iyz*(wy^2 - wz^2);...
    My - Iyz*wx*wy + Ixy*wy*wz + (Izz-Ixx)*wx*wz + Ixz*(wz^2 - wx^2);...
    Mz - Ixz*wy*wz + Iyz*wx*wz + (Ixx-Iyy)*wx*wy + Ixy*(wx^2 - wy^2)];%,[Mx, My, Mz],[Tx + Fz*d1y - Fy*d1z, Ty + Fx*d1z - Fz*d1x, Tz - Fx*d1y + Fy*d1x]);
ang_accel = I\F_ang_acc;
wxdot = simplify(ang_accel(1));
wydot = simplify(ang_accel(2));
wzdot = simplify(ang_accel(3));

% Angular Position
psidot = (1/cos(th))*(wy*sin(phi) + wz*cos(phi));
thdot = wy*cos(phi) - wz*sin(phi);
phidot = wx + (1/cos(th))*(wy*sin(phi)*sin(th) + wz*cos(phi)*sin(th));

% Linear Position
pxdot = Vx*cos(psi)*cos(th) - Vy*(cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(th)) + Vz*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(th));
pydot = Vx*sin(psi)*cos(th) + Vy*(cos(phi)*cos(psi) + sin(psi)*sin(th)*sin(phi)) - Vz*(cos(psi)*sin(phi) - sin(psi)*sin(th)*cos(phi));
pzdot = -Vx*sin(th) + Vy*cos(th)*sin(phi) + Vz*cos(th)*cos(phi);

% Linear Velocity
Vxdot = (1/m)*Fx + g*sin(th) - Vz*wy + Vy*wz;
Vydot = (1/m)*Fy - g*cos(th)*sin(phi) + Vz*wx - Vx*wz;
Vzdot = (1/m)*Fz - g*cos(th)*cos(phi) - Vy*wx + Vx*wy;

xdot = [wxdot; wydot; wzdot; psidot; thdot; phidot; pxdot; pydot; pzdot; Vxdot; Vydot; Vzdot];
xdot = subs(xdot,[Mx, My, Mz],[Tx + Fz*d1y - Fy*d1z, Ty + Fx*d1z - Fz*d1x, Tz - Fx*d1y + Fy*d1x]);
for i = 1:12
A(i,:) = [diff(xdot(i), wx), diff(xdot(i), wy), diff(xdot(i), wz), diff(xdot(i), psi), diff(xdot(i), th), diff(xdot(i), phi), diff(xdot(i), px), diff(xdot(i), py), diff(xdot(i), pz), diff(xdot(i), Vx), diff(xdot(i), Vy), diff(xdot(i), Vz)];  
B(i,:) = [diff(xdot(i), Tx), diff(xdot(i), Ty), diff(xdot(i), Tz), diff(xdot(i), Fx), diff(xdot(i), Fy), diff(xdot(i), Fz)];
end

A = double(subs(A,[wx, wy, wz, Vx, Vy, Vz, psi, th, phi],[0,0,0,0,0,0,0,0,0]));
B = double(subs(B,[wx, wy, wz, Vx, Vy, Vz, psi, th, phi],[0,0,0,0,0,0,0,0,0]));
% x = [p, q,  r,  psi, th, phi, px,  py,   pz, u, v, w]';
%C = eye(12);
C = [0 0 0 1 0 0 0 0 0 0 0 0;...
    0 0 0 0 1 0 0 0 0 0 0 0;...
    0 0 0 0 0 1 0 0 0 0 0 0;...
    0 0 0 0 0 0 1 0 0 0 0 0;...
    0 0 0 0 0 0 0 1 0 0 0 0;...
    0 0 0 0 0 0 0 0 1 0 0 0];
%D = zeros(12,6);
D = zeros(6,6);

%% Check for controllability and observability
CO = ctrb(A,B);
rank(CO)
OB = obsv(A,C);
rank(OB)


%% LQR Design or LQG if I get it working
Q = eye(12);
Q(1,1) = 10;
Q(2,2) = 10;
Q(3,3) = 10;
Q(4,4) = 100;
Q(5,5) = 100;
Q(6,6) = 100;
Q(7,7) = 100;
Q(8,8) = 100;
Q(9,9) = 100;
Q(10,10) = 10;
Q(11,11) = 10;
Q(12,12) = 10;
%rho = 0.001;
S = care(A', C', 100*Q);
N = eye(6);
L = S*C'*inv(N);
rho = 0.1;
R = rho*eye(6);
[K,P,E] = lqr(A,B,Q,R);

%% Simulate System
dt = 0.01;
t0 = 0;
t1 = 6;
t = t0:dt:t1; 
model_type = 'n';
xdes = 10;
ydes = 15;
zdes = 20;
[t,x] = ode45(@object_deriv, t, [xh0;x0], [], I, m, [], [], d, K, u_eq, model_type, A, B, L, C);  
for i= 1:length(x)
    u(:,i) = -K*x(i,1:12)' + u_eq;
end
for i = 1:length(t)
    drawObject([x(i,1:12), t(i)], a, b, c)
    pause(0.001);
end
figure()
suptitle('R matrix 0.1')
subplot(3,2,4)
plot(t,x(:,1)*180/pi)
hold on
plot(t,x(:,2))
plot(t,x(:,3))
ylabel('Angular Velocity (Deg/sec)')
legend('wx','wy','wz')

subplot(3,2,2)
plot(t,x(:,4)*180/pi)
hold on
plot(t,x(:,5))
plot(t,x(:,6))
ylabel('Euler Angles (Deg)')
legend('psi','theta','phi')

subplot(3,2,1)
plot(t,x(:,7))
hold on
plot(t,x(:,8))
plot(t,x(:,9))
ylabel('Position (m)')
legend('x','y','z')

subplot(3,2,3)
plot(t,x(:,10))
hold on
plot(t,x(:,11))
plot(t,x(:,12))
ylabel('Velocity (m/s)')
legend('Vx','Vy','Vz')

subplot(3,2,5)
hold on
plot(t,u(1,:))
plot(t,u(2,:))
plot(t,u(3,:))
ylabel('Torque N-m')
xlabel('Time (sec)')
legend('Tx', 'Ty', 'Tz')

subplot(3,2,6)
hold on
plot(t,u(4,:))
plot(t,u(5,:))
plot(t,u(6,:))
ylabel('Force N')
xlabel('Time (sec)')
legend('Fx', 'Fy', 'Fz')
