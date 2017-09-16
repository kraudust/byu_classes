close all
clear all

%% Problem 17.1
A = [-1 0; 0 -1];
B = [1 0; 0 1];
C = [-1 1];
D = [2 1];

Controllability = ctrb(A,B);
rank(Controllability);

Observability = obsv(A,C);
rank(Observability);

T = [-1 1; 1 1];
Abar = inv(T)*A*T;
Bbar = inv(T)*B;
Cbar = C*T;
Dbar = D;

%% Problem 1
A = [0 1 0; 0 0 1; -2.1 -0.65 -0.1];
B = [0;0;-2];
Q = [1 0 0; 0 0 0; 0 0 0];
R = 1;
[K, P, E] = lqr(A,B,Q,R);
x0 = [1;1;1];
Jmin = x0'*P*x0;

%% Problem 2
% x1 = x;
% x2 = xdot;
% x1dot = x2;
% x2dot = -wn^2*x - 2*wn*zeta + u;
wn = 1.5;
zeta = 2;
A = [0 1; -wn^2 0];
B = [0; 1];
%C = eye(2);
%C = [1 0; 0 0];
C = [1 0];
Q = 1*eye(2);
R = .001;
Qk = eye(2);
Rk = .1;
S = care(A', C', Q, R);
L = S*C'*inv(R);
L1 = lqr(A',C',Q,R);
L2 = lqe(A,eye(2),C,Q,R);
K = lqr(A,B,Qk,Rk);
x0 = [1; 1];
xhat0 = [0; 0];
init = [x0; xhat0];

tspan = [0 100];
[t, out] = ode45(@dynamics,tspan,init, [], zeta, wn, A, B, L, C, K);

figure()
plot(t,out(:,1))
hold on
plot(t,out(:,3), '--')
xlabel('Time (sec)')
ylabel('Magnitude')
legend('x','xhat')

figure()
plot(t,out(:,2))
hold on
plot(t,out(:,4), '--')
xlabel('Time (sec)')
ylabel('Magnitude')
legend('xdot','xhatdot')

figure()
plot(t,out(:,1)-out(:,3))
hold on
plot(t,out(:,2)-out(:,4), '--')
xlabel('Time (sec)')
ylabel('x-xhat')
legend('error x', 'error xdot')

eig(A-L*C);

function out = dynamics(t,states, zeta, wn, A, B, L, C, K)
    x = states(1:2);
    xh = states(3:4);
    %u = 3 + 0.5*sin(0.75*t) - 2*zeta*wn;
    u = -K*xh;
    xd = A*x + B*u;
    xhd = (A - L*C)*xh + B*u + L*C*x;
    out = [xd;xhd];
end


