% checking random set point function
close all;
clear all;


A =-3; B=2; C=1; D=0;
G = C; H=D;


% original system
sys = ss(A,B,C,D);

w = 10;
Q = w*C'*C;
R = 10000;
[K P e] = lqr(A,B,Q,R);

% unaugmented LQR (no set point)
Alqr = A-B*K;
Blqr = B;
Clqr = C;
Dlqr = D;
[numlqr, denlqr] = ss2tf(Alqr,Blqr,Clqr,Dlqr);
syslqr = tf(numlqr,denlqr);


% look at the set point tracking approach
Aa = [A B; G H];
temp= inv(Aa)*[0;1];
F=temp(1);
N=temp(2);

s = tf('s');
L_hat = K*inv(s-A)*B;
Gp_hat = G*inv(s-A)*B+H;

CLTF = Gp_hat*inv(1+L_hat)*(N+K*F);



% desired tracking point
r=5;

t = 0:0.01:5;
figure(2); hold on;
step(r*sys,t);
step(r*syslqr,t);
step(r*CLTF,t)
grid on
legend( 'original sys'  ,'lqr sys', 'set point sys' )
hold off