%% Problem 1c - 4-13
% Create Variables
syms psi theta phi psidot thetadot phidot

% Create R
Rz_psi = [...
    cos(psi),   -sin(psi),  0;...
    sin(psi),   cos(psi),   0;...
    0       ,   0       ,   1];
Ry_th = [...
    cos(theta),     0,  sin(theta);...
    0         ,     1,  0;...
    -sin(theta),    0,  cos(theta)];
Rz_phi = [...
    cos(phi),   -sin(phi),  0;...
    sin(phi),   cos(phi),   0;...
    0       ,   0       ,   1];
R = Rz_psi*Ry_th*Rz_phi;

% Create Skew Symmetric Matrix
w = [cos(psi)*sin(theta)*phidot - sin(psi)*thetadot;...
    sin(psi)*sin(theta)*phidot + cos(psi)*thetadot;...
    psidot + cos(theta)*phidot];

S_w = [0, -w(3), w(2);...
    w(3), 0, -w(1);...
    -w(2), w(1), 0]; %equation 4.5 on pg 122

% Find Derivative using S*R
dR_dt_skew = simplify(S_w*R);

% Find Derivative using symbolic derivative
dR_dt_sym = diff(R,psi)*psidot + diff(R,theta)*thetadot +...
    diff(R,phi)*phidot;
dR_dt_sym = simplify(dR_dt_sym);

% Compare the two derivatives
check_equality = simplify(dR_dt_sym - dR_dt_skew)

%% Problem 1f - 4-17 Check
clear all
close all
clc
syms th1 th2 th3 a1 a2 a3
A1 = calc_A(th1, a1, 0, 90);
A2 = calc_A(th2, 0, a2, 0);
A3 = calc_A(th3, 0, a3, 0);

T2_0 = A1*A2;
T2_0 = simplify(T2_0);
T3_0 = A1*A2*A3;
T3_0 = simplify(T3_0);

J11 = simplify([cross([0;0;1],T3_0(1:3,4)),...
    cross([sin(th1); -cos(th1); 0], T3_0(1:3,4)-A1(1:3,4)),...
    cross([sin(th1); -cos(th1); 0], T3_0(1:3,4) - T2_0(1:3,4))]);

det = simplify(det(J11));

%% Problem 2a
close all
clear all
clc
syms th1 th2 a1 a2
A1 = calc_A(th1, 0, a1, -90);
A2 = calc_A(th2, 0, a2, 0);
T2_0 = simplify(A1*A2);
z0_0 = [0;0;1];
z1_0 = A1(1:3, 3);
o2_0 = T2_0(1:3,4);
o1_0 = A1(1:3,4);
J2_0 = simplify([cross(z0_0,o2_0),cross(z1_0,o2_0-o1_0);...
    z0_0, z1_0]); %jacobian at point 2 in frame 0
% i) Find jacobian at point 2 in frame 2 by direct calculation
o2_1 = A2(1:3,4);
o0_0 = [0;0;0];
T0_2 = T2_0^-1;
o0_2 = simplify(T0_2(1:3,4));
o1_2 = simplify(T0_2*[o1_0;1]);
o1_2 = o1_2(1:3);
R0_2 = simplify(T2_0(1:3, 1:3).');
z0_2 = simplify(R0_2*z0_0);
z1_2 = simplify(R0_2*z1_0);
J2_2_direct = simplify([cross(z0_2,-o0_2), cross(z1_2,-o1_2); z0_2, z1_2])

% ii) Find jacobian at point 2 in frame 2 using rotation matrix
R2_0 = T2_0(1:3, 1:3); % p0 = Rn_0*pn
R0_2 = R2_0.'; %if I only use ', conj appears
mat_0 = [0,0,0;0,0,0;0,0,0];
J2_2_rot = simplify([R0_2, mat_0;mat_0, R0_2]*J2_0)

%% Problem 2b
close all
clear all
clc
syms th1 th2 a1 a2
A1 = calc_A(th1, 0, a1, -90);
A2 = calc_A(th2, 0, a2, 0);
T2_0 = simplify(A1*A2);
z0_0 = [0;0;1];
z1_0 = A1(1:3, 3);
o2_0 = T2_0(1:3,4);
o1_0 = A1(1:3,4);
J2_0 = simplify([cross(z0_0,o2_0),cross(z1_0,o2_0-o1_0);...
    z0_0, z1_0]); %jacobian at point 2 in frame 0

J2_0_i = subs(J2_0,[th1 th2 a1 a2],[0 pi/4 1 1]);
J2_0_ii = subs(J2_0,[th1 th2 a1 a2],[0 pi/2 1 1]);
J2_0_iii = subs(J2_0,[th1 th2 a1 a2],[pi/4 pi/4 1 1]);
J2_0_iv = subs(J2_0,[th1 th2 a1 a2],[0 0 1 1]);
J2_0_v = subs(J2_0,[th1 th2 a1 a2],[0 0 1 1]);

Fi = [-1;0;0;0;0;0];
Fii = [-1;0;0;0;0;0];
Fiii = [-1;-1;0;0;0;0];
Fiv = [0;0;1;0;0;0];
Fv = [1;0;0;0;0;0];

Tau_i = double(J2_0_i.'*Fi)
Tau_ii = double(J2_0_ii.'*Fii)
Tau_iii = double(J2_0_iii.'*Fiii)
Tau_iv = double(J2_0_iv.'*Fiv)
Tau_v = double(J2_0_v.'*Fv)

%If I switch to reaction forces, then the signs switch

    
    
    
    