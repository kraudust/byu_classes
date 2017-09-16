%% Problem 1: 2.20 from the text 
%Find Symbolic Rotation Matrix
syms phi th psi %create sybolic variables for phi, theta, and psi
Rz_phi = [...
    cos(phi)    sin(phi)    0;...
    -sin(phi)   cos(phi)    0;...
    0           0           1]; %from XYZ to X'Y'Z'

Rx_th = [...
    1       0           0;...
    0       cos(th)    sin(th);...
    0       -sin(th)   cos(th)]; %from X'Y'Z' to X"Y"Z"

Rz_psi = [...
    cos(psi)    sin(psi)    0;...
    -sin(psi)   cos(psi)    0;...
    0           0           1]; %from X"Y"Z" to xyz

R = Rz_psi*Rx_th*Rz_phi %find overall rotation matrix

%Evaluate R at given angles
R_eval = eval(subs(R,[phi, th, psi], [pi/6, pi/4, -pi/3]))

%Find coordinates in XYZ of point r = -5i + 3j in xyz
r_xyz = [-5; 3; 0];
r_XYZ = R_eval.'*r_xyz

%% Problem 2
RV = calc_R('z', 40);
RE = calc_R('x', 20);
RN = calc_R('y', 10);

R = RN*RE*RV;
a_env = [0; 0.5; -2]; %all times g
a_ENV = R.'*a_env

%% Problem 3
X_prime = [1 0 0];
x = [-50 20 0];
phi = acos(dot(X_prime,x)/(norm(X_prime)*norm(x)))*180/pi;
R_Zprime = calc_R('z', phi);
r_B_A = [-50 20 0];
r_C_A = [-50 0 40];
ABC_perp_p = cross(r_B_A, r_C_A);
ABC_perp_pp = R_Zprime*ABC_perp_p.';
Z_pp = [0;0;1];
theta = acos(dot(ABC_perp_pp,Z_pp)/(norm(ABC_perp_pp)*norm(Z_pp)))*180/pi;
R_Xpp = calc_R('x',theta);
R = R_Xpp*R_Zprime
r_O_A = R*[-50;0;0]

%% Problem 4
R_Z = calc_R('z',40);  
theta = acos((cosd(43.96))/cosd(40))*180/pi;
R_Yp = calc_R('y', theta);
R = R_Yp*R_Z

%% Problem 5
R1 = calc_R('z', atan(3/2)*180/pi);
R2 = calc_R('y', 45);
R3 = R1.';
r_CA_IV = [0;2;1];
r_CA = R1'*R2'*R3'*r_CA_IV;
r_A = [3;0;0];
r_C = r_A + r_CA
