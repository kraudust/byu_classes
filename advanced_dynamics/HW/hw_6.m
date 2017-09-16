%% Problem 1
syms m L
Ig = m*L^2*[1/8, 0, 0; 0, 1/8, 0; 0, 0, 1/4];
d = [-L/2; 0; 0];
Io1 = Ig + 2*m*(d.'*d*eye(3) - d*d.');
phi_dd = -6*sind(30)*cosd(30)/(13 + 3*sind(30)^2);

%% Problem 4
clear 
clc
R = calc_R('y', 60);
Ig = [17 0 0; 0 16 0; 0 0 1];
I = R*Ig*R.'


