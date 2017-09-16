%% Problem 1
clc
clear 
close 

Ixx_com = (4^2 + 5^2)/12; %not including mass
Iyy_com = (5^2 + 7^2)/12; %not including mass
Izz_com = (4^2 + 7^2)/12; %not including mass
Ig = [Ixx_com 0 0; 0 Iyy_com 0; 0 0 Izz_com]; %not including mass
d = [-7/2; -2; -5/2];
Io = Ig + (d.'*d*eye(3) - d*d.'); %not including mass
th1 = atan(5/7);
th2 = atan(4/sqrt(25+49));
Rz = [cos(th2) sin(th2) 0; -sin(th2) cos(th2) 0; 0 0 1];
Ry = [cos(-th1) 0 -sin(-th1); 0 1 0; sin(-th1) 0 cos(-th1)];
R = Rz*Ry;
Io_pp = R*Io*R.' % Multiply this by mass to have final answer

%% Problem 2
clear 
close 

m = 0.5; % mass of each rod in kg
L = 0.72; % length of each rod in m

IG1 = ((m*L^2)/12)*[0 0 0; 0 1 0; 0 0 1];
IG2 = ((m*L^2)/12)*[1 0 0; 0 1 0; 0 0 0];
IG3 = IG1;
IG4 = ((m*L^2)/12)*[1 0 0; 0 0 0; 0 0 1];
IG5 = IG2;
IG6 = IG4;

IG = {IG1, IG2, IG3, IG4, IG5, IG6}; %create cell array containing IG's

d1 = [-L/2;     0;      -L];
d2 = [-L;       0;      -L/2];
d3 = [-L/2;     0;      0];
d4 = [0;        -L/2;   0];
d5 = [0;        -L;     L/2];
d6 = [0;        -L/2;   L];

d = {d1, d2, d3, d4, d5, d6}; %create cell array containing d's

Ioi = cell(1,6); %initialize cell array to hold the Io's

%Calculate each Io
for i = 1:6
    Ioi{i} = IG{i} + m*(d{i}.'*d{i}*eye(3) - d{i}*d{i}.');
end

%Sum of individual Io's to get total Io
Io = zeros(3);
for i = 1:6
   Io = Io + Ioi{i};
end

Io

%% Problem 4
clear
close

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% From Problem 1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ixx_com = (4^2 + 5^2)/12; %not including mass
Iyy_com = (5^2 + 7^2)/12; %not including mass
Izz_com = (4^2 + 7^2)/12; %not including mass
Ig = [Ixx_com 0 0; 0 Iyy_com 0; 0 0 Izz_com]; %not including mass
d = [-7/2; -2; -5/2];
Io = Ig + (d.'*d*eye(3) - d*d.'); %not including mass
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%V is eigenvectors in columns and D is diagonal matrix of eigenvalues
[V, D] = eig(Io);
R = V.';
check = det(R); %should be 1, if it's -1 it's a LH coordinate system
R(3,:) = -1*R(3,:); %convert to a RH coordinate system since check = -1
R*Io*R'; %just to make sure it works
D
R


%% Problem 5
a = 4;
b = 3;
c = 5;
% all need to be multiplied by m
Ixx = (a^2)/6 + (c^2)/3; 
Iyy = (a^2)/6 + (b^2)/6; 
Izz = (b^2)/6 + (c^2)/3; 
Ixy = b*c/6;
Ixz = a*b/12;
Iyz = a*c/6;
Io = [   Ixx   -Ixy    -Ixz; ...
        -Ixy    Iyy    -Iyz; ...
        -Ixz   -Iyz     Izz];
[V, D] = eig(Io);
R = V.';
det(R); %this is negative 1, so switch sign of third row
R(3,:) = -1*R(3,:);
D
R



