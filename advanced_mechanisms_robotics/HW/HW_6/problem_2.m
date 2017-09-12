clear all
clc;
% School Linux
%run ~/Desktop/rvctools/startup_rvc.m
% Personal laptop
%run C:\Users\Dustan\Desktop\rvctools\startup_rvc.m

%Define Robot
L = 0.4; %link length
m = 1; %link mass
Izz = 0.01; %rotational inertia
grav = [0; 0; 9.81];
I = [Izz, 0, 0; 0, Izz, 0; 0, 0, Izz];

Lk1 = Link('revolute', 'offset', 0, 'd', 0, 'a', L, 'alpha', 0);

Lk2(1) = Link('revolute', 'offset', 0, 'd', 0, 'a', L, 'alpha', 0);
Lk2(2) = Link('revolute', 'offset', 0, 'd', 0, 'a', L, 'alpha', 0);

Lk3(1) = Link('revolute', 'offset', 0, 'd', 0, 'a', L, 'alpha', 0);
Lk3(2) = Link('revolute', 'offset', 0, 'd', 0, 'a', L, 'alpha', 0);
Lk3(3) = Link('revolute', 'offset', 0, 'd', 0, 'a', L, 'alpha', 0);

Lk3(1).m = m;
Lk3(1).I = I;
Lk3(1).r = [L/2, 0, 0];

Lk3(2).m = m;
Lk3(2).I = I;
Lk3(2).r = [L/2, 0, 0];

Lk3(3).m = m;
Lk3(3).I = I;
Lk3(3).r = [L/2, 0, 0];

bot1 = SerialLink(Lk1, 'name', '1_link');
bot2 = SerialLink(Lk2, 'name', '2_link');
bot3 = SerialLink(Lk3, 'name', '3_link');

bot = [bot1, bot2, bot3];

bot3.nofriction('all');

q = [pi/4; pi/4; pi/4];
qd = [pi/6; -pi/4; pi/3];
qdd = [-pi/6; pi/3; pi/6];
bot3.plot(q.')
view([0 90])

%Find w,alpha, and a
w_iminus1 = [0; 0; 0]; %initial omega
al_iminus1 = [0; 0; 0]; %inital alpha
ae_iminus1 = [0; 0; 0]; %initial a_e
z_iminus1_0 = [0; 0; 1];

w = zeros(3);
al = zeros(3);
ac = zeros(3);
for i = 1:3,
    T = bot(i).fkine(q(1:i));
    R_0_i = T(1:3,1:3)'; %rotation from frame 0 to i
    R_iminus1_i = [...
        cos(q(i))   sin(q(i))   0;...
        -sin(q(i))  cos(q(i))   0;...
        0           0           1]; %rotation from frame i-1 to i
    w_i = R_iminus1_i*w_iminus1 + R_0_i*z_iminus1_0*qd(i);
    w(i,:) = w_i.';
    al_i = R_iminus1_i*al_iminus1 + R_0_i*z_iminus1_0*qdd(i) +...
        cross(w_i, R_0_i*z_iminus1_0*qd(i));
    al(i,:) = al_i.';
    ae_i = R_iminus1_i*ae_iminus1 + cross(al_i,[L;0;0]) +...
        cross(w_i,cross(w_i,[L;0;0]));
    ac_i = R_iminus1_i*ae_iminus1 + cross(al_i,[L/2;0;0]) +...
        cross(w_i,cross(w_i,[L/2;0;0]));
    ac(i,:) = ac_i';
    w_iminus1 = w_i;
    al_iminus1 = al_i;
    ae_iminus1 = ae_i;
end

f_i_pl_1 = [0; 0; 0];
tau_i_pl_1 = [0; 0; 0];
j = 3;

while j>0,
    R_i_pl_1_i = [...
        cos(q(j))   sin(q(j))   0;...
        -sin(q(j))  cos(q(j))   0;...
        0           0           1].'; %rotation from frame i+1 to i
    f_i = R_i_pl_1_i*f_i_pl_1 + m*ac(j,:).' - m*grav;
    tau_i = R_i_pl_1_i*tau_i_pl_1 - cross(f_i,[L/2;0;0]) +...
        cross(R_i_pl_1_i*f_i_pl_1,[-L/2;0;0]) + I*al(j,:).' + ...
        cross(w(j,:).',I*w(j,:).');
    j = j - 1;
end
tau_i
tau_toolbox = bot3.rne(q.', qd.', qdd.')
%They don't match

