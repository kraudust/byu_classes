clear

L = 0.4;
m = 1;
Ixx = 0.005;
Iyy = 0.005;
Izz = 0.01;
I = [Ixx 0 0; 0 Iyy 0; 0 0 Izz];
grav = [0 0 9.81]';
q = [pi/4 pi/4 pi/4]';
q_dot = [pi/6 -pi/4 pi/3]';
q_ddot = [-pi/6 pi/3 pi/6]';

L1 = Link('d',0,'a',L,'alpha',0);

L2(1) = Link('d',0,'a',L,'alpha',0);
L2(2) = Link('d',0,'a',L,'alpha',0);

L3(1) = Link('d',0,'a',L,'alpha',0);
L3(2) = Link('d',0,'a',L,'alpha',0);
L3(3) = Link('d',0,'a',L,'alpha',0);

L3(1).m = m;
L3(2).m = m;
L3(3).m = m;

L3(1).I = I;
L3(2).I = I;
L3(3).I = I;

L3(1).r = [-L/2 0 0];
L3(2).r = [-L/2 0 0];
L3(3).r = [-L/2 0 0];

bot1 = SerialLink(L1, 'name','robot 1');
bot2 = SerialLink(L2, 'name','robot 2');
bot3 = SerialLink(L3, 'name','robot 3');
bot  = [bot1 bot2 bot3];

bot3.nofriction('all');
 
%R_minusi_i = [1 0 0; 0 1 0; 0 0 1];
omega = [0 0 0]';
alpha = [0 0 0]';
z_minusi_0 = [0 0 1]';
a_minus_e = [0 0 0]'; 
for i = 1:3
T = bot(i).fkine(q(1:i));
R_0_i = T(1:3,1:3)';
%R_minusi_i = R_prev'*R_0_i;
R_minusi_i = [cos(q(i)) sin(q(i)) 0;
              -sin(q(i)) cos(q(i)) 0;
              0         0        1];
omega_i(:,i) = R_minusi_i*omega + R_0_i*z_minusi_0*q_dot(i);
alpha_i(:,i) = R_minusi_i*alpha + R_0_i*z_minusi_0*q_ddot(i) + cross(omega_i(:,i),(R_0_i*z_minusi_0*q_dot(i)));
a_i(:,i) = R_minusi_i*a_minus_e + cross(alpha_i(:,i),[L/2 0 0])' + cross(omega_i(:,i),(cross(omega_i(:,i),[L/2 0 0])))';
a_e(:,i) = R_minusi_i*a_minus_e + cross(alpha_i(:,i),[L 0 0])' + cross(omega_i(:,i),(cross(omega_i(:,i),[L 0 0])))';
a_minus_e = a_e(:,i);
alpha = alpha_i(:,i);
omega = omega_i(:,i);
%R_prev = R_minusi_i;
end

j = 3;
f = [0 0 0]';
tau = [0 0 0]';
R_plusi_i = [1 0 0; 0 1 0; 0 0 1];
while j>0
    f_j(:,j) = R_plusi_i*f + m*a_i(:,j);
    tau_j(:,j) = R_plusi_i*tau - cross(f_j(:,j),[L/2 0 0])' + cross(R_plusi_i*f_j(:,j),[-L/2 0 0])' + I*alpha_i(:,j) + cross(omega_i(:,j),(I*omega_i(:,j)));
    j = j-1;
end

tau_newton_euler = tau_j
tau_comp = bot3.rne(q',q_dot',q_ddot')
