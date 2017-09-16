function statesdot = object_deriv(t, x, I, m, T, F, d, K, u_eq, model_type, A, B, L, C)
%OBJECT_DERIV Equations of motion for an object in 3D space with forces and
%torques applied at given locations
%INPUTS:    I - 3x3 inertia matrix
%           m - mass of object
%           T = torques applied to object T = [T1x, T1y, T1z; T2x, T2y, T2z; ...]
%           F - forces applied to object F = [F1x, F1y, F1z; F2x, F2y, F2z; ...]
%           d = vector from com to position force is applied d = [d1x, d1y, d1z; d2x, d2y, d2z; ...]
%           K = 3x3 matrix from LQR
%           u_eq = equillibrium force and torque
%           model_type = l to use linear model, n for non-linear model
%           A & B - linearized A and B matrices xdot = Ax + Bu
%Find Force and torque from LQR gain K
t
if size(K,1) == 0
    T = T;
    F = F;
else
xh = x(1:12); 
x = x(13:24);
u_input = (-K*xh)';
F = u_input(4:6) + u_eq(4:6)';
T = u_input(1:3) + u_eq(1:3)';
end

wx = xh(1); %angular velocity about body fixed x axis
wy = xh(2); %angular velocity about body fixed y axis
wz = xh(3); %angular velocity about body fixed z axis
psi = x(4); %rotation angle about inertial z axis
th = x(5); %rotation angle about y' axis
phi = x(6); %rotation angle about x'' axis
px = x(7); %x position of center of mass relative to and in terms of inertial frame
py = x(8); %y position of center of mass relative to and in terms of inertial frame
pz = x(9); %z position of center of mass relative to and in terms of inertial frame
Vx = xh(10); %x velocity of center of mass in terms of body fixed axes
Vy = xh(11); %y velocity of center of mass in terms of body fixed axes
Vz = xh(12); %z velocity of center of mass in terms of body fixed axes

Ixy = -I(1,2);
Ixz = -I(1,3);
Iyz = -I(2,3);
Ixx = I(1,1);
Iyy = I(2,2);
Izz = I(3,3);
g = 9.81; %gravity

if model_type == 'l'
    xdot = A*x + B*u_input';
    xhdot = (A-L*C)*xh +B*u_input' + L*C*[wx;wy;wz;psi;th;phi;px;py;pz;Vx;Vy;Vz];
    statesdot = [xhdot;xdot];
else

% Angular Velocity
num_force = size(F,1);
num_torque = size(T,1);
Tx = 0;
Ty = 0;
Tz = 0;
for i = 1:num_torque
    Tx = Tx + T(i,1);
    Ty = Ty + T(i,2);
    Tz = Tz + T(i,3);
end
for j = 1:num_force
    Tx = Tx - F(j,2)*d(j,3) + F(j,3)*d(j,2);
    Ty = Ty + F(j,1)*d(j,3) - F(j,3)*d(j,1);
    Tz = Tz - F(j,1)*d(j,2) + F(j,2)*d(j,1);
end
F_ang_acc = [...
    Tx - Ixy*wx*wz + Ixz*wx*wy + (Iyy - Izz)*wy*wz + Iyz*(wy^2 - wz^2);...
    Ty - Iyz*wx*wy + Ixy*wy*wz + (Izz-Ixx)*wx*wz + Ixz*(wz^2 - wx^2);...
    Tz - Ixz*wy*wz + Iyz*wx*wz + (Ixx-Iyy)*wx*wy + Ixy*(wx^2 - wy^2)];
    
ang_accel = I\F_ang_acc;
wxdot = ang_accel(1);
wydot = ang_accel(2);
wzdot = ang_accel(3);

% Angular Position
psidot = (1/cos(th))*(wy*sin(phi) + wz*cos(phi));
thdot = wy*cos(phi) - wz*sin(phi);
phidot = wx + (1/cos(th))*(wy*sin(phi)*sin(th) + wz*cos(phi)*sin(th));

% Linear Position
pxdot = Vx*cos(psi)*cos(th) - Vy*(cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(th)) + Vz*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(th));
pydot = Vx*sin(psi)*cos(th) + Vy*(cos(phi)*cos(psi) + sin(psi)*sin(th)*sin(phi)) - Vz*(cos(psi)*sin(phi) - sin(psi)*sin(th)*cos(phi));
pzdot = -Vx*sin(th) + Vy*cos(th)*sin(phi) + Vz*cos(th)*cos(phi);

%Linear Velocity
Fx = 0;
Fy = 0;
Fz = 0;
for k = 1:num_force
   Fx = Fx + F(k,1);
   Fy = Fy + F(k,2);
   Fz = Fz + F(k,3);
end

Vxdot = (1/m)*Fx + g*sin(th) - Vz*wy + Vy*wz;
Vydot = (1/m)*Fy - g*cos(th)*sin(phi) + Vz*wx - Vx*wz;
Vzdot = (1/m)*Fz - g*cos(th)*cos(phi) - Vy*wx + Vx*wy;

xdot = [wxdot; wydot; wzdot; psidot; thdot; phidot; pxdot; pydot; pzdot; Vxdot; Vydot; Vzdot];
xhdot = (A-L*C)*xh +B*u_input' + L*C*x;
statesdot = [xhdot;xdot];
end
end

