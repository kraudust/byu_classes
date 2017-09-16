clear
close all
clc


%% Task A
% We did part a in mielke_kraus.m
%% Task A
% Part a)------------------------------------------------------------------
%   Part aa)
    conv_in = 0.0254; %gives us m from inches
    conv_lb = 4.44822/9.81; %gives us kg from lb-force
    conv_slug_ft2 = 1.35581795; %gives us kg*m^2 from slug*ft^2
    mCM = 9730*conv_lb; %mass of command module in kg
    mSM = 9690*conv_lb; %mass of service module in kg
    mPr = 37295*conv_lb; %mass of propellant in kg
    tot_mass = mCM + mSM + mPr; %total mass in kg
    pCM = [1043.1; -0.1; 7.8]*conv_in; %com of command module rel to A in m
    pSM = [908.2; 0.7; -0.6]*conv_in; %com of service module rel to A in m
    pPr = [905.9; 5.6; -2.4]*conv_in; %com of propellant rel to A in m
    
    %Calculate com of the CSM including propellant in meters relative to A
    P_cm_CSM = [...
        (pCM(1)*mCM + pSM(1)*mSM + pPr(1)*mPr)/tot_mass;...
        (pCM(2)*mCM + pSM(2)*mSM + pPr(2)*mPr)/tot_mass;...
        (pCM(3)*mCM + pSM(3)*mSM + pPr(3)*mPr)/tot_mass];
        
%   Part ab)
    dCM = pCM - P_cm_CSM;
    dSM = pSM - P_cm_CSM;
    dPr = pPr - P_cm_CSM;
    
    %Inertia Matrices of each part around their c.o.m.
    ICM = [4474, 0, 0; 0, 3919, 0; 0, 0, 3684]*conv_slug_ft2; 
    ISM = [6222, 0, 0; 0, 10321, 0; 0, 0, 10136]*conv_slug_ft2;
    IPr = [19162, 0, 0; 0, 19872, 0; 0, 0, 26398]*conv_slug_ft2;
    
    %Parallel Axis Theorem
    Ib_CM = ICM + mCM*(dCM.'*dCM*eye(3) - dCM*dCM.');
    Ib_SM = ISM + mSM*(dSM.'*dSM*eye(3) - dSM*dSM.');
    Ib_Pr = IPr + mPr*(dPr.'*dPr*eye(3) - dPr*dPr.');
    
    %Total Inertia Matrix about B axis
    Ib = Ib_CM + Ib_SM + Ib_Pr;
    
% Part b)------------------------------------------------------------------
%   Part ba)
    %Location of center of mass of CSM in meters relative to A frame
    P_cm_CSM_s = [...
        (pCM(1)*mCM + pSM(1)*mSM + pPr(1)*mPr)/tot_mass;...
        0;...
        0];
%   Part bb)
    dCM_s = [pCM(1); 0; 0] - P_cm_CSM_s;
    dSM_s = [pSM(1); 0; 0] - P_cm_CSM_s;
    dPr_s = [pPr(1); 0; 0] - P_cm_CSM_s;
    
    Ib_CM_s = ICM + mCM*(dCM_s.'*dCM_s*eye(3) - dCM_s*dCM_s.');
    Ib_SM_s = ISM + mSM*(dSM_s.'*dSM_s*eye(3) - dSM_s*dSM_s.');
    Ib_Pr_s = IPr + mPr*(dPr_s.'*dPr_s*eye(3) - dPr_s*dPr_s.');
    
    Ib_s = Ib_CM_s + Ib_SM_s + Ib_Pr_s;
%% Task B
% We derived the equations of motion for the general case
    
%% Task C
%Part b)
t = 0:0.001:30; %time vector for simulation
Mx = 176*cos(0.2*t); %applied x torque N-m
My = 54*ones(length(t),1); %applied y torque N-m
Mz = 98*sin(0.3*t); %applied z torque N-m
wx0 = 0; %initial angular velocity about body x axis (deg/s)
wy0 = 0; %initial angular velocity about body y axis (deg/s)
wz0 = 0; %initial angular velocity about body z axis (deg/s)
psi0 = 0; %original yaw angle (deg)
th0 = 0; %original pitch angle (deg)
phi0 = 0; %original roll angle (deg)

[wx, wy, wz, psi, theta, phi] = mielke_kraus(wx0, wy0, wz0, psi0, th0, phi0, t, Mx, My, Mz);


figure()
subplot(3,2,6)
suptitle('General Case: Applied Torques')
plot(t,wx)
ylabel('w_x (deg/s)')
xlabel('Time (sec)')
subplot(3,2,4)
plot(t,wy)
ylabel('w_y (deg/s)')
subplot(3,2,2)
plot(t,wz)
ylabel('w_z (deg/s)')
subplot(3,2,1)
plot(t,psi)
ylabel('psi (deg)')
subplot(3,2,3)
plot(t,theta)
ylabel('theta (deg)')
subplot(3,2,5)
plot(t,phi)
ylabel('phi (deg)')
xlabel('Time (sec)')

n = 10;
tic
animate_CSM(t,phi,theta,psi, n)
toc

Rows = {'Min';'Max'};
Wx = [min(wx); max(wx)];
Wy = [min(wy); max(wy)];
Wz = [min(wz); max(wz)];
Psi = [min(psi); max(psi)];
Theta = [min(theta); max(theta)];
Phi = [min(phi); max(phi)];
table(Wx, Wy, Wz, Psi, Theta, Phi, 'RowNames',Rows)

%Part c)
t = 0:0.01:1000; %time vector for simulation
wx0 = 1; %initial angular velocity about body x axis (deg/s)
wy0 = 0; %initial angular velocity about body y axis (deg/s)
wz0 = 0; %initial angular velocity about body z axis (deg/s)
psi0 = 0; %original yaw angle (deg)
th0 = 0; %original pitch angle (deg)
phi0 = 0; %original roll angle (deg)
Mx = zeros(length(t),1); %applied x torque N-m
My = (-Ib(1,3)*(wx0*pi/180)^2)*ones(length(t),1); %applied y torque N-m
Mz = (Ib(1,2)*(wx0*pi/180)^2)*ones(length(t),1); %applied z torque N-m

[wx, wy, wz, psi, theta, phi] = mielke_kraus(wx0, wy0, wz0, psi0, th0, phi0, t, Mx, My, Mz);

figure()
subplot(3,2,6)
suptitle('Calculated Input Torque Barbeque Mode')
plot(t,wx)
ylabel('w_x (deg/s)')
xlabel('Time (sec)')
subplot(3,2,4)
plot(t,wy)
ylabel('w_y (deg/s)')
subplot(3,2,2)
plot(t,wz)
ylabel('w_z (deg/s)')
subplot(3,2,1)
plot(t,psi)
ylabel('psi (deg)')
subplot(3,2,3)
plot(t,theta)
ylabel('theta (deg)')
subplot(3,2,5)
plot(t,phi)
ylabel('phi (deg)')
xlabel('Time (sec)')

n = 100;
tic
animate_CSM(t,phi,theta,psi, n)
toc

[wx, wy, wz, psi, theta, phi] = mielke_kraus(wx0, wy0, wz0, psi0, th0, phi0, t, 0*Mx, 0*My, 0*Mz);

figure()
subplot(3,2,6)
suptitle('Zero Input Torque Barbeque Mode')
plot(t,wx)
ylabel('w_x (deg/s)')
xlabel('Time (sec)')
subplot(3,2,4)
plot(t,wy)
ylabel('w_y (deg/s)')
subplot(3,2,2)
plot(t,wz)
ylabel('w_z (deg/s)')
subplot(3,2,1)
plot(t,psi)
ylabel('psi (deg)')
subplot(3,2,3)
plot(t,theta)
ylabel('theta (deg)')
subplot(3,2,5)
plot(t,phi)
ylabel('phi (deg)')
xlabel('Time (sec)')

n = 100;
tic
animate_CSM(t,phi,theta,psi, n)
toc

%% Task D
%Part b
Ixx = Ib(1,1);
Iyy = Ib(2,2);
Izz = Ib(3,3);
Ixy = -Ib(1,2);
Ixz = -Ib(1,3);
Iyz = -Ib(2,3);
alpha_x = 5*pi/180; %deg/s
alpha_y = 2.5*pi/180;
alpha_z = 2.5*pi/180;
omega = 6600*360/60*pi/180; %deg/s
th1dot = 30*pi/180;
th2dot = 30*pi/180;
th3dot = 30*pi/180;
% For rotation in x
Irx = max([abs(-Ixx*alpha_x/(omega*th3dot)), abs(Ixy*alpha_x/(omega*th1dot)), abs(Ixz*alpha_x/(omega*th2dot))]);
% For rotation in y
Iry = max([abs(Ixy*alpha_y/(omega*th3dot)), abs(-Iyy*alpha_y/(omega*th1dot)), abs(Iyz*alpha_y/(omega*th2dot))]);
% For roation in z
Irz = max([abs(Ixz*alpha_z/(omega*th3dot)), abs(Iyz*alpha_z/(omega*th1dot)), abs(-Izz*alpha_z/(omega*th2dot))]);

Ir = max([Irx, Iry, Irz]);

%% Task E
%part b
t = 0:0.001:30; %time vector for simulation
wx0 = 0; %initial angular velocity about body x axis (deg/s)
wy0 = 0; %initial angular velocity about body y axis (deg/s)
wz0 = 0; %initial angular velocity about body z axis (deg/s)
psi0 = 0; %original yaw angle (deg)
theta0 = 0; %original pitch angle (deg)
phi0 = 0; %original roll angle (deg)
th1 = 15*sin((2*pi/30)*t);
th2 = zeros(length(t),1);
th3 = zeros(length(t),1);

[wx, wy, wz, psi, theta, phi] = mielke_kraus_cmg(wx0, wy0, wz0, psi0, theta0, phi0, t, th1, th2, th3);

figure()
subplot(3,2,6)
suptitle('Task E Part b')
plot(t,wx)
ylabel('w_x (deg/s)')
xlabel('Time (sec)')
subplot(3,2,4)
plot(t,wy)
ylabel('w_y (deg/s)')
subplot(3,2,2)
plot(t,wz)
ylabel('w_z (deg/s)')
subplot(3,2,1)
plot(t,psi)
ylabel('psi (deg)')
subplot(3,2,3)
plot(t,theta)
ylabel('theta (deg)')
subplot(3,2,5)
plot(t,phi)
ylabel('phi (deg)')
xlabel('Time (sec)')

Rows = {'Min';'Max'};
Wx = [min(wx); max(wx)];
Wy = [min(wy); max(wy)];
Wz = [min(wz); max(wz)];
Psi = [min(psi); max(psi)];
Theta = [min(theta); max(theta)];
Phi = [min(phi); max(phi)];
table(Wx, Wy, Wz, Psi, Theta, Phi, 'RowNames',Rows)

n = 3;
tic
animate_CSM(t,phi,theta,psi, n)
toc

%part c
th1 = 15*(1./(1+exp(-0.3*t)) - 0.5);
th2 = 5*sin((2*pi/30)*t);
th3 = -5*sin((2*pi/30)*t);
[wx, wy, wz, psi, theta, phi] = mielke_kraus_cmg(wx0, wy0, wz0, psi0, theta0, phi0, t, th1, th2, th3);

figure()
subplot(3,2,6)
suptitle('Task E Part c)')
plot(t,wx)
ylabel('w_x (deg/s)')
xlabel('Time (sec)')
subplot(3,2,4)
plot(t,wy)
ylabel('w_y (deg/s)')
subplot(3,2,2)
plot(t,wz)
ylabel('w_z (deg/s)')
subplot(3,2,1)
plot(t,psi)
ylabel('psi (deg)')
subplot(3,2,3)
plot(t,theta)
ylabel('theta (deg)')
subplot(3,2,5)
plot(t,phi)
ylabel('phi (deg)')
xlabel('Time (sec)')
Rows = {'Min';'Max'};

n = 3;
tic
animate_CSM(t,phi,theta,psi, n)
toc

Wx = [min(wx); max(wx)];
Wy = [min(wy); max(wy)];
Wz = [min(wz); max(wz)];
Psi = [min(psi); max(psi)];
Theta = [min(theta); max(theta)];
Phi = [min(phi); max(phi)];
table(Wx, Wy, Wz, Psi, Theta, Phi, 'RowNames',Rows)