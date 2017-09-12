function [wx, wy, wz, psi, th, phi] = mielke_kraus(wx0, wy0, wz0, psi0, theta0, phi0, t, Mx, My, Mz)
%MIELKE_KRAUS Simulates Apollo motion
%   Inputs: units are SI, and degrees!!
%   Outputs: units are degrees and degrees/sec

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

%% Task C
% Part a)
%Ib = Ib_s; %comment this line out for the general case
Ixx = Ib(1,1);
Iyy = Ib(2,2);
Izz = Ib(3,3);
Ixy = -Ib(1,2);
Ixz = -Ib(1,3);
Iyz = -Ib(2,3);
M = [Ib, zeros(3); zeros(3), eye(3)];

function xdot = apollo_deriv(time, x)
    wx = x(1);
    wy = x(2);
    wz = x(3);
    psi = x(4);
    th = x(5);
    phi = x(6);
    
%     index = max(find(t <= time)); %find which torque values to use
%     F = [-Ixy*wx*wz + Ixz*wx*wy + (Iyy - Izz)*wy*wz + Iyz*(wy^2 - wz^2) + Mx(index);...
%         -Iyz*wx*wy + Ixy*wy*wz + (Izz - Ixx)*wx*wz + Ixz*(wz^2 - wx^2) + My(index);...
%         -Ixz*wy*wz + Iyz*wx*wz + (Ixx - Iyy)*wx*wy + Ixy*(wx^2 - wy^2) + Mz(index);...
%         (1/cos(th))*(wy*sin(phi) + wz*cos(phi));...
%         wy*cos(phi) - wz*sin(phi);...
%         (1/cos(phi))*(wy*sin(th)*sin(phi) + wz*sin(th)*cos(phi)) + wx];
    F = [-Ixy*wx*wz + Ixz*wx*wy + (Iyy - Izz)*wy*wz + Iyz*(wy^2 - wz^2) + interp1(t,Mx,time);...
        -Iyz*wx*wy + Ixy*wy*wz + (Izz - Ixx)*wx*wz + Ixz*(wz^2 - wx^2) + interp1(t,My,time);...
        -Ixz*wy*wz + Iyz*wx*wz + (Ixx - Iyy)*wx*wy + Ixy*(wx^2 - wy^2) + interp1(t,Mz,time);...
        (1/cos(th))*(wy*sin(phi) + wz*cos(phi));...
        wy*cos(phi) - wz*sin(phi);...
        (1/cos(phi))*(wy*sin(th)*sin(phi) + wz*sin(th)*cos(phi)) + wx];
    xdot = M\F;
end

%Convert initial conditions to radians not degrees
wx0 = wx0*pi/180;
wy0 = wy0*pi/180;
wz0 = wz0*pi/180;
psi0 = psi0*pi/180;
theta0 = theta0*pi/180;
phi0 = phi0*pi/180;

%Simulate the system
[t, x] = ode45(@apollo_deriv,t,[wx0; wy0; wz0; psi0; theta0; phi0]);

%Convert back to degrees
rad2deg = 180/(pi);
wx = x(:, 1)*rad2deg;
wy = x(:, 2)*rad2deg;
wz = x(:, 3)*rad2deg;
psi = x(:, 4)*rad2deg;
th = x(:, 5)*rad2deg;
phi = x(:, 6)*rad2deg;
end

