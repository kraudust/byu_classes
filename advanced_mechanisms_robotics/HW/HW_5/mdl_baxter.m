% Generate Robotics-Toolbox SerialLink Model for Baxter Robot
% Standard Denavit-Hartenberg Notation
% 
% Input:
% mode
%   'sim' for baxter geometry of baxter from baxter gazebo simulator model
%   'real' for geometry of real robot (different shoulder position due to
%   manufactoring tolerances
% 
% Sources
% [1] Denavit-Hartenberg Model from Marc Killpack, Brigham Young University
% [2] baxter.urdf.xml, Rethink Robotics, Baxter Simulator

% Moritz Schappler, schappler@irt.uni-hannover.de, 2014-07
% (c) Institut f�r Regelungstechnik, Universit�t Hannover

function [Baxter_l, Baxter_r] = mdl_baxter(mode)

%% Geometry

h = 0.229525; %distance from last frame to end effector along z-axis
% 0.115975+0.11355 = 0.229525


% defining right arm using dh parameters [joint angle (q), d, a, alpha, 0 or 1 for revolute or prismatic, joint angle offset]
% because the sign on joints is the same for right and left arm joint variables,
% only the base transformation shown below differentiates the left and right
% arms
Lr(1) = Link ([0, 0.27035, 0.069, -pi/2, 0, 0], 'standard'); % -> left_s1
Lr(2) = Link ([0, 0, 0, pi/2, 0, pi/2], 'standard'); % left_e0
Lr(3) = Link ([0, 0.36442, 0.0690, -pi/2, 0, 0], 'standard'); % left_e1; 0.26242+0.102=0.36442
Lr(4) = Link ([0, 0, 0, pi/2, 0, 0], 'standard');
Lr(5) = Link ([0, 0.37429, 0.010, -pi/2, 0, 0], 'standard'); % 0.2707+0.10359=0.37429
Lr(6) = Link ([0, 0, 0, pi/2, 0, 0], 'standard');
Lr(7) = Link ([0, h, 0, 0, 0, 0], 'standard');



%% Dynamic Parameters (Rigid Body Model)
% The dynamic parameters are extracted from baxter.urdf.xml file from the
% baxter simulator. Inertias and centers of gravity have to be adapted,
% because the coordinate frames are different between gazebo and the
% robotics toolbox (Denavit-Hartenberg)

% (1) right_upper_shoulder
Lr(1).m = 5.70044;
Lr(1).I = [
   0.047091022620000  -0.006148700390000   0.000127875560000;
  -0.006148700390000   0.035959884780000  -0.000780868990000;
   0.000127875560000  -0.000780868990000   0.037669764550000
   ];
Lr(1).r = [ -0.05117   0.07908    0.00086];

% (2) right_lower_shoulder
Lr(2).m = 3.22698;
Lr(2).I = [
   0.027885975200000  -0.000188219930000  -0.000300963980000
  -0.000188219930000   0.020787492980000   0.002076757620000
  -0.000300963980000   0.002076757620000   0.011752094190000
   ];
Lr(2).r = [ 0.00269   -0.00529   0.06845];

% (3) right_upper_elbow
Lr(3).m = 4.31272;
Lr(3).I = [
   0.026617335570000  -0.003921898870000   0.000292706340000
  -0.003921898870000   0.012480083220000  -0.001083893300000
   0.000292706340000  -0.001083893300000   0.028443552070000
   ];
Lr(3).r = [-0.07176    0.08156  0.00132];

% (4) right_lower_elbow
Lr(4).m = 2.07206;
Lr(4).I = [
   0.013182278760000  -0.000196634180000   0.000360361730000
  -0.000196634180000   0.009268520640000   0.000745949600000
   0.000360361730000   0.000745949600000   0.007115826860000
   ];
Lr(4).r = [0.00159   -0.01117   0.02611];

% (5) right_upper_forearm
Lr(5).m = 2.24665;
Lr(5).I = [
   0.016677428250000  -0.000186576290000   0.000184037050000
  -0.000186576290000   0.003746311500000   0.000647323520000
   0.000184037050000   0.000647323520000   0.016754572640000
   ];
Lr(5).r = [-0.01168    0.13118   0.0046];

% (6) right_lower_forearm
Lr(6).m = 1.60979;
Lr(6).I = [
   0.007005379140000   0.000153480670000  -0.000443847840000
   0.000153480670000   0.005527552400000  -0.000211150380000
  -0.000443847840000  -0.000211150380000   0.003876071520000
   ];

Lr(6).r = [0.00697   0.0060     0.06041];
% (7) right_wrist
m_w = 0.35093;
% (7) right_hand (append)
m_h = 0.19125;
% (7) combined
m_7 = m_w+m_h;

Lr(7).m = m_7;
Lr(7).I = [
   0.001214774578194   0.000008143398948  -0.000065507808991
   0.000008143398948   0.001276357346664   0.000004715353236
  -0.000065507808991   0.000004715353236   0.000554866773537
   ];
Lr(7).r = [0.005137046552805   0.000957223615773  -0.066893467114243];

%% Drive Train
% Gear Ratio
% has to be set to zero at least, so the dynamic-functions of robotics
% toolbox work
Lr(1).G =  0;
Lr(2).G =  0;
Lr(3).G =  0;
Lr(4).G =  0;
Lr(5).G =  0;
Lr(6).G =  0;
Lr(7).G =  0;

% Motor Inertia / [kg*m^2]
Lr(1).Jm =  0;
Lr(2).Jm =  0;
Lr(3).Jm =  0;
Lr(4).Jm =  0;
Lr(5).Jm =  0;
Lr(6).Jm =  0;
Lr(7).Jm =  0;

%% Save in Robot structure
Ll = Lr;
if strcmp(mode, 'sim')
    % simulated robot is the same everywhere
    fprintf('Loaded Baxter Model in Simulation Mode (urdf-data)\n');
    % left_torso_arm_mount
    T_tamL = transl(0.024645, 0.219645, 0.118588)* ... 
        trotz(pi/4); 
    % right_torso_arm_mount
    T_tamR = transl(0.024645, -0.219645, 0.118588)*trotz(-pi/4);
    T_tool = transl([0;0;0.025]);
else
    % these parameters are different for each baxter robot due to a
    % calibration at the end of the manufactoring process
    
    % Extracted from `$ rosrun baxter_pykdl display_urdf.py`
    fprintf('Loaded Baxter Model in Reality Mode (calibrated data)\n');
    T_tamL = transl([0.0256429, 0.219376, 0.107957]) * ...
        rpy2tr([-0.0033612, -0.00248605, 0.77988]);
    % torso -> right_arm_mount
    T_tamR = transl([0.024576, -0.219834, 0.108643]) * ...
        rpy2tr([-0.00243505, -0.00226129, -0.783992]);
    T_tool = transl([0.0, 0.0, 0.045]);
end

% Set up SerialLink class
Baxter_l = SerialLink(Ll(1:7), 'name', 'Baxter_L', 'base' , ...
    T_tamL * ... % left_torso_arm_mount
    transl(0.055695, 0, 0.011038), ... % left_s0
    'tool', T_tool); % hand -> gripper

Baxter_r = SerialLink(Lr(1:7), 'name', 'Baxter_R', 'base' , ...
    T_tamR * ...% right_torso_arm_mount
    transl(0.055695, 0, 0.011038), ...
    'tool', T_tool);