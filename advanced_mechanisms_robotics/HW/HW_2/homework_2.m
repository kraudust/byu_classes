%Run Robotics Toolbox
run ~/Desktop/rvctools/startup_rvc.m

%% Problem 3-6
%I did this part before I wrote the function calc_A
syms th1 th2 th3 l1 l2 l3
A1 = [cos(th1)  0   sin(th1)    0;...
    sin(th1)    0   -cos(th1)   0;...
    0           1       0       l1;...
    0           0       0       1];
A2 = [cos(th2)  -sin(th2)   0   l2*cos(th2);...
    sin(th2)    cos(th2)    0   l2*sin(th2);...
    0           0           1       0;...
    0           0           0       1];
A3 = [cos(th3)  -sin(th3)   0   l3*cos(th3);...
    sin(th3)    cos(th3)    0   l3*sin(th3);...
    0           0           1       0;...
    0           0           0       1];
T_3_0 = A1*A2*A3;

%% Problem 3-8 or 1(g)
syms th1 th2 th3 th4 th5 th6 l1 l2 l3 l4 l5 l6
A1 = calc_A(th1, l1, 0, 90);
A2 = calc_A(th2, 0, l2, 0);
A3 = calc_A(th3+(pi/2),0,0,90);
A4 = calc_A(th4 + (pi/2), (l3+l4),0,90);
A5 = calc_A(th5,0,0,90);
A6 = calc_A(th6,(-l5-l6),0,180);

T6_0 = A1*A2*A3*A4*A5*A6;

%% 1(h)
%Select link lengths for the robot in 3-8. Find a way to determine
%   the reachable workspace using forward kinematics
clear all
close all

% Create Robot
L1 = Link('d',1,'a',0,'alpha',pi/2);
L2 = Link('d',0,'a',1,'alpha',0);
L3 = Link('d',0,'a',0,'alpha',pi/2,'offset',pi/2);
L4 = Link('d',2,'a',0,'alpha',pi/2,'offset',pi/2);
L5 = Link('d',0,'a',0,'alpha',pi/2);
L6 = Link('d',-2,'a',0,'alpha',pi);

bot = SerialLink([L1 L2 L3 L4 L5 L6], 'name', 'Dustan');
%position of the origin of the end effector in the end effector frame
end_eff_pos = [0;0;0;1]; 
inc = 0.25;
pos = [];
%joint 1
for i = -pi/6:inc:pi/6,
    %joint 2
    for j = -pi/6:inc:pi/6,
        %joint 3
        for k = -pi/6:inc:pi/6,
            %joint 4
            for l = -pi/6:inc:pi/6,
                %joint 5
                for m = -pi/6:inc:pi/6,
                    H = bot.fkine([i,j,k,l,m,0]);
                    pos(:,end+1) = H*end_eff_pos;
                end
            end
        end
    end
end
figure()
plot3(pos(1,:),pos(2,:),pos(3,:),'*');
hold on
bot.plot([0 0 0 0 0 0])
view(-45,30)
xlim([-6, 6])
ylim([-6,6])

figure()
plot3(pos(1,:),pos(2,:),pos(3,:),'*');
hold on
bot.plot([0 0 0 0 0 0])
view(45,30)
xlim([-6, 6])
ylim([-6,6])
