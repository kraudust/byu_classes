%% Problem 1(a)
% School Linux
%run ~/Desktop/rvctools/startup_rvc.m
% Personal laptop
%run C:\Users\Dustan\Desktop\rvctools\startup_rvc.m

%define the robotics toolbox Puma 560 arm
mdl_puma560;

%set the Coulomb friction terms to zero to help with numerical simulation
p560 = p560.nofriction;

P.kp = diag([40 40 20 15 15 15])*eye(6);
P.kd = diag([18 14 3 3 3 3])*eye(6);
%P.q_des = (pi/180)*[-30;30;-70;15;25;70];
P.q_des = (pi/180)*[30;60;45;35;-55;-50];

% %% Make Plots
% figure(2)
% subplot(3,2,1)
% hold on
% plot(t_sim,q_sim(:,1)*180/pi,'b')
% plot(t_sim,P.q_des(1)*(180/pi)*ones(1,length(t_sim)),'r')
% ylabel('Joint 1 Angle (rad)')
% legend('x_{act}','x_{des}')
% 
% subplot(3,2,2)
% hold on
% plot(t_sim,q_sim(:,2)*180/pi,'b')
% plot(t_sim,P.q_des(2)*(180/pi)*ones(1,length(t_sim)),'r')
% ylabel('Joint 2 Angle (rad)')
% legend('x_{act}','x_{des}')
% 
% subplot(3,2,3)
% hold on
% plot(t_sim,q_sim(:,3)*180/pi,'b')
% plot(t_sim,P.q_des(3)*(180/pi)*ones(1,length(t_sim)),'r')
% ylabel('Joint 3 Angle (rad)')
% legend('x_{act}','x_{des}')
% 
% subplot(3,2,4)
% hold on
% plot(t_sim,q_sim(:,4)*180/pi,'b')
% plot(t_sim,P.q_des(4)*(180/pi)*ones(1,length(t_sim)),'r')
% ylabel('Joint 4 Angle (rad)')
% legend('x_{act}','x_{des}')
% 
% subplot(3,2,5)
% hold on
% plot(t_sim,q_sim(:,5)*180/pi,'b')
% plot(t_sim,P.q_des(5)*(180/pi)*ones(1,length(t_sim)),'r')
% ylabel('Joint 5 Angle (rad)')
% xlabel('Time (sec)')
% legend('x_{act}','x_{des}')
% 
% subplot(3,2,6)
% hold on
% plot(t_sim,q_sim(:,6)*180/pi,'b')
% plot(t_sim,P.q_des(6)*(180/pi)*ones(1,length(t_sim)),'r')
% ylabel('Joint 6 Angle (rad)')
% xlabel('Time (sec)')
% legend('x_{act}','x_{des}')