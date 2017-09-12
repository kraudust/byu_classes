function [xp] = p560_dyn_eqns(t,x,bot,torque,time)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%Dq'' + Cq' + G = Tau
%bot.accel(q',q,Tau) = q''
%x1 = q
%x2 = q'
%xp = [x1'; x2'] = [q'; q'']
x1 = x(1:6);
x2 = x(7:12);
% D = bot.inertia(x1.');
% C = bot.coriolis(x1.',x2.');
% G = bot.gravload(x1.');
% G = G.';
xp = zeros(12,1);
xp(1:6) = x2;
Tau = interp1(time,torque,t);
%Tau = Tau.';
%xp(7:12) = D\(-C*x2 - G + Tau);
xp(7:12) = bot.accel(x1.',x2.',Tau);
end

