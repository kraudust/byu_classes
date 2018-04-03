clear all
close all


%% Problem 55
%note: I wasn't sure what the book was asking, so I did my own application
%of an RLS filter

a = 0.01;
b = 2;
c = 3;
d = 10;
n = 4; %number of parameters to estimate
t = 0:.001:20;
t = t';
%y = a*t.^2 + b.*t + c + 5*randn(1,length(t))';
y = a*t.^2 + b.*t + c  + d*sin(t) + 5*randn(1,length(t))';
%Use RLS Filter to step through all data and update the model (a,b,c,d)
%Initialization
%Pm = eye(3,3);
Pm = eye(n,n);
%x_m = zeros(3,1);
x_m = zeros(n,1)+50;
x_store(1,:) = x_m;

for i = 1:length(t)
    am1 = [t(i)^2; t(i); 1; sin(t(i))];
    bm1 = y(i);
    Km1 = (Pm*am1)/(1+am1'*Pm*am1);
    xm1 = x_m + Km1*(bm1 - am1'*x_m);    
    Pm1 = Pm - Km1*am1'*Pm;    
    Pm = Pm1;
    x_m = xm1;
    x_store(i+1,:) = x_m';
end

figure()
plot(t,y)
xlabel('t (sec)')
ylabel('y')
hold on
three_vec = x_m(3)*ones(length(t),1);
% plot(t,x_m(1)*t.^2 + x_m(2).*t + three_vec + x_m(4)*sin(t))
plot(t,x_store(1:end-1,1).*t.^2 + x_store(1:end-1,2).*t + x_store(1:end-1,3) + x_store(1:end-1,4).*sin(t))
legend('True Data', 'RLS Estimate')

figure()
plot(t,x_store(1:20001,:))
legend('a', 'b', 'c', 'd')
xlabel('Time (sec)')
ylabel('Parameter Values')
title('Recursive Least Squares Paramter Estimation')

