clear all
close all

%% Phils Problem 55

a = 1;
b = 2;
c = 3;

t = 0:.001:10;
y = a*t.^2 + b.*t + c + 5*randn(1,length(t));

figure(1)
plot(t,y)
title('ax^2 + bx + c with noise')

% Recursive LS
Pm = eye(3,3);
%xm = A(3,:)';
xm = zeros(3,1);

xlist = [];
for i=1:length(t)
    
    am1 = [t(i)^2  t(i)  1]';
    bm1 = y(i);
    
    Km1 = (Pm*am1)/(1+am1'*Pm*am1);
    xm1 = xm + Km1*(bm1 - am1'*xm);
    Pm1 = Pm - Km1*am1'*Pm;
    Pm = Pm1;
    xm = xm1;
    xlist = [xlist xm];
    k_store(i,:) = Km1;
    
end

figure(2)
plot(xlist')
legend('a','b','c')
title('Estimated a, b and c')

figure()
plot(k_store)