%Check Cayley-Hamilton Theorem Example using general method
clear
close 
clc

%% Problem 6.2
clear
close
clc

syms t s

A1 = [1 1 0; 0 1 0; 0 0 1];
A2 = [1 1 0; 0 0 1; 0 0 1];
A3 = [2 0 0 0; 2 2 0 0; 0 0 3 3; 0 0 0 3];

%Check Answer A1
eA1t = expm(A1*t);
A1*A1*A1; %I used this to see pattern for A^t

%Check Answer A2
sI_Ainv = inv(s*eye(3) - A2);
eA2t = expm(A2*t);
A2*A2*A2; %I used this to see pattern for A^t

%Check Answer A3
sI_A31 = [s-2, 0; -2, s-2];
sI_A31inv = inv(sI_A31);
sI_A32inv = inv([s-3 -3; 0 s-3]);
eA3t = expm(A3*t);
A3*A3*A3; %I used this to see pattern for A^t
%% Problem 7.1
clear
close 
clc 

syms w s
A = [0 -2*w^2 0 -w^4 0; 1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 1 0]
J = [0 0 0 0 0; 0 -w*i 1 0 0; 0 0 -w*i 0 0; 0 0 0 w*i 1; 0 0 0 0 w*i];
[V,Lamda] = eig(A); %compute eigenvalues and eigenvectors
v_4 = [4*w^3*i -3*w^2 -2*w*i 1 0].';
v_5 = [-4*w^3*i -3*w^2 2*w*i 1 0].';
P = [V(:,1), V(:,2), v_4, V(:,3), v_5];
P*J*inv(P)





