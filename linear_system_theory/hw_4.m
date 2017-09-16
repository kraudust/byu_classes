%% Problem 12.2
%part a)
syms w
%w = 0;
A = [0 1 0 0; 3*w^2 0 0 2*w; 0 0 0 1; 0 -2*w 0 1];
B = [0 0; 1 0; 0 0; 0 1];
C = [B, A*B, A*A*B, A*A*A*B]
rank(C)
%part b)
B = [0 0; 0 0; 0 0; 0 1];
C = [B, A*B, A*A*B, A*A*A*B]

rank(C)

B = [0 0; 1 0; 0 0; 0 0];
C = [B, A*B, A*A*B, A*A*A*B]
rank(C)

%% Problem 14.3
A = [6 4 1; -5 -4 0; -4 -3 -1];
B = [1; -1; -1];
C = ctrb(A,B)
P = poly(A)
T = C*[1 P(2) P(3);0 1 P(2); 0 0 1]
inv(T)*A*T

%% Problem 13.1
A = [-1 0; 0 -1];
B = [-1; 1];
C = [1 0; 0 1];
contr = [B, A*B]
U = [-1 1; 1 1];
Uinv = inv(U);
Abar = Uinv*A*U
Bbar = Uinv*B
Cbar = C*U
