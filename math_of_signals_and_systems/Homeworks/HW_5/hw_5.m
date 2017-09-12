clear all
close all
clc
%% Problem 4
A1 = [4 3; 3 6];
A2 = [1 2; 3 0];
A3 = [1 2; 0 1];

linf_A1 = norm(A1,Inf)
l1_A1 = norm(A1,1)
l2_A1 = norm(A1,2)
lF_A1 = norm(A1,'fro')

linf_A2 = norm(A2,Inf)
l1_A2 = norm(A2,1)
l2_A2 = norm(A2,2)
lF_A2 = norm(A2,'fro')

linf_A3 = norm(A3,Inf)
l1_A3 = norm(A3,1)
l2_A3 = norm(A3,2)
lF_A3 = norm(A3,'fro')

%% Problem 29
A = [1 4; 2 8; 3 12];
A_range = colspace(sym(A))
A_null = null(A,'r')
A_ad_range = colspace(sym(A.'))
A_ad_null = null(A.','r')
