function [Vhat, policy] = backup_step(V, m, n)
global nom_cell_reward gamma prob

p_not = (1-prob)/2;
% North
vn = prob*V(m, n+1) + p_not*V(m+1, n) + p_not*V(m-1, n);
% East
ve = prob*V(m+1,n) + p_not*V(m, n+1) + p_not*V(m,n-1);
% South
vs = prob*V(m, n-1) + p_not*V(m+1, n) + p_not*V(m-1, n);
% West
vw = prob*V(m-1, n) + p_not*V(m, n+1) + p_not*V(m, n-1);

[val, policy] = max([vn, vw, vs, ve]); %policy 1 is north, 2 is east, 3 is south, 4 is west
Vhat = gamma*(val + nom_cell_reward);
end

