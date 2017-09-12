function q = calc_q(q_i, x_f, obs_pos, obs_rad, bot)
%calc_q Generate a joint space path while avoiding an obstacle
%   q_i = initial configuration [q1, q2, q3, q4]
%   x_f = final position goal in task space [x,y,z]'
%   obs_pos = obstacle position in task space [x,y,z]'
%   obs_rad = obstace radius in meters
%   bot = Serial link robot using robotics toolbox
H = bot.fkine(q_i);
x = H(1:3,4);
J = bot.jacob0(q_i);
Ja = J(1:3,:);
K = 0.01*eye(3);
qk = q_i;
q = [qk];
while sum((x-x_f).^2) > 1e-2,
    qdot = Ja.'*K*(x_f-x);
    q_k_pl_1 = qk + qdot.';
    [H, all] = bot.fkine(q_k_pl_1);
    %position of joint 3 which is at the same radius as the obstacle
    x_c = all(1:3,4,3);
    %distance between the obstacle center and joint 3
    int_dist = sqrt(sum((x_c' - obs_pos).^2));
    %adjust q_k_pl_1 to not hit obstacle assuming link radius of 0.75
    while int_dist < obs_rad+0.75,
        q_k_pl_1(1) = q_k_pl_1(1) + 0.001;
        [H,all] = bot.fkine(q_k_pl_1);
        x_c = all(1:3,4,3);
        int_dist = sqrt(sum((x_c' - obs_pos).^2));
    end
    x = H(1:3,4);
    J = bot.jacob0(q_k_pl_1);
    Ja = J(1:3,:);
    qk = q_k_pl_1;
    q(end+1,:) = qk;
end

end

