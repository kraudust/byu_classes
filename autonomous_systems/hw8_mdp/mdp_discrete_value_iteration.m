function [Vhat, policy] = mdp_discrete_value_iteration(map)
    Vhat = map;
    global iter

    N = size(map,1);
    policy = zeros(N,N);
    conv = 5;
    iters = 0;
    while conv > 1
        for i = 1:N
            for j = 1:N
                if iter(i,j) == 1
                    [Vhat(i,j),policy(i,j)] = backup_step(Vhat,i,j); 
                end
            end
        end
        conv = norm(Vhat-map)
        map = Vhat;
        iters = iters + 1
    end

end

