function [Q,R] = myqr(A)
%MYQR Splits an mxn matrix (A) into it's QR factorization
%   Q is unitary, and R is upper triangular
m = size(A,1); %number of rows
n = size(A,2); %number of columns
R = A;
if m > n,
    Qi = zeros(m,m,n);
    for i = 1:n,
        a = R(i:m,i);
        e = eye(length(a),1);
        na = sqrt(sum(a.*(a').')); %norm of a
        v = a + sign(a(1))*na*e;
        Hv = eye(m-(i-1)) - 2*(v*v')/(v'*v); %householder transform of v
        dim_Hv = size(Hv,1);
        if m-dim_Hv == 0,
            Qi(:,:,i) = Hv;
        else
            Qi(:,:,i) = [...
            eye(m-dim_Hv)                   zeros(m - dim_Hv,dim_Hv);...
            zeros(dim_Hv,m-dim_Hv)          Hv];
        end
        R = Qi(:,:,i)*R;
    end
    Q = Qi(:,:,1)';
    for i = 1:n-1
        Q = Q*Qi(:,:,i+1)';
    end
else
    Qi = zeros(m,m,m-1);
    for i = 1:m,
        a = R(i:m,i);
        e = eye(length(a),1);
        na = sqrt(sum(a.*(a').')); %norm of a
        v = a + sign(a(1))*na*e;
        Hv = eye(m-(i-1)) - 2*(v*v')/(v'*v); %householder transform of v
        dim_Hv = size(Hv,1);
        if m-dim_Hv == 0,
            Qi(:,:,i) = Hv;
        else
            Qi(:,:,i) = [...
            eye(m-dim_Hv)                   zeros(m - dim_Hv,dim_Hv);...
            zeros(dim_Hv,m-dim_Hv)          Hv];
        end
        R = Qi(:,:,i)*R;
    end
    Q = Qi(:,:,1)';
    for i = 1:m-1
        Q = Q*Qi(:,:,i+1)';
    end
end

end

