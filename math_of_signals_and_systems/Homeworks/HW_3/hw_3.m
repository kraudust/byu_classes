clear all
close all

%% Problem 1 Part a
t = linspace(-1, 1, 10000);
p = zeros(10000, 5);
for kk = 1:5
    p(:, kk) = t.^(kk-1);
end
p1 = p(:,1);
p2 = p(:,2);
p3 = p(:,3);
p4 = p(:,4);
p5 = p(:,5);
%Step 1: Normalize p1
induced_norm_p1 = inner_product(t,p1,p1).^0.5;
q1 = p1/induced_norm_p1;

%Step 2: project p2 onto q1, and find the difference, then normalize it
e2 = p2 - inner_product(t,p2,q1)*q1;
q2 = e2/inner_product(t,e2,e2).^0.5;

%Step 3: project p3 onto the span of the previous vectors, find the
%difference (orthogonal to the span), and normalize that difference
e3 = p3 - inner_product(t,p3,q1)*q1 - inner_product(t,p3,q2)*q2;
q3 = e3/inner_product(t,e3,e3).^0.5;

%Step 4: Rinse and repeat
e4 = p4 - inner_product(t,p4,q1)*q1 - inner_product(t,p4,q2)*q2 ...
    - inner_product(t,p4,q3)*q3;
q4 = e4/inner_product(t,e4,e4).^0.5;

e5 = p5 - inner_product(t,p5,q1)*q1 - inner_product(t,p5,q2)*q2 ...
    - inner_product(t,p5,q3)*q3 - inner_product(t,p5,q4)*q4;
q5 = e5/inner_product(t,e5,e5).^0.5;

figure()
plot(t,q1,t,q2,t,q3,t,q4,t,q5);
title('Legendre Polynomials')
xlim([-1 1])
ylim([-1 1])
xlabel('t')
ylabel('f(t)')
legend('q1', 'q2', 'q3', 'q4', 'q5')


%% Problem 1 Part b
f = exp(-t');
f_est = inner_product(t,f,q1)*q1 + inner_product(t,f,q2)*q2 + ...
    inner_product(t,f,q3)*q3 + inner_product(t,f,q4)*q4 + ...
    inner_product(t,f,q5)*q5;
error_norm_leg = norm(f-f_est);

figure()
plot(t,f,t,f_est)
title('Legendre polynomial fit')
legend('f', 'f_{est}')

%% Problem 1 Part c
%The Chebyshev polynomials aren't defined at the endpoints (goes to inf)
t = t(2:end-1);
p = p(2:end-1,:);
p1 = p(:,1);
p2 = p(:,2);
p3 = p(:,3);
p4 = p(:,4);
p5 = p(:,5);

induced_norm_p1 = w_inner_product(t,p1,p1).^0.5;
q1 = p1/induced_norm_p1;

%Step 2: project p2 onto q1, and find the difference, then normalize it
e2 = p2 - w_inner_product(t,p2,q1)*q1;
q2 = e2/w_inner_product(t,e2,e2).^0.5;

%Step 3: project p3 onto the span of the previous vectors, find the
%difference (orthogonal to the span), and normalize that difference
e3 = p3 - w_inner_product(t,p3,q1)*q1 - w_inner_product(t,p3,q2)*q2;
q3 = e3/w_inner_product(t,e3,e3).^0.5;

%Step 4: Rinse and repeat
e4 = p4 - w_inner_product(t,p4,q1)*q1 - w_inner_product(t,p4,q2)*q2 ...
    - w_inner_product(t,p4,q3)*q3;
q4 = e4/w_inner_product(t,e4,e4).^0.5;

e5 = p5 - w_inner_product(t,p5,q1)*q1 - w_inner_product(t,p5,q2)*q2 ...
    - w_inner_product(t,p5,q3)*q3 - w_inner_product(t,p5,q4)*q4;
q5 = e5/w_inner_product(t,e5,e5).^0.5;

figure()
plot(t,q1,t,q2,t,q3,t,q4,t,q5);
title('Chebyshev Polynomials')
xlim([-1 1])
ylim([-1 1])
xlabel('t')
ylabel('f(t)')
legend('q1', 'q2', 'q3', 'q4', 'q5')

%% Problem 1 Part d
f = exp(-t');
f_est = w_inner_product(t,f,q1)*q1 + w_inner_product(t,f,q2)*q2 + ...
    w_inner_product(t,f,q3)*q3 + w_inner_product(t,f,q4)*q4 + ...
    w_inner_product(t,f,q5)*q5;
error_norm_chev = norm(f-f_est);

figure()
plot(t,f,t,f_est)
title('Chebyshev Polynomial Fit')
legend('f', 'f_{est}')

%% Problem 1 Part e
%My error from the Legendre polynomial was slightly smaller than my error
%from the Chebyshev polynomial fit. I think this is because of the
%weighting.

%% Problem 2
%   Load in the matrices needed for this problem
load('prob2.mat');

%   There are two matrices, one containing 20
%   basis images, and one containing the image
%   x that we are trying to decode.

%   Display the 20 basis images in matrix A
figure;
for kk = 1:20
    subplot(4,5,kk);
    imshow(A(:,:,kk),[]);
end

%   And display the image x
figure;
imshow(x,[]);

%   The image you are trying to decode is hidden
%   in x.  In order to decode it, you need to
%   decompose x = x_hat + e, where x_hat is an
%   orthogonal projection of x onto the space
%   spanned by the 20 images in A, and e is the
%   error vector between x and its projection.
%   The hidden image will be the error vector e.

%   Some useful hints:

%   (1) To deal with images as vectors, you should
%       "flatten" each image into a column vector that
%       is 256*256 = 65,536 elements long.
%       The "reshape" command in Matlab will come
%       in handy here.
%
%   (2) Once you are done with your vector math, you
%       will probably want to reshape your result
%       back to a 256 x 256 image so you can display it.
%
%   (3) Make sure your images are either scaled from
%       0 to 1 if you are calling "imshow(image)", or
%       that you use the form "imshow(image, [])"
%       where the "[]" tells imshow to scale black
%       and white values between the minimum and
%       maximum values in the image.

%first flatten images
A = reshape(A,256*256,20);
x = reshape(x, 256*256,1);

% A_orth = zeros(size(A));
%Orthogoanalize A
A_orth = orth(A);

%find x-hat
x_hat = zeros(size(x));
for i = 1:20
   x_hat = x_hat + dot(x,A_orth(:,i))*A_orth(:,i); 
end

%find image
e = x-x_hat;

x_hat = reshape(x_hat,256,256);
e = reshape(e,256,256);

figure()
imshow(e,[]);
title('Decoded Image');

figure()
imshow(x_hat,[]);
title('x_hat');


