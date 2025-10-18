clc; clear;
close all;

S = [];
L0_iso = [];
L0_ani = [];
wavelet =[];

L0_iso_1 = log(L0_iso);
L0= [L0_iso_1;L0_ani];

[n_sample,n_trace]=size(L0_M);

sita_1=10;
sita_2=20;
sita_3=30;
sita_4=40;
sita_5=50;
sita_6=60;
angle=[sita_1,sita_2,sita_3,sita_4,sita_5];    ang=angle*pi/180;

sita_phi_1=45;
sita_phi_2=90;

angle_phi=[sita_phi_1,sita_phi_2];    ang_phi=angle_phi*pi/180;
beta = 70 * pi / 180;

for i=1:n_trace
    g(:,i)=(mean(L0_MU(:,i))^2)/((mean(L0_M(:,i))^2));
end


for k=1:length(ang)
    a(k)=0.25*(sec(ang(k)))^2.0;
    b(k)= -2*g(:,i)*(sin(ang(k)))^2.0;
    c(k)= 0.25*cos(2*ang(k))*(cos(ang(k)))^2.0;
    d(k) = 0.5*(sin(ang(k)))^2.0;
    e(k) = 0.5*(sin(ang(k)))^2.0*(tan(ang(k)))^2.0;
    for j = 1:length(ang_phi)
        f(k,j) = (4/3)*(-1/((-1 + g(:,i))*g(:,i))*(-(-1/2 + g(:,i)*sin(beta)^2)^2 - 1/4*(1 - 4*g(:,i) ...
            + 2*g(:,i)*sin(beta)^2)*(1 + 2*g(:,i)*cos(2*ang_phi(j))*sin(beta)^2)*tan(ang(k))^2 - ...
            g(:,i)^2*sin(ang(k))^2*(-1 + sin(beta)^2*sin(ang_phi(j))^2)^2*tan(ang(k))^2)...
            + (1/(-3 + 2*g(:,i)))*4*g(:,i)*(cos(beta)^2*sec(ang(k))^2*sin(beta)^2 ...
            + (-1 - sin(beta)^2*sin(ang_phi(j))^2 + 2*sin(beta)^4*sin(ang_phi(j))^2 ...
            + sin(ang(k))^2*(1 - sin(beta)^4*sin(ang_phi(j))^4))*tan(ang(k))^2));
    end
end

blocks_temp = cell(5, 2); 
for i = 1:5          
    for j = 1:2      
        blocks_temp{i, j} = [
            diag(a(i) * ones(n_sample-1, 1)), ...
            diag(b(i) * ones(n_sample-1, 1)), ...
            diag(c(i) * ones(n_sample-1, 1)), ...
            diag(d(i) * ones(n_sample-1, 1)), ...
            diag(e(i) * ones(n_sample-1, 1)), ...
            diag(f(i, j) * ones(n_sample-1, 1)) ...
        ];
    end
end

G1 = [blocks_temp{1,1}; blocks_temp{2,1}; blocks_temp{3,1}; blocks_temp{4,1}; blocks_temp{5,1}; ...
     blocks_temp{1,2}; blocks_temp{2,2}; blocks_temp{3,2}; blocks_temp{4,2}; blocks_temp{5,2}];


W=wavelet_matrix(wavelet,n_sample-1);
W1 = kron(eye(10), W);


v=linspace(-1,-1,size(W,2));
v1=linspace(1,1,size(W,2)-1);
Dif=zeros(size(W,2));
Dif=diag(v);
Dif=Dif+diag(v1,1);
Dif2=zeros(size(Dif,1),1);
Dif2(end,:)=[1];
Dif=[Dif Dif2];

Dif1_temp = kron(eye(6), Dif);

G = W1 *G1 * Dif1_temp;

sigma_e_par = 0.001;
coeffs = [1; 1; 0.5; 0.5;0.5;0.05];
base = ones(n_sample, n_trace);
sigma_params = [coeffs(1)*base; coeffs(2)*base; coeffs(3)*base; coeffs(4)*base; coeffs(5)*base; coeffs(6)*base];

n_iter            = 2;
initial_step_size = 0.001;
burn_in           = round(0.5 * n_iter);
thin_interval     = 10;
target_acceptance = 0.234;
adapt_interval    = 100;

param_len = n_sample;
scales    = [1, 1, 1, 0.8, 0.8, 0.5].';
scale_vec = repelem(scales, param_len); 


N_data = size(S, 1);
temp_ERR = S - G * L0;
sigma_e  = sigma_e_par * std(temp_ERR(:));
log_like_const = -0.5*N_data*log(2*pi*sigma_e^2);


log_prior_const = -0.5*sum(log(2*pi*(sigma_params.^2)));

n_param = 6;
R_result = zeros(n_param*param_len, n_trace, 'double');

total_tic = tic;


for j = 1:n_trace
   
    S_col = S(:, j); 
    L0j    = L0(:, j); 
    scale_vec_g  = scale_vec;   

    R_current = L0j;
    data_residual = S_col - G * R_current;
    log_likelihood = log_like_const - (0.5/(sigma_e^2)) * (data_residual' * data_residual);
    prior_residual = R_current - L0j;
    log_prior = log_prior_const - 0.5 * sum((prior_residual.^2) ./ (sigma_params.^2));
    logp_current = log_likelihood + log_prior;

    running_mean = zeros(size(R_current));
    count = 0;

    step_size = initial_step_size;
    accept_count_block = 0;
    print_every = 1000;   


    for t = 1:n_iter
        if mod(t, adapt_interval) == 0 && t < burn_in
            current_acc = accept_count_block / adapt_interval;
            if current_acc < target_acceptance
                step_size = step_size * 0.9;
            else
                step_size = step_size * 1.1;
            end
            accept_count_block = 0;
        end

        noise = randn(size(R_current));
        R_proposal = R_current + step_size * (scale_vec_g .* noise);

        data_residual_p = S_col - G * R_proposal;
        log_like_p = log_like_const - (0.5/(sigma_e^2)) * (data_residual_p' * data_residual_p);

        prior_residual_p = R_proposal - L0j;
        log_prior_p = log_prior_const - 0.5 * sum((prior_residual_p.^2) ./ (sigma_params.^2));

        logp_proposal = log_like_p + log_prior_p;

        alpha = min(1, exp(logp_proposal - logp_current));
        if rand() < alpha
            R_current    = R_proposal;
            logp_current = logp_proposal;
            accept_count_block = accept_count_block + 1;
        end

        if t > burn_in && mod(t, thin_interval) == 0
            count = count + 1;
            running_mean = running_mean + (R_current - running_mean) / count;
        end

        if mod(t, print_every) == 0
            acc_rate_est = accept_count_block / min(adapt_interval, t);
            fprintf('Trace %d | iter %d / %d | local acc ~= %.2f | step=%.3g\n', ...
                j, t, n_iter, acc_rate_est, step_size);
        end
    end

    R_result(:, j) = running_mean;
end

total_time = toc(total_tic);
fprintf('\nTime: %.4f \n', total_time);

%% ===================== result =====================


L1_iso_temp = R_result(1:3*param_len, :);
L1_iso      = exp(L1_iso_temp);
L1_ani      = R_result(3*param_len+1:6*param_len, :);


M_result       = L1_iso(1:param_len, :);
MU_result      = L1_iso(param_len+1:2*param_len, :);
Den_result     = L1_iso(2*param_len+1:3*param_len, :);
Delta_result   = L1_ani(1:param_len, :);
Epsilon_result = L1_ani(param_len+1:2*param_len, :);
Fracture_den_result = L1_ani(2*param_len+1:3*param_len, :);

