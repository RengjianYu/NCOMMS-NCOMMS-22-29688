clear all
% read the pattern
load_data = load('H_pattern_conductance');
H_pattern_conductance = load_data.H_pattern_conductance;
load_data = load('I_pattern_conductance');
I_pattern_conductance = load_data.I_pattern_conductance;
load_data = load('J_pattern_conductance');
J_pattern_conductance = load_data.J_pattern_conductance;
load_data = load('K_pattern_conductance');
K_pattern_conductance = load_data.K_pattern_conductance;
load_data = load('L_pattern_conductance');
L_pattern_conductance = load_data.L_pattern_conductance;
[width,height,num] = size(H_pattern_conductance);

% construct the dataset
data_set(:,:,:,1) = H_pattern_conductance;
data_set(:,:,:,2) = I_pattern_conductance;
data_set(:,:,:,3) = J_pattern_conductance;
data_set(:,:,:,4) = K_pattern_conductance;
data_set(:,:,:,5) = L_pattern_conductance;

% set the label (real output current)
H_label = 5;
I_label = 10;
J_label = 15;
K_label = 20;
L_label = 25;

% construct the labelset
label_set=[H_label,I_label,J_label,K_label,L_label];
lr = 0.001; % learning rate
w = rand(3,3)/3; b = rand(1);
for epoch=1:200
    % training
    for iter = 1:200
        for pattern = 1:5
            x = data_set(:,:,iter,pattern);
            y = label_set(pattern);
            [w, b, loss] = weightupdata(lr, w, b, x, y);
            
        end
    end
    loss_all(epoch)=loss;
    correct = 0;
    rec_matrix = zeros(5,5);
    % testing
    for iter = 1:200
        for pattern = 1:5
            x = data_set(:,:,iter,pattern);
            y = label_set(pattern);
            logit = sum(sum(w.*x))+b;
            if (logit>=25)
                rec_pattern = 5;
            else if (logit<=5)
                rec_pattern = 1;
                else
                rec_pattern = round(logit/5);
                end
            end
            rec_matrix(pattern,rec_pattern) = rec_matrix(pattern,rec_pattern)+1;
            if (rec_pattern == pattern)
            correct = correct+1;
            end
            if (epoch ==200)
                out(iter,pattern)=logit;
            end
        end
    end
    accuracy(epoch) = correct/1000;   
end
