function [mini_data1 mini_data] = func_sal_(mini_image,R,r,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w_class)
[rows,cols,~] = size(mini_image);
N = (rows-2*R)*(cols-2*R);

data = zeros(N,(R*2+1)^2*3);
data1 = zeros(N,(r*2+1)^2*3);

for i = 1:rows-2*R
    for j = 1:cols-2*R
        
        i1 = i+R;
        j1 = j+R;
        
        w = double(mini_image(i1-R:i1+R,j1-R:j1+R,:));
        data((i-1)*(cols-2*R)+j,:) = w(:)';
        w_ = double(mini_image(i1-r:i1+r,j1-r:j1+r,:));
        data1((i-1)*(cols-2*R)+j,:) = w_(:)';
    end
end
data = data/255;
data1 = data1/255;

data = [data ones(N,1)];
w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
w4probs = 1./(1 + exp(-w3probs*w4)); w4probs = [w4probs  ones(N,1)];
w5probs = w4probs*w5; w5probs = [w5probs  ones(N,1)];
w6probs = 1./(1 + exp(-w5probs*w6)); w6probs = [w6probs  ones(N,1)];
w7probs = 1./(1 + exp(-w6probs*w7)); w7probs = [w7probs  ones(N,1)];
w8probs = 1./(1 + exp(-w7probs*w8)); w8probs = [w8probs  ones(N,1)];
w9probs = 1./(1 + exp(-w8probs*w9)); w9probs = [w9probs  ones(N,1)];
w10probs = 1./(1 + exp(-w9probs*w10)); w10probs = [w10probs  ones(N,1)];
dataout = 1./(1 + exp(-w10probs*w_class));
% residual = data1-dataout;
% mini_sal = zeros(rows-2*R, cols-2*R, (r*2+1)^2*3);
% for k = 1:(r*2+1)^2*3
%     mini_sal_ = residual(:,k);
%     mini_sal(:,:,k) = reshape(mini_sal_, [cols-2*R, rows-2*R])';
% end
mini_data1 = zeros(rows-2*R, cols-2*R, (r*2+1)^2*3);
mini_data = zeros(rows-2*R, cols-2*R, (r*2+1)^2*3);
for k = 1:(r*2+1)^2*3
    mini_data1_ = data1(:,k);
    mini_data1(:,:,k) = reshape(mini_data1_, [cols-2*R, rows-2*R])';
    mini_data_ = dataout(:,k);
    mini_data(:,:,k) = reshape(mini_data_, [cols-2*R, rows-2*R])';
end