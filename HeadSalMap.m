%% HeadSalMap.m ---
%
% Filename: HeadSalMap.m
% Author: Fred Qi
% Created: 2013-05-30 11:13:19(+0800)
%
% Last-Updated: 2017-05-30 14:12:19(+0800) [by Fred Qi]
%     Update #: 44
%=====================================================================
%% Commentary:
%  imgIn: the input equirectangular image organised in an RGB, 
%         with size(imgIn) being [Height,Width,3].
%  matOut: the output “double” matrix having the saliency values.
%          Its size is [Height,Width]
%
%=====================================================================
%% Change Log:
%  - Originaly created by Chen XIA
%  - Revised by Hao LEE
%  - Revised for ICME 2017 GC: Salient 360!
%    + Chunhuan LIN
%    + Zhaohui XIA
%    + Fred QI
%
%=====================================================================

function [matOut] = HeadSalMap(imgIn)

% disp('Predicting HeadSalMap...');

[input_h,input_w,~] = size(imgIn);
if input_h*2 == input_w
    equi = imgIn;
else
    % disp('resizing...');
    equi = imresize(imgIn, [input_h, input_h*2]);
end

% create cubic images
cubic = equi2cubic(imgIn);

% Show the cube faces 
% figure;
% for idx = 1 : 6
%     % Show image in figure and name the face
%     subplot(2,3,idx);
%     imshow(cubic{idx});
%     title('faces');
%     % Write the image to disk
%     % imwrite(out{idx}, names_to_save{idx});
% end

% restore = cubic2equi(cubic{5}, cubic{6}, cubic{4}, cubic{2}, cubic{1}, cubic{3});
% figure;
% imshow(restore);
% imwrite(restore, 'restored_image.jpg')
% title('restored image')

center = horzcat(cubic{1:4});

% addpath('D:/godqi_sal360-master');
R = 15;
r = 7;
num = 10;
each_image_patches = 200;
batchsize = 100;

maxepoch = 100;
epsilonw      = 0.1;   % Learning rate for weights
epsilonvb     = 0.1;   % Learning rate for biases of visible units
epsilonhb     = 0.1;   % Learning rate for biases of hidden units
weightcost    = 0.0002;
initialmomentum  = 0.5;
finalmomentum    = 0.9;


epsilonw_l    = 0.001; % Learning rate for weights
epsilonvb_l   = 0.001; % Learning rate for biases of visible units
epsilonhb_l   = 0.001; % Learning rate for biases of hidden units

maxepoch_bp = 10;

numhid1 = 512; numpen = 256; numpen2 = 128; numpen3 = 64; numopen = 16;

%parfor img_idx = 1:120
    

sur_patch_size = R*2+1;
cen_patch_size = r*2+1;
training_data = zeros(each_image_patches, sur_patch_size.^2*3);
targets_data = zeros(each_image_patches, cen_patch_size.^2*3);
    
    % infilename = sprintf('Results_n3/%d+3.jpg', img_idx);
    % img_exist = exist(infilename, 'file');
    % if img_exist == 0
    
    % filename = sprintf('Bruce/%d.jpg', img_idx);
    % filename = sprintf('/home/shenchong/work/saliency/Judd_all/%d.jpg', img_idx);
	% filename = sprintf('Kootstra/%d.png', img_idx);
	% filename = sprintf('NUSEF/%d.jpg', img_idx);
    
img = double(center);
[row_s,col_s,~] = size(img);
img = imresize(img, 128/min(size(img, 1),size(img, 2)),'bilinear');
img = double(func_rgb2opponent(img));
[row,col,~] = size(img);
    
row_bg = floor(row*0.4);
col_bg = floor(col*0.4);
for patch_idx = 1:each_image_patches
    xPos_sam = randi([-row_bg,row_bg]);
    yPos_sam = randi([-col_bg,col_bg]);
    if xPos_sam>0
        xPos = xPos_sam+R;
    else
        xPos = row+xPos_sam-R;
    end
        
    if yPos_sam>0
        yPos = yPos_sam+R;
    else
        yPos = col+yPos_sam-R;
    end
        
    img_s = img(xPos-R:xPos+R,yPos-R:yPos+R,:);
    img_c = img_s(R+1-r:R+1+r,R+1-r:R+1+r,:);
    training_data(patch_idx, :) = img_s(:)';
    targets_data(patch_idx, :) = img_c(:)';
end
training_data = training_data/255;
targets_data = targets_data/255;
    %% mnistdeepauto
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% begin makebatches %%%%%%%%%%%%%%%%%%%%%%%%%%
    totnum = size(training_data,1);
    rand('state',0);
    randomorder = randperm(totnum);
    numbatches = totnum/batchsize;
    numdims = size(training_data,2);
    numdims1 = size(targets_data,2);
    % batchsize = 100;
    batchdata = zeros(batchsize, numdims, numbatches);
    batchtargets_data = zeros(batchsize, numdims1, numbatches);
    for b = 1:numbatches
        batchdata(:,:,b) = training_data(randomorder(1+(b-1)*batchsize:b*batchsize), :);
        batchtargets_data(:,:,b) = targets_data(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    end;
    rand('state',sum(100*clock));
    randn('state',sum(100*clock));
    %%%%%%%%%%%%%%%%%%%%%%%%%%% end makebatches %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% rbm;
    numhid = numhid1;
    [numcases, numdims, numbatches] = size(batchdata);
    vishid     = 0.1*randn(numdims, numhid);
    hidbiases  = zeros(1,numhid);
    visbiases  = zeros(1,numdims);
    vishidinc  = zeros(numdims,numhid);
    hidbiasinc = zeros(1,numhid);
    visbiasinc = zeros(1,numdims);
    batchposhidprobs = zeros(numcases,numhid,numbatches);
    
    for epoch = 1:maxepoch,
        for batch = 1:numbatches,
            %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            data = batchdata(:,:,batch);
            poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));
            batchposhidprobs(:,:,batch)=poshidprobs;
            posprods    = data' * poshidprobs;
            poshidact   = sum(poshidprobs);
            posvisact = sum(data);
            %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            poshidstates = poshidprobs > rand(numcases,numhid);
            %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
            neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));
            negprods  = negdata'*neghidprobs;
            neghidact = sum(neghidprobs);
            negvisact = sum(negdata);
            %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if epoch>5,
                momentum=finalmomentum;
            else
                momentum=initialmomentum;
            end;
            %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
            visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
            hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
            vishid = vishid + vishidinc;
            visbiases = visbiases + visbiasinc;
            hidbiases = hidbiases + hidbiasinc;
            %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end;
    vishid1 = vishid; hidrecbiases = hidbiases; visbiases1 = visbiases;
    
    batchdata = batchposhidprobs;
    numhid = numpen;
    %% rbm;
    [numcases, numdims, numbatches] = size(batchdata);
    vishid     = 0.1*randn(numdims, numhid);
    hidbiases  = zeros(1,numhid);
    visbiases  = zeros(1,numdims);
    vishidinc  = zeros(numdims,numhid);
    hidbiasinc = zeros(1,numhid);
    visbiasinc = zeros(1,numdims);
    batchposhidprobs = zeros(numcases,numhid,numbatches);
    
    for epoch = 1:maxepoch,
        for batch = 1:numbatches,
            %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            data = batchdata(:,:,batch);
            poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));
            batchposhidprobs(:,:,batch)=poshidprobs;
            posprods    = data' * poshidprobs;
            poshidact   = sum(poshidprobs);
            posvisact = sum(data);
            %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            poshidstates = poshidprobs > rand(numcases,numhid);
            %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
            neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));
            negprods  = negdata'*neghidprobs;
            neghidact = sum(neghidprobs);
            negvisact = sum(negdata);
            %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if epoch>5,
                momentum=finalmomentum;
            else
                momentum=initialmomentum;
            end;
            %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
            visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
            hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
            vishid = vishid + vishidinc;
            visbiases = visbiases + visbiasinc;
            hidbiases = hidbiases + hidbiasinc;
            %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end;
    
    hidpen = vishid; penrecbiases = hidbiases; hidgenbiases = visbiases;
    
    batchdata = batchposhidprobs;
    numhid = numpen2;
    %% rbm;
    [numcases, numdims, numbatches] = size(batchdata);
    vishid     = 0.1*randn(numdims, numhid);
    hidbiases  = zeros(1,numhid);
    visbiases  = zeros(1,numdims);
    vishidinc  = zeros(numdims,numhid);
    hidbiasinc = zeros(1,numhid);
    visbiasinc = zeros(1,numdims);
    batchposhidprobs = zeros(numcases,numhid,numbatches);
    
    for epoch = 1:maxepoch,
        for batch = 1:numbatches,
            %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            data = batchdata(:,:,batch);
            poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));
            batchposhidprobs(:,:,batch)=poshidprobs;
            posprods    = data' * poshidprobs;
            poshidact   = sum(poshidprobs);
            posvisact = sum(data);
            %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            poshidstates = poshidprobs > rand(numcases,numhid);
            %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
            neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));
            negprods  = negdata'*neghidprobs;
            neghidact = sum(neghidprobs);
            negvisact = sum(negdata);
            %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if epoch>5,
                momentum=finalmomentum;
            else
                momentum=initialmomentum;
            end;
            %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
            visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
            hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
            vishid = vishid + vishidinc;
            visbiases = visbiases + visbiasinc;
            hidbiases = hidbiases + hidbiasinc;
            %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end;
    hidpen2 = vishid; penrecbiases2 = hidbiases; hidgenbiases2 = visbiases;
    % save mnisthp2 hidpen2 penrecbiases2 hidgenbiases2;
    
    % fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d \n',numpen2,numpen3);
    batchdata = batchposhidprobs;
    numhid = numpen3;
    %% rbm;
    [numcases, numdims, numbatches] = size(batchdata);
    vishid     = 0.1*randn(numdims, numhid);
    hidbiases  = zeros(1,numhid);
    visbiases  = zeros(1,numdims);
    vishidinc  = zeros(numdims,numhid);
    hidbiasinc = zeros(1,numhid);
    visbiasinc = zeros(1,numdims);
    batchposhidprobs = zeros(numcases,numhid,numbatches);
    
    for epoch = 1:maxepoch,
        for batch = 1:numbatches,
            %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            data = batchdata(:,:,batch);
            poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));
            batchposhidprobs(:,:,batch)=poshidprobs;
            posprods    = data' * poshidprobs;
            poshidact   = sum(poshidprobs);
            posvisact = sum(data);
            %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            poshidstates = poshidprobs > rand(numcases,numhid);
            %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
            neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));
            negprods  = negdata'*neghidprobs;
            neghidact = sum(neghidprobs);
            negvisact = sum(negdata);
            %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if epoch>5,
                momentum=finalmomentum;
            else
                momentum=initialmomentum;
            end;
            %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
            visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
            hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
            vishid = vishid + vishidinc;
            visbiases = visbiases + visbiasinc;
            hidbiases = hidbiases + hidbiasinc;
            %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end;
    hidpen3 = vishid; penrecbiases3 = hidbiases; hidgenbiases3 = visbiases;
    
    batchdata = batchposhidprobs;
    numhid = numopen;
    %% rbmhidlinear;
    [numcases, numdims, numbatches] = size(batchdata);
    vishid     = 0.1*randn(numdims, numhid);
    hidbiases  = zeros(1,numhid);
    visbiases  = zeros(1,numdims);
    vishidinc  = zeros(numdims,numhid);
    hidbiasinc = zeros(1,numhid);
    visbiasinc = zeros(1,numdims);
    sigmainc = zeros(1,numhid);
    batchposhidprobs = zeros(numcases,numhid,numbatches);
    for epoch = 1:maxepoch,
        for batch = 1:numbatches,
            %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            data = batchdata(:,:,batch);
            poshidprobs =  (data*vishid) + repmat(hidbiases,numcases,1);
            batchposhidprobs(:,:,batch)=poshidprobs;
            posprods    = data' * poshidprobs;
            poshidact   = sum(poshidprobs);
            posvisact = sum(data);
            %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            poshidstates = poshidprobs+randn(numcases,numhid);
            %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
            neghidprobs = (negdata*vishid) + repmat(hidbiases,numcases,1);
            negprods  = negdata'*neghidprobs;
            neghidact = sum(neghidprobs);
            negvisact = sum(negdata);
            %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if epoch>5,
                momentum=finalmomentum;
            else
                momentum=initialmomentum;
            end;
            %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            vishidinc = momentum*vishidinc + ...
                epsilonw_l*( (posprods-negprods)/numcases - weightcost*vishid);
            visbiasinc = momentum*visbiasinc + (epsilonvb_l/numcases)*(posvisact-negvisact);
            hidbiasinc = momentum*hidbiasinc + (epsilonhb_l/numcases)*(poshidact-neghidact);
            vishid = vishid + vishidinc;
            visbiases = visbiases + visbiasinc;
            hidbiases = hidbiases + hidbiasinc;
            %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end
    hidtop = vishid; toprecbiases = hidbiases; topgenbiases = visbiases;
    
    %% backprop_cs;
    %%%%%%%%%%%%%%%%%%%%%%%%%% begin makebatches %%%%%%%%%%%%%%%%%%%%%%%%%%
    % totnum = size(training_data,1);
    rand('state',0);
    randomorder = randperm(totnum);
    numbatches = totnum/batchsize;
    numdims = size(training_data,2);
    numdims1 = size(targets_data,2);
    % batchsize = 100;
    batchdata = zeros(batchsize, numdims, numbatches);
    batchtargets_data = zeros(batchsize, numdims1, numbatches);
    for b = 1:numbatches
        batchdata(:,:,b) = training_data(randomorder(1+(b-1)*batchsize:b*batchsize), :);
        batchtargets_data(:,:,b) = targets_data(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    end;
    rand('state',sum(100*clock));
    randn('state',sum(100*clock));
    %%%%%%%%%%%%%%%%%%%%%%%%%%% end makebatches %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % [numcases,numdims,numbatches]=size(batchdata);
    % [~,numdims1,~]=size(batchtargets_data);
    %%%%%%%%%%%%%%% PREINITIALIZE WEIGHTS OF THE AUTOENCODER %%%%%%%%%%%%%%
    w1=[vishid1; hidrecbiases];
    w2=[hidpen; penrecbiases];
    w3=[hidpen2; penrecbiases2];
    w4=[hidpen3; penrecbiases3];
    w5=[hidtop; toprecbiases];
    w6=[hidtop'; topgenbiases];
    w7=[hidpen3'; hidgenbiases3];
    w8=[hidpen2'; hidgenbiases2];
    w9=[hidpen'; hidgenbiases];
    w10=[vishid1'; visbiases1];
    w_class = 0.1*randn(size(w10,2)+1,numdims1);
    %%%%%%%%%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS %%%%%%%%%%%%%%%%%
    l1=size(w1,1)-1;
    l2=size(w2,1)-1;
    l3=size(w3,1)-1;
    l4=size(w4,1)-1;
    l5=size(w5,1)-1;
    l6=size(w6,1)-1;
    l7=size(w7,1)-1;
    l8=size(w8,1)-1;
    l9=size(w9,1)-1;
    l10=size(w10,1)-1;
    l11=l1;
    l12=numdims1;
    
    for epoch = 1:maxepoch_bp
        tt=0;
        for batch = 1:numbatches/2
            %%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%
            tt=tt+1;
            data=[];
            targets_data=[];
            
            for kk=1:2
                data=[data
                    batchdata(:,:,(tt-1)*2+kk)];
                targets_data=[targets_data
                    batchtargets_data(:,:,(tt-1)*2+kk)];
            end
            %%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%
            max_iter=3;
            VV = [w1(:)' w2(:)' w3(:)' w4(:)' w5(:)' w6(:)' w7(:)' w8(:)' w9(:)' w10(:)' w_class(:)']';
            Dim = [l1; l2; l3; l4; l5; l6; l7; l8; l9; l10; l11; l12];
            
            [X, fX] = minimize(VV,'CG_MNIST_cs',max_iter,Dim,data,targets_data);
            
            w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
            xxx = (l1+1)*l2;
            w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
            xxx = xxx+(l2+1)*l3;
            w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
            xxx = xxx+(l3+1)*l4;
            w4 = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
            xxx = xxx+(l4+1)*l5;
            w5 = reshape(X(xxx+1:xxx+(l5+1)*l6),l5+1,l6);
            xxx = xxx+(l5+1)*l6;
            w6 = reshape(X(xxx+1:xxx+(l6+1)*l7),l6+1,l7);
            xxx = xxx+(l6+1)*l7;
            w7 = reshape(X(xxx+1:xxx+(l7+1)*l8),l7+1,l8);
            xxx = xxx+(l7+1)*l8;
            w8 = reshape(X(xxx+1:xxx+(l8+1)*l9),l8+1,l9);
            xxx = xxx+(l8+1)*l9;
            w9 = reshape(X(xxx+1:xxx+(l9+1)*l10),l9+1,l10);
            xxx = xxx+(l9+1)*l10;
            w10 = reshape(X(xxx+1:xxx+(l10+1)*l11),l10+1,l11);
            xxx = xxx+(l10+1)*l11;
            w_class = reshape(X(xxx+1:xxx+(l11+1)*l12),l11+1,l12);
            %%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%
        end
    end
    
    [rows,cols,dim] = size(img);
    min_size = min(rows,cols);
    level = 0;
    while min_size > 2*R+1
        level = level+1;
        rows_ = ceil(rows/2^(level-1));
        cols_ = ceil(cols/2^(level-1));
        min_size = min(rows_,cols_);
    end
    pyramid_input = cell(level,1);
    pyramid_output_1 = cell(level,1);
    
    min_size = min(rows,cols);
    level = 0;
    while min_size > 2*R+1
        level = level+1;
        rows_ = ceil(rows/2^(level-1));
        cols_ = ceil(cols/2^(level-1));
        min_size = min(rows_,cols_);
        pyramid_input{level} = imresize(img,[rows_,cols_]);
    end
    for k = 1 : 4
        if k>1
            num=5;
        else
            num=10;
        end
        
        img_data = pyramid_input{k};
        img1 = zeros(size(img_data,1)+2*R, size(img_data,2)+2*R, dim);
        img1(:,:,1) = padarray(img_data(:,:,1),[R R],'symmetric');
        img1(:,:,2) = padarray(img_data(:,:,2),[R R],'symmetric');
        img1(:,:,3) = padarray(img_data(:,:,3),[R R],'symmetric');
        
        rows_cord = ceil(size(img_data,1)/num);
        cols_cord = ceil(size(img_data,2)/num);
        
        saliency_map_1 = [];
        
        for i = 1:num
            saliency_map_r1 = [];
            
            for j = 1:num;
                top_cord = rows_cord*(i-1)+1;
                left_cord = cols_cord*(j-1)+1;
                bottom_cord = min(rows_cord+2*R+rows_cord*(i-1),size(img_data,1)+2*R);
                right_cord = min(cols_cord+2*R+cols_cord*(j-1),size(img_data,2)+2*R);
                mini_img = img1(top_cord:bottom_cord,left_cord:right_cord,:);
                mini_sal1 = func_sal(mini_img,R,r,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w_class);
                saliency_map_r1 = [saliency_map_r1 mini_sal1];
            end
            saliency_map_1 = [saliency_map_1;saliency_map_r1];
        end
        pyramid_output_1{k} = saliency_map_1/max(saliency_map_1(:));
    end
    
    saliency_map_ = zeros(rows,cols);
    
    for k = 1 : 4
        img_fusion = imresize(pyramid_output_1{k},[rows, cols]);
        img_fusion_norm = (img_fusion-min(img_fusion(:)))/(max(img_fusion(:))-min(img_fusion(:)));
        saliency_map_ = saliency_map_+img_fusion_norm;
        
        %saliency_s = imresize(saliency_map_,[row_s,col_s],'bicubic');
		%imwrite(mat2gray(saliency_s),sprintf('./result/center_%d_%d_%d_%d_out.jpg',k,each_image_patches,R,r));
    end
    
%% 2. normlization
    saliency_map_ = (saliency_map_-min(saliency_map_(:)))/(max(saliency_map_(:))-min(saliency_map_(:)));
    saliency_map_ = exp(saliency_map_);
    saliency_map_ = double( ( saliency_map_ - min(saliency_map_(:)) ) / ( max(saliency_map_(:)) - min(saliency_map_(:)) ) * 255 );
    saliency_s = saliency_map_;
    saliency_s = imresize(saliency_s,[row_s,col_s],'bicubic');
    %imwrite(mat2gray(saliency_s),sprintf('./result/center_%d_%d_%d_%d_normlization.jpg',k,each_image_patches,R,r));

    
%% ...    
%     img_t = double(center);
%     saliency_map_ = imresize(saliency_map_,[size(img_t,1),size(img_t,2)],'bicubic');

%% 4. Restore the cubic images to equirectangular image
w = 128;
top = zeros(w, w, 3);
bottom = zeros(w, w, 3);
sal2rgb = zeros(w, w*4, 3);
sal2rgb(:, :, 1) = saliency_map_;
sal2rgb(:, :, 2) = saliency_map_;
sal2rgb(:, :, 3) = saliency_map_;
saliency_map_ = sal2rgb;
saliency_map_ = cubic2equi(top, bottom, saliency_map_(:,3*w+1:4*w, :), saliency_map_(:,w+1:2*w, :), saliency_map_(:,1:w, :), saliency_map_(:,2*w+1:3*w, :));
saliency_map_ = saliency_map_(:, :, 1);
[r,c,~] = size(saliency_map_);

saliency_map_ = imresize(saliency_map_,[r, c]);
saliency_map_ = imresize(saliency_map_,[input_h, input_w]);

% add center bias
mapSize = size(saliency_map_);
kSize = mapSize(2)*0.04;
saliency_map_ = imfilter(saliency_map_, fspecial('gaussian', round([kSize, kSize]*2), kSize));

% normalization
saliency_map_ = double(saliency_map_-min(saliency_map_(:)));
saliency_map_ = saliency_map_/sum(saliency_map_(:));
matOut = saliency_map_;

% figure;
% imshow(mat2gray(saliency_map_))
% $$$ name = strtok(Files(number).name,'.');
% $$$ imwrite(mat2gray(saliency_map_), sprintf('%sSH_%s.jpg',out_path,name));
% $$$ save(sprintf('%sSH_%s.mat', out_path, name), 'saliency_map_');
% $$$ fid = fopen(sprintf('%sSH_%s.bin',out_path, name),'wb');
% $$$ fwrite(fid,'saliency_map_','double');
% $$$ disp(['Finishing ', Files(number).name])
% $$$ end

%=====================================================================
% HeadSalMap.m ends here
