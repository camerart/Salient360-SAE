function [vishid,hidbiases,visbiases,batchposhidprobs] = rbm( numhid, batchdata)

maxepoch = 100;
epsilonw      = 0.1;   % Learning rate for weights
epsilonvb     = 0.1;   % Learning rate for biases of visible units
epsilonhb     = 0.1;   % Learning rate for biases of hidden units
weightcost    = 0.0002;
initialmomentum  = 0.5;
finalmomentum    = 0.9;

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



end

