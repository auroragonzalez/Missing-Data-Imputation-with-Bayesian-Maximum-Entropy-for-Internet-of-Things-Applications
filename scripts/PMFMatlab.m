% Version 2.000
% Code provided by Aurora González-Vidal (aurora.gonzalez2@um.es) and José Mendoza-Bernal
%
% Permission is granted for anyone to copy, use, modify, or distribute this program and accompanying 
% programs and documents for any purpose, provided this copyright notice is retained and prominently 
% displayed. The PMF original code was extracted from Ruslan Salakhutdinov webpage: https://www.cs.toronto.edu/~rsalakhu/BPMF.html
% If you use this file for academic purposes, please cite the scientific paper: "González-Vidal, A., Rathore, P., Rao, A. S., 
% Mendoza-Bernal, J., Palaniswami, M., & Skarmeta-Gómez, A. F. (2020). Missing Data Imputation with Bayesian Maximum Entropy 
% for Internet of Things Applications. IEEE Internet of Things Journal.
% The programs and documents are distributed without any warranty, express or implied.  As the programs 
% were written for research purposes only, they have  not been tested to the degree that would be advisable 
% in any important  application.  All use of these programs is entirely at the user's own risk.


clear;
clc;

%%%define result vectors
RMSE_normalised = [];

maxepoch = 100;     % maximum number of iterations for PMF
numclus = [20 1];
N1 = 600;

% dataset to be run (1 = IBRL, 2 = Beach)
for ds = 1:1  % for each dataset

    maxk = numclus(ds);
    cluster = repelem(1:maxk,1:maxk);

    RMSE_inner = zeros(maxepoch, length(cluster));  % vector of RMSE values within PMF iterations

    folder = {'IBRL', 'Beach'}; % folder for datasets
    R = dlmread(['data/' char(folder(ds)) '/R.csv']);       % data file for soft experiment
    Rgt = dlmread(['data/' char(folder(ds)) '/Rgt.csv']);   % data file for hard experiment
    Rgt = Rgt(:,1:N1);  % for hard experiment choose Rgt, for soft experiment choose R

    NAS = 45;            % NA percentage
    [Nt,M] = size(Rgt);
    msize = Nt*M;

    final = zeros(length(NAS),maxk); % will save RMSEs
    finalT= zeros(length(NAS),maxk); % will save running time

    ind=1;
    for kk = NAS:NAS  %select the percentageof NA for each iteration, from 1 to 75
        kk
        RMSEhard = [];
        timehard = [];
        nNA = Nt*M*kk/100; % total number of NAs (counting all sensors)
        rng(kk);
        matrixForNA=zeros(Nt,M);
        elements = randperm(msize, nNA);   % elements in the matrix that are going to be NA
        matrixForNA(elements) = 1;         % matrix for NA (1 = is NA, 0 = is not NA)

        ssaux = 1
        for k = 1:maxk   % for each number of clusters
            CLU = dlmread(['data/' char(folder(ds)) '/groupsk.txt']);   % clusters file
            CLU = CLU(:,[1,(k+1)]);  % cluster belonging

            for index = 1:k  % for each cluster group (if we are at k = 4 --> then 1,2,3,4
                index
                fprintf('############');
                fprintf('index= %s',index);
                submotes = CLU(find(CLU(:,2)==index),1).';
                submatrixForNA = matrixForNA(submotes,:);
                elements2 = find(submatrixForNA');

                mote = repelem(1:length(submotes),N1);
                obs = repmat(1:N1,1,length(submotes));

                temp = [];  % variable that is missing (temperature, humidity,...)
                tempN = []; % variable that is missing normalised

                for i= 1:length(submotes)
                    a = Rgt(submotes(i),:);
                    temp = [temp; a'];
                    b = (a-min(a))/(max(a)-min(a));
                    tempN = [tempN; b'];
                end

                all_data = zeros(length(mote),3);

                all_data(:,1) = mote;
                all_data(:,2) = obs;
                all_data(:,3) = tempN;

                all_index = 1:length(mote);
                test_index = elements2;
                ismem = ismember(all_index,test_index);
                train_index = all_index(~ismem);

                train = all_data(train_index,:);
                test = all_data(test_index,:);

                restart = 0;
                epsilon = 50; % Learning rate
                lambda = 0.01; % Regularization parameter
                momentum = 0.8;

                epoch = 1;

                mean_rating = mean(train(:,3));

                pairs_tr = length(train); % training data
                pairs_pr = length(test); % validation data

                numbatches = 1; % Number of batches
                num_m = size(Rgt,2);  % Number of observations
                num_p = size(Rgt,1);  % Number of subjets
                num_feat = 10; % Rank 10 decomposition

                w1_M1 = 0.1*randn(num_m, num_feat); % Observations feature vectors
                w1_P1 = 0.1*randn(num_p, num_feat); % Subjets feature vecators
                w1_M1_inc = zeros(num_m, num_feat);
                w1_P1_inc = zeros(num_p, num_feat);
                foo = 1;
                tic
                
                for epoch = epoch:maxepoch
                    
                    epoch

                    rr = randperm(pairs_tr);
                    train = train(rr,:);
                    clear rr

                    for batch = 1:numbatches

                        N=length(train)/numbatches; % number training triplets per batch

                        aa_p   = double(train((batch-1)*N+1:batch*N,1));
                        aa_m   = double(train((batch-1)*N+1:batch*N,2));
                        rating = double(train((batch-1)*N+1:batch*N,3));
                        rating = rating-mean_rating; % Default prediction is the mean rating.

                        %%%%%%%%%%%%%% Compute Predictions %%%%%%%%%%%%%%%%%
                        pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2);

                        %%%%%%%%%%%%%% Compute Gradients %%%%%%%%%%%%%%%%%%%
                        IO = repmat(2*(pred_out - rating),1,num_feat);
                        Ix_m = IO.*w1_P1(aa_p,:) + lambda*w1_M1(aa_m,:);
                        Ix_p = IO.*w1_M1(aa_m,:) + lambda*w1_P1(aa_p,:);

                        dw1_M1 = zeros(num_m,num_feat);
                        dw1_P1 = zeros(num_p,num_feat);

                        for ii=1:N
                            dw1_M1(aa_m(ii),:) =  dw1_M1(aa_m(ii),:) +  Ix_m(ii,:);
                            dw1_P1(aa_p(ii),:) =  dw1_P1(aa_p(ii),:) +  Ix_p(ii,:);
                        end

                        %%%% Update movie and user features %%%%%%%%%%%
                        w1_M1_inc = momentum*w1_M1_inc + epsilon*dw1_M1/N;
                        w1_M1 =  w1_M1 - w1_M1_inc;

                        w1_P1_inc = momentum*w1_P1_inc + epsilon*dw1_P1/N;
                        w1_P1 =  w1_P1 - w1_P1_inc;
                    end

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%% Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%%
                    NN = pairs_pr;

                    aa_p = double(test(:,1));
                    aa_m = double(test(:,2));

                    pred_out = sum(w1_M1(aa_m,:).*w1_P1(aa_p,:),2) + mean_rating;

                    realTest = temp(test_index);
                    maxTrain = max(temp(train_index));
                    minTrain = min(temp(train_index));

                    predTest = pred_out*(maxTrain-minTrain) + minTrain;

                    RMSE = sqrt(mean((realTest - predTest).^2));
                    RMSE_inner(epoch, ssaux) = RMSE;
                end

                timehard = [timehard, toc]
                %RMSE_normalised = [RMSE_normalised, err_valid(epoch)]
                RMSEhard = [RMSEhard,RMSE]

                foo = foo+1;
                ssaux = ssaux+1;
            end          
        end
        
        df = mat2dataset([cluster', RMSEhard']);
        dfstat = grpstats(df,'Var1');
        final(ind, :) = double(dfstat(:,3));

        df2 = mat2dataset([cluster', timehard']);
        dfstat = grpstats(df2,'Var1',{'sum'});
        finalT(ind, :) = double(dfstat(:,3));
        csvwrite(['data/' char(folder(ds)) '/PMFres/RMSEhard.csv'],final)  % change name depending on hard or soft experiment
        csvwrite(['data/' char(folder(ds)) '/PMFres/timeshard.csv'],finalT) % change name depending on hard or soft experiment
        final
        finalT
        ind=ind+1;
    end
    
end
