%=============================== Start ====================================
clc
clear

close all;
%================= Load Data & Define Input and Output ====================

load ShahroodDamghanWithoutPCI.mat;
Data=ShahroodDamghanWithoutPCI;
X = Data(:,2:end);
Y = Data(:,1);

InputNum = size(X,2);
OutputNum = size(Y,2);

%============================ Normalization ===============================

MinX = min(X);
MaxX = max(X);

MinY = min(Y);
MaxY = max(Y);

XN = X;
YN = Y;

for ii = 1:InputNum
    XN(:,ii) = Normalize_Fcn(X(:,ii),MinX(ii),MaxX(ii));
end

for ii = 1:OutputNum
    YN(:,ii) = Normalize_Fcn(Y(:,ii),MinY(ii),MaxY(ii));
end

DataNum = size(X,1);

%====================== Split Data For Train & Test =======================

TrPercent = 80;
TrNum = round(DataNum * TrPercent / 100);
TsNum = DataNum - TrNum;

R = randperm(DataNum);
trIndex = R(1 : TrNum);
tsIndex = R(1+TrNum : end);

Xtr = XN(1:trIndex,:);
Ytr = YN(1:trIndex,:);

Xts = XN(1+tsIndex:end,:);
Yts = YN(1+tsIndex:end,:);


%============================= Statements =================================

TotalNum = 3;
VarLow = 100;
VarHigh = 1000;
f = 0; % Exit Flag
e = 0.0001; % terminate condition

%=============================== Noises ===================================

% Mean and variance of initial state.
muC = zeros(TotalNum,1);
varC = 1e-1*eye(TotalNum);

% Mean and variance of process noise.
muU = zeros(TotalNum,1);
varU = 1e-1*eye(TotalNum);

% Mean and variance of measurement noise.
muV = 0.0;
varV = 1e-1;

T = 20; % Number of time steps.

%=========================== Primary Statements ===========================

% Sample the initial state (normally distributed).
C = abs(rand(1,TotalNum) * (VarHigh - VarLow) + VarLow);
W_best = zeros(1,TotalNum+1);

% Set the number of particles.
nPart = 8;
y_up = zeros(1,nPart);

% Sample the particles from the prior (initial) distribution.
CPart = abs(rand(TotalNum ,nPart));
wPart = (1/nPart)*ones(1,size(CPart,2));
    
    
Cost = SVM_Cost(C,Xtr,Ytr);
W_best = [C Cost];
if  (Cost<e)
   f = 1;
    
end


%============================== Particles =================================


i = 1;
while((f==0)&(i<=T))
    C(i+1,:) = abs(C(i,:) + (muV+(sqrt(varU)*randn(TotalNum,1)))');
    
    Cost = SVM_Cost(C(i+1,:),Xtr,Ytr);
    if (Cost<W_best(end))
        W_best = [C(i+1,:) Cost];
    end
    CPart_up = abs(CPart + repmat(muU,1,nPart)+(sqrt(varU)*randn(TotalNum,nPart)));
    
    for j=1:nPart
        y_up(j) = SVM_Cost(CPart_up(:,j)',Xtr,Ytr);
        wPart(j) =   normpdf(Cost,y_up(j),varV);
        if (y_up(j)<W_best(end))
        W_best = [CPart_up(:,j)' y_up(j)];
        end
    end
    
    
    % Normalize to form a probability distribution (i.e. sum to 1).
    wPart = wPart./sum(wPart);
    
    % Resampling: From this new distribution, now we randomly sample from
    % it to generate our new estimate particles
    for j = 1 : nPart
        CPart(j) = abs(CPart_up(find(rand <= cumsum(wPart),1)));
    end
    [t_val , t_inx] = min(y_up);

    
    if (y_up(t_inx)<e)
        f = 1;

    end

    i= i+1
    W_best(end)
    
end
%================================ Result ==================================

Mdl2 = fitrsvm(Xtr , Ytr ,'Standardize',true,'KernelFunction','gaussian', 'KernelScale' ,...
W_best(1:1) , 'Epsilon' , W_best(2:2) , 'BoxConstraint' , W_best(3:3));

ytrfit = predict(Mdl2 , Xtr);
ytstfit = predict(Mdl2 , Xts);
yallfit = predict(Mdl2 , XN);

YNN=[];
for ii = 1:OutputNum
    YNN(:,ii) = unormal(ytstfit(:,ii),MinY(ii),MaxY(ii));
end

[R,MSE,MAE,RMSE] = PlotResults1(Yts,ytstfit,'Test Data');
[R,MSE,MAE,RMSE] = PlotResults1(Ytr,ytrfit,'Train Data');
[R,MSE,MAE,RMSE] = PlotResults1(YN,yallfit,'All Data');

%================================== End ===================================
    [round(YNN) Y(1+tsIndex:end,:)]