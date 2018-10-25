%=============================== Start ====================================
clc
clear
close all;
%=============================== Data =====================================

load SemnanFiroozkooh.mat;
Data=SemnanFiroozkooh;
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

%============================ Train and Predict============================

Mdl2 = fitrsvm(Xtr , Ytr ,'KernelFunction','gaussian','Kernelscale',...
    'auto','Boxconstraint',10e+4);

ytrfit = predict(Mdl2 , Xtr);
ytstfit = predict(Mdl2 , Xts);
yallfit = predict(Mdl2 , XN);

%================================= Result =================================

[R,MSE,MAE,RMSE] = PlotResults1(Yts,ytstfit,'Test Data');
[R,MSE,MAE,RMSE] = PlotResults1(Ytr,ytrfit,'Train Data');
[R,MSE,MAE,RMSE] = PlotResults1(YN,yallfit,'All Data');