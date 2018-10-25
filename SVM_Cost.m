function Cost = SVM_Cost( XX , Xtr , Ytr )
%SVM_COST Summary of this function goes here
%   Detailed explanation goes here

Cost = zeros(size(XX,1),1);

for ii = 1:size(XX,1)
    X = XX(ii,:);
    Mdl = fitrsvm(Xtr , Ytr ,'Standardize',true,'KernelFunction','gaussian','KernelScale'...
        , X(1:1) , 'Epsilon' , X(2:2) , 'BoxConstraint' , X(3:3));
    YFit = predict(Mdl , Xtr);
    
    
%     YtrArray = table2array(Ytr);
    C = sqrt(mse(YFit - Ytr));
    Cost(ii,1) = C;    
end

end

