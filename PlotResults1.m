function  [R,MSE,MAE,RMSE]=PlotResults1(t,y,name)

    % T &Y
    figure;
    subplot(2,2,1);
    plot(y,'-ob');
    hold on;
    plot(t,'-sr');
    legend('Outputs','Targets');
    title(name);

    % Corrolation Plot
    subplot(2,2,2);
    plot(y,t,'ko');
    hold on;
    xmin=min(min(t),min(y));
    xmax=max(max(t),max(y));
    plot([xmin,xmax],[xmin,xmax],'b','LineWidth',2 );
    R=corr(t,y);
    title(['R=' num2str(R)]);



    % Error
    subplot(2,2,3);
    e=t-y;
    plot(e,'b');
    legend('Error');
    MSE= mse(e);
    MAE = mae(e);
    RMSE = sqrt(MSE);
    title(['MSE = ' num2str(MSE) ' ,RMSE = ' num2str(RMSE) ' ,MAE = ' num2str(MAE)]);

    subplot(2,2,4);
    histfit(e,50);
    eMean=mean(e);
    eStd=std(e); 
    title(['\mu =  ' num2str(eMean) ',\sigma =' num2str(eStd)]);


end