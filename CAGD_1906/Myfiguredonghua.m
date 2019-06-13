myfigure=figure;
set(myfigure,'Visible','off');
set(gcf,'position',[100,300,700,600])
%hp=uipanel('Title','My_Panel','FontSize',20,'BackgroundColor','white','Position',[.2 .2 .6 .6]);
%inspect(hp)
axis([-637 637 -590 590]);
title('18210180025-李欣嘉');
%text(400,1100,'计算机辅助几何设计作业','FontSize',12);
%colormap cool;
%colorbar('location','eastoutside')
grid off;
box on;
zoom on;%用户可自动方法缩小
hold on;