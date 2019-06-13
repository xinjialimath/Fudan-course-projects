%最终的demo代码,直接运行后可得到动画,帧数较多,需要等待约1分钟
fmat=moviein(21+11+11+11+6+7+14+6+6+14+11);%总帧数
%fmat=moviein(51+51);
j=1;%帧计数器

%画校徽的过程,21帧
[fmat,j]=donghuatest02(fmat,0.5,0);
%从小到大11帧
for a=0.5:0.05:1
    fmat(:,j)=donghuatest2(a,0);
    j=j+1;
end

%旋转11帧
for sita=0:0.2*pi:2*pi
    fmat(:,j)=donghuatest2(a,sita);
    j=j+1;
end

%出现字符,20帧
[fmat,j]=donghuatest03(j,fmat,1,0);


%日下落 11帧
for l=0:6:60
    fmat(:,j)=donghuatestfinalRD(a,sita,-l);
    j=j+1; 
end


%日+手臂的下摆动 11帧,较慢
for beta=0:0.01*pi:0.1*pi 
    l=beta/(0.005*pi)+60;
    fmat(:,j)=donghuatestfinal(a,sita,-beta,-l);
    j=j+1;
end

%日+手臂恢复6帧,较慢
for beta=0.1*pi:-0.01*pi:0.05*pi 
    l=beta/(0.005*pi)+60-7*(0.1*pi-beta)/(0.035*pi);
    fmat(:,j)=donghuatestfinal(a,sita,-beta,-l);
    j=j+1;
end
%日+手臂快速上升7帧,速度较快
for beta=0.05*pi:-0.05*pi:-0.25*pi 
    l=beta/(0.005*pi)+60-7*(0.1*pi-beta)/(0.035*pi);
    fmat(:,j)=donghuatestfinal(a,sita,-beta,-l);
    j=j+1;
end

%日飞起,同时手臂回摆,共14帧
nn=13;
for t=1:nn
     beta=-0.25*pi+(0.25*pi)*t/nn;
     fmat(:,j)=donghuatestfinalRZ(a,sita,-l,t,-beta);
    j=j+1;   
end

%日+右手下降缓冲,6帧
t0=t;
 for gamma=0:0.02*pi:0.1*pi 
     t=t0+0.15*gamma/(0.02*pi);
     fmat(:,j)=donghuatestfinalYS(a,sita,-l,t,-beta,gamma);
     
     j=j+1;
 end
 
 
 %日+右手恢复形状,6帧
 for gamma=0.1*pi:-0.02*pi:0
     t=t0+0.15*gamma/(0.02*pi);
     fmat(:,j)=donghuatestfinalYS(a,sita,-l,t,-beta,gamma);
     
     j=j+1;
 end
 
 %日飞回,14帧
 for t=13:-1:1
     beta=-0.25*pi+(0.25*pi)*t/13;
     fmat(:,j)=donghuatestfinalRZ(a,sita,-l,t,-beta);
    j=j+1;   
 end

 %左手伸出接住日,11帧
 for beta=-0.25*pi:0.035*pi:0.1*pi %11帧
    l=beta/(0.005*pi)+60-7*(0.1*pi-beta)/(0.035*pi);
    fmat(:,j)=donghuatestfinal(a,sita,-beta,-l);
    j=j+1;
end

% aviobj=VideoWriter('example0.avi');
% open(aviobj);
% for i=1:(21+11+11+11+6+7+13+6+6+13+11)
% writeVideo(aviobj,fmat(:,i));
% end
% close(aviobj);
movie(fmat,5);