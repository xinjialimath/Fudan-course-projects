%日的旋转操作
%反算Bezier_spline3的控制点
%希望反解出7个点,使其过x,y
%P为要过点的坐标,若为N个,则输出B为控制点为N+2个.
%t为当前帧数,共13帧
%xx,yy为原图上的旋转点,即日的中心
function [return_x,return_y]=myinv_fill_donghuaRZ0(l,xx,yy,t,P)
%l为向下平移的个数,即y增加的个数
return_x=[];
return_y=[];
P_len=length(P(:,1));%选择点的个数
xx=xx-637;
yy=yy-590+l;%重新初始化旋转点
%先做平移
x0=(P(:,1)-637)-18*t;
y0=(P(:,2)-590+l)+22*t;
xx=xx-18*t;
yy=yy+22*t;
%平移结束

x1=x0-xx;
y1=y0-yy;
r=sqrt(x1.^2+y1.^2);
sita=t*pi/13;%13帧旋转完180度
sita0=atan(y1./x1);%原始按x正方形逆时针旋转角度
sita1=sita0-sita+pi;%按支点旋转后的角度
x=xx+r.*cos(sita1);
y=yy+r.*sin(sita1);
%旋转结束



S=zeros(P_len+2,P_len+2);
S(1,1)=-1;
S(1,3)=1;
S(P_len+2,P_len)=-1;
S(P_len+2,P_len+2)=1;
for i=2:P_len+1
    S(i,i-1)=1;
    S(i,i)=4;
    S(i,i+1)=1;
end

K=zeros(P_len+2,2);
for i=2:P_len+1
    K(i,1)=6*x(i-1);
    K(i,2)=6*y(i-1);
end

B=inv(S)*K;
for i=1:length(P(:,1))-1
    [draw_x,draw_y]=draw_Bezier_spline_3_donghua(B(i:i+3,:));
    return_x = [return_x draw_x];
    return_y = [return_y draw_y];
end

end