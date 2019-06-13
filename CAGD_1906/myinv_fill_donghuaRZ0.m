%�յ���ת����
%����Bezier_spline3�Ŀ��Ƶ�
%ϣ�������7����,ʹ���x,y
%PΪҪ���������,��ΪN��,�����BΪ���Ƶ�ΪN+2��.
%tΪ��ǰ֡��,��13֡
%xx,yyΪԭͼ�ϵ���ת��,���յ�����
function [return_x,return_y]=myinv_fill_donghuaRZ0(l,xx,yy,t,P)
%lΪ����ƽ�Ƶĸ���,��y���ӵĸ���
return_x=[];
return_y=[];
P_len=length(P(:,1));%ѡ���ĸ���
xx=xx-637;
yy=yy-590+l;%���³�ʼ����ת��
%����ƽ��
x0=(P(:,1)-637)-18*t;
y0=(P(:,2)-590+l)+22*t;
xx=xx-18*t;
yy=yy+22*t;
%ƽ�ƽ���

x1=x0-xx;
y1=y0-yy;
r=sqrt(x1.^2+y1.^2);
sita=t*pi/13;%13֡��ת��180��
sita0=atan(y1./x1);%ԭʼ��x��������ʱ����ת�Ƕ�
sita1=sita0-sita+pi;%��֧����ת��ĽǶ�
x=xx+r.*cos(sita1);
y=yy+r.*sin(sita1);
%��ת����



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