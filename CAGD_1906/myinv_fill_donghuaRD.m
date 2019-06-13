%�յ�ֱ��ֱ�²���
%����Bezier_spline3�Ŀ��Ƶ�
%ϣ�������7����,ʹ���x,y
%PΪҪ���������,��ΪN��,�����BΪ���Ƶ�ΪN+2��.
function [return_x,return_y]=myinv_fill_donghuaRD(l,P)
%lΪ����ƽ�Ƶĸ���,��y���ӵĸ���
return_x=[];
return_y=[];
P_len=length(P(:,1));%ѡ���ĸ���
x=(P(:,1)-637);
y=(P(:,2)-590+l);
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