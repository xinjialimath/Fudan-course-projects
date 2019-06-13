%����Bezier_spline3�Ŀ��Ƶ�
%ϣ�������7����,ʹ���x,y
%PΪҪ���������,��ΪN��,�����BΪ���Ƶ�ΪN+2��.
function [return_x,return_y]=myinv_fill_donghua_Hermite_v(a,P)
a=1;
return_x=[];
return_y=[];
P_len=length(P(:,1));%ѡ���ĸ���
v=unifrnd (-0.1, 0.1, 1, P_len);
delta=1;%�ȼ�໮��ʱ����Ϊ1
x=a*(P(:,1)-637);
y=a*(P(:,2)-590);
S=zeros(P_len,P_len);
S(1,1)=1;
S(P_len,P_len)=1;
for i=2:P_len-1
    S(i,i-1)=1*delta;
    S(i,i)=4*delta+v(i);
    S(i,i+1)=1*delta;
end

K=zeros(P_len,2);
for i=2:P_len-1
    K(i,1)=3*(x(i+1)-x(i-1));
    K(i,2)=3*(y(i+1)-y(i-1));
end

B=inv(S)*K;
for i=1:length(P(:,1))-1
    [draw_x,draw_y]=Hermite_spline(x(i),y(i),B(i,1),B(i,2),x(i+1),y(i+1),B(i+1,1),B(i+1,2),10);
    %[draw_x,draw_y]=Hermite(1,x(i),y(i),sqrt(K(i,1)^2+K(i,2)^2),x(i+1),y(i+1),sqrt(K(i+1,1)^2+K(i+1,2)^2));
    return_x = [return_x draw_x];
    return_y = [return_y draw_y];
end

%end

