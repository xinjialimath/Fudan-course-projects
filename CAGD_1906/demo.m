%���յ�demo����,ֱ�����к�ɵõ�����,֡���϶�,��Ҫ�ȴ�Լ1����
fmat=moviein(21+11+11+11+6+7+14+6+6+14+11);%��֡��
%fmat=moviein(51+51);
j=1;%֡������

%��У�յĹ���,21֡
[fmat,j]=donghuatest02(fmat,0.5,0);
%��С����11֡
for a=0.5:0.05:1
    fmat(:,j)=donghuatest2(a,0);
    j=j+1;
end

%��ת11֡
for sita=0:0.2*pi:2*pi
    fmat(:,j)=donghuatest2(a,sita);
    j=j+1;
end

%�����ַ�,20֡
[fmat,j]=donghuatest03(j,fmat,1,0);


%������ 11֡
for l=0:6:60
    fmat(:,j)=donghuatestfinalRD(a,sita,-l);
    j=j+1; 
end


%��+�ֱ۵��°ڶ� 11֡,����
for beta=0:0.01*pi:0.1*pi 
    l=beta/(0.005*pi)+60;
    fmat(:,j)=donghuatestfinal(a,sita,-beta,-l);
    j=j+1;
end

%��+�ֱۻָ�6֡,����
for beta=0.1*pi:-0.01*pi:0.05*pi 
    l=beta/(0.005*pi)+60-7*(0.1*pi-beta)/(0.035*pi);
    fmat(:,j)=donghuatestfinal(a,sita,-beta,-l);
    j=j+1;
end
%��+�ֱۿ�������7֡,�ٶȽϿ�
for beta=0.05*pi:-0.05*pi:-0.25*pi 
    l=beta/(0.005*pi)+60-7*(0.1*pi-beta)/(0.035*pi);
    fmat(:,j)=donghuatestfinal(a,sita,-beta,-l);
    j=j+1;
end

%�շ���,ͬʱ�ֱۻذ�,��14֡
nn=13;
for t=1:nn
     beta=-0.25*pi+(0.25*pi)*t/nn;
     fmat(:,j)=donghuatestfinalRZ(a,sita,-l,t,-beta);
    j=j+1;   
end

%��+�����½�����,6֡
t0=t;
 for gamma=0:0.02*pi:0.1*pi 
     t=t0+0.15*gamma/(0.02*pi);
     fmat(:,j)=donghuatestfinalYS(a,sita,-l,t,-beta,gamma);
     
     j=j+1;
 end
 
 
 %��+���ָֻ���״,6֡
 for gamma=0.1*pi:-0.02*pi:0
     t=t0+0.15*gamma/(0.02*pi);
     fmat(:,j)=donghuatestfinalYS(a,sita,-l,t,-beta,gamma);
     
     j=j+1;
 end
 
 %�շɻ�,14֡
 for t=13:-1:1
     beta=-0.25*pi+(0.25*pi)*t/13;
     fmat(:,j)=donghuatestfinalRZ(a,sita,-l,t,-beta);
    j=j+1;   
 end

 %���������ס��,11֡
 for beta=-0.25*pi:0.035*pi:0.1*pi %11֡
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