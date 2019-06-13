function [x,y]=Hermite_spline(x0,y0,dfx0,dfy0,x1,y1,dfx1,dfy1,n)
 if (nargin<9)
     n=10;
 end
t=0:0.1:1;
x=x0*(1-3*t.^2+2*t.^3)+x1*(3*t.^2-2*t.^3)+dfx0*(t-2*t.^2+t.^3)+dfx1*(-t.^2+t.^3);
y=y0*(1-3*t.^2+2*t.^3)+y1*(3*t.^2-2*t.^3)+dfy0*(t-2*t.^2+t.^3)+dfy1*(-t.^2+t.^3);
% plot(x,y,'b-')
% hold on
end