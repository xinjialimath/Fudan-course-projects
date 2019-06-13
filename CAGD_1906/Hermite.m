function [x,y]=Hermite(a,x0,f0,df0,x1,f1,df1,n)
 if (nargin<8)
     n=10;
 end
x=linspace(x0,x1,n);
y=f0*(1+2*(x-x0)./(x1-x0)).*(((x-x1)/(x0-x1)).^2)+f1*(1+2*(x-x1)./(x0-x1)).*(((x-x0)/(x1-x0)).^2)+df0.*(x-x0).*(((x-x1)/(x0-x1)).^2)+df1*(x-x1).*(((x-x0)/(x1-x0)).^2);
x=a*(x-637);
y=a*(y-590);
plot(x,y,'b-')
end