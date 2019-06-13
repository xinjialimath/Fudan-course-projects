function [x,y]=draw_line(a,p1,p2,n)
 if (nargin<4)
     n=2;
 end
x=linspace(p1(1),p2(1),n);
y=linspace(p1(2),p2(2),n);
x=a*(x-637);
y=a*(y-590);
%plot(x,y,'b-')
end