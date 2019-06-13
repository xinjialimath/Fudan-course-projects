function [Bezier_spline_3_x,Bezier_spline_3_y]=draw_Bezier_spline_3_N(input,w)
x=input(:,1);
y=input(:,2);
t=0:0.1:1;
B03=(1-t).^3;
B13=3*t.*(1-t).^2;
B23=3*t.^2.*(1-t);
B33=t.^3;
% Bezier_spline_3_x=(w(1)*B03*x(1)+w(2)*B13*x(2)+w(3)*B23*x(3)+w(4)*B33*x(4))./(w(1)*B03+w(2)*B13+w(3)*B23+w(4)*B33);
% Bezier_spline_3_y=(w(1)*B03*y(1)+w(2)*B13*y(2)+w(3)*B23*y(3)+w(4)*B33*y(4))./(w(1)*B03+w(2)*B13+w(3)*B23+w(4)*B33);
Bezier_spline_3_x=(B03*x(1)+B13*x(2)+B23*x(3)+B33*x(4));
Bezier_spline_3_y=(B03*y(1)+B13*y(2)+B23*y(3)+B33*y(4));
plot(Bezier_spline_3_x,Bezier_spline_3_y);
hold on 
end