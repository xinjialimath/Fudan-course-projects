function [Bezier_spline_3_x,Bezier_spline_3_y]=draw_Hermite_spline_3_donghua(input)
x=input(:,1);
y=input(:,2);
BS3_a0=x(1)+4*x(2)+x(3);
BS3_a1=-3*x(1)+3*x(3);
BS3_a2=3*x(1)-6*x(2)+3*x(3);
BS3_a3=-x(1)+3*x(2)-3*x(3)+x(4);

BS3_b0=y(1)+4*y(2)+y(3);
BS3_b1=-3*y(1)+3*y(3);
BS3_b2=3*y(1)-6*y(2)+3*y(3);
BS3_b3=-y(1)+3*y(2)-3*y(3)+y(4);

t=0:0.02:1;
Bezier_spline_3_x=(BS3_a0+BS3_a1*t+BS3_a2*(t.^2)+BS3_a3*(t.^3))/6;
Bezier_spline_3_y=(BS3_b0+BS3_b1*t+BS3_b2*(t.^2)+BS3_b3*(t.^3))/6;
plot(Bezier_spline_3_x,Bezier_spline_3_y,'b-')
% hold on

end