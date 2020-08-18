
function h = circle(x,y,r)
hold on
th = 0:pi/50:2*pi;
x_ = r * cos(th) + x;
y_ = r * sin(th) + y;
h = plot(x_, y_,'linewidth',1.5);
hold off
end