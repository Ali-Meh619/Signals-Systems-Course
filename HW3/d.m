function im =d(img, x, y, r)

im = img;
[xn, yn] = size(im);

for i = 1:360
    
    xd = ceil(x + r*cosd(i));
    yd = ceil(y + r*sind(i));
    
    if xd <= xn && yd <= yn
        im(max(xd, 1), max(yd, 1)) = 1;
    end
    
end

end