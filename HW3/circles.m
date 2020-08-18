
function y = circles(img, min, maxx)

r1 = size(img, 1);
r2 = size(img, 2);
img2 = zeros(maxx, r1, r2);

for r = min:maxx
    
    kernel = zeros(2*r+1, 2*r+1);

    for t = 1:360
        x = r*cosd(t);
        y = r*sind(t);
        kernel(ceil(x)+r+1, ceil(y)+r+1) = 1 / r;
    end
    
    img1 = conv2(img, kernel, 'same');
    
    img2(r, :, :) = img1;
 
    
end

img2 = img2 / max(max(max(img2)));
y = img2;

end
